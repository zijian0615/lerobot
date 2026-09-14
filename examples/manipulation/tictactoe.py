# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tic-tac-toe on a named support (default: stand). Robot pieces are screws."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

EMPTY = "."
HUMAN = "X"
ROBOT = "O"

Cell = tuple[int, int]  # (row, col) 1-based, row1=far/+x, col1=left/−y
PerceiveFn = Callable[[], tuple[Mapping[str, Any], Mapping[str, Any]]]
ExecuteFn = Callable[[list[dict[str, Any]], Mapping[str, Any]], str]


def cell_name(cell: Cell) -> str:
    return f"r{cell[0]}c{cell[1]}"


def empty_board() -> list[list[str]]:
    return [[EMPTY, EMPTY, EMPTY] for _ in range(3)]


def winner(board: Sequence[Sequence[str]]) -> str | None:
    lines = []
    for i in range(3):
        lines.append([board[i][0], board[i][1], board[i][2]])
        lines.append([board[0][i], board[1][i], board[2][i]])
    lines.append([board[0][0], board[1][1], board[2][2]])
    lines.append([board[0][2], board[1][1], board[2][0]])
    for line in lines:
        if line[0] != EMPTY and line[0] == line[1] == line[2]:
            return line[0]
    return None


def is_full(board: Sequence[Sequence[str]]) -> bool:
    return all(board[r][c] != EMPTY for r in range(3) for c in range(3))


def empties(board: Sequence[Sequence[str]]) -> list[Cell]:
    return [(r + 1, c + 1) for r in range(3) for c in range(3) if board[r][c] == EMPTY]


def _score(board: Sequence[Sequence[str]], depth: int) -> int | None:
    w = winner(board)
    if w == ROBOT:
        return 10 - depth
    if w == HUMAN:
        return depth - 10
    if is_full(board):
        return 0
    return None


def best_move(board: Sequence[Sequence[str]]) -> Cell | None:
    """Optimal O move. None if no legal cell."""
    options = empties(board)
    if not options:
        return None

    def minimax(state: list[list[str]], maximizing: bool, depth: int) -> int:
        done = _score(state, depth)
        if done is not None:
            return done
        if maximizing:
            best = -10_000
            for r, c in empties(state):
                state[r - 1][c - 1] = ROBOT
                best = max(best, minimax(state, False, depth + 1))
                state[r - 1][c - 1] = EMPTY
            return best
        best = 10_000
        for r, c in empties(state):
            state[r - 1][c - 1] = HUMAN
            best = min(best, minimax(state, True, depth + 1))
            state[r - 1][c - 1] = EMPTY
        return best

    def _pref(cell: Cell) -> int:
        if cell == (2, 2):
            return 0
        if cell in {(1, 1), (1, 3), (3, 1), (3, 3)}:
            return 1
        return 2

    work = [row[:] for row in board]
    chosen = options[0]
    best_s = -10_000
    for r, c in options:
        work[r - 1][c - 1] = ROBOT
        s = minimax(work, False, 1)
        work[r - 1][c - 1] = EMPTY
        if s > best_s or (s == best_s and _pref((r, c)) < _pref(chosen)):
            best_s = s
            chosen = (r, c)
    return chosen


def format_board(board: Sequence[Sequence[str]]) -> str:
    rows = ["  1 2 3   ← 列 (左→右 / image-left→right)", "远"]
    labels = ["1", "2", "3"]
    tags = ["上", "  ", "近"]
    for i, row in enumerate(board):
        rows.append(f"{labels[i]} {' '.join(row)}  {tags[i]}")
    rows.append("你=X  机器人=O")
    return "\n".join(rows)


def find_board(
    geometric: Mapping[str, Any],
    board_substr: str,
) -> tuple[str, Polygon] | None:
    needle = board_substr.lower()
    for obj in geometric.get("objects") or []:
        name = str(obj.get("name") or "")
        if needle in name.lower():
            fp = obj.get("footprint")
            if fp is not None and not getattr(fp, "is_empty", True):
                return name, fp
    return None


def occupied_cells(
    geometric: Mapping[str, Any],
    board: Polygon,
    board_name: str,
) -> dict[Cell, str]:
    """Map occupied 3×3 cells → object name (anything whose xy is on the board)."""
    minx, miny, maxx, maxy = board.bounds
    span_x = maxx - minx
    span_y = maxy - miny
    if span_x <= 0 or span_y <= 0:
        return {}
    out: dict[Cell, str] = {}
    skip = board_name.lower()
    for obj in geometric.get("objects") or []:
        name = str(obj.get("name") or "")
        if skip in name.lower() and "screw" not in name.lower():
            continue
        xy = obj.get("xy")
        if xy is None or len(xy) < 2:
            continue
        pt = Point(float(xy[0]), float(xy[1]))
        if not (board.contains(pt) or board.covers(pt) or board.intersects(pt.buffer(1e-4))):
            continue
        x, y = float(xy[0]), float(xy[1])
        row = int((maxx - x) / span_x * 3.0) + 1
        col = int((y - miny) / span_y * 3.0) + 1
        row = min(3, max(1, row))
        col = min(3, max(1, col))
        out[(row, col)] = name
    return out


def pick_offboard_piece(
    geometric: Mapping[str, Any],
    *,
    piece_substr: str,
    board: Polygon,
    board_name: str,
    arm: str | None = None,
) -> dict[str, Any] | None:
    needle = piece_substr.lower()
    skip = board_name.lower()
    cands: list[dict[str, Any]] = []
    for obj in geometric.get("objects") or []:
        name = str(obj.get("name") or "")
        if needle not in name.lower():
            continue
        if skip in name.lower() and needle not in skip:
            continue
        xy = obj.get("xy")
        if xy is None or len(xy) < 2:
            continue
        pt = Point(float(xy[0]), float(xy[1]))
        if board.contains(pt) or board.covers(pt):
            continue
        if arm:
            ws = obj.get("in_workspace")
            # geometric objects may not have in_workspace; caller can filter via symbolic
            if isinstance(ws, list) and ws and arm not in [str(a) for a in ws]:
                continue
        cands.append(dict(obj))
    if not cands:
        return None
    cx, cy = float(board.centroid.x), float(board.centroid.y)

    def _dist(o: Mapping[str, Any]) -> float:
        x, y = float(o["xy"][0]), float(o["xy"][1])
        return (x - cx) ** 2 + (y - cy) ** 2

    cands.sort(key=_dist, reverse=True)
    return cands[0]


def rename_piece(geometric: Mapping[str, Any], piece: Mapping[str, Any], new_name: str) -> dict[str, Any]:
    geo = dict(geometric)
    objs = []
    px, py = float(piece["xy"][0]), float(piece["xy"][1])
    for obj in geometric.get("objects") or []:
        row = dict(obj)
        ox, oy = float(row["xy"][0]), float(row["xy"][1])
        if abs(ox - px) < 1e-9 and abs(oy - py) < 1e-9:
            row["name"] = new_name
        objs.append(row)
    geo["objects"] = objs
    return geo


def plan_robot_move(
    *,
    piece_name: str,
    board_name: str,
    cell: Cell,
    arm: str,
) -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "arm": arm,
            "primitive": "Grasp",
            "args": {"object": piece_name},
            "depends_on": [],
        },
        {
            "step": 2,
            "arm": arm,
            "primitive": "Place",
            "args": {
                "object": piece_name,
                "destination": board_name,
                "region": cell_name(cell),
            },
            "depends_on": [1],
        },
    ]


def sync_human_from_scene(
    board: list[list[str]],
    occupied: Mapping[Cell, str],
) -> list[Cell]:
    """Mark newly occupied empty cells as human. Returns those cells."""
    found: list[Cell] = []
    for (r, c), _name in occupied.items():
        if board[r - 1][c - 1] == EMPTY:
            board[r - 1][c - 1] = HUMAN
            found.append((r, c))
    return found


def play_tictactoe(
    *,
    perceive: PerceiveFn,
    execute_plan: ExecuteFn,
    go_home: Callable[[], None] | None = None,
    board_substr: str = "stand",
    piece_substr: str = "screw",
    arm: str = "xarm",
    first: str = "human",
    dry_run: bool = False,
) -> int:
    """
    Interactive loop. You place a mark on the stand, press Enter; the robot
    puts a screw in a 3×3 cell (r1c1 = far-left).
    """
    board = empty_board()
    print("井字棋：stand 是棋盘（3×3）。你放棋子，机器人用桌上的螺丝回一手。")
    print("格子：r1 远/上，r3 近/下；c1 左，c3 右。")
    print(format_board(board))

    robot_to_move = first.strip().lower() in {"robot", "o", "robot_first"}

    while True:
        if not robot_to_move and not dry_run:
            try:
                input("你下完后按回车（机器人在 home 看着棋盘）… ")
            except EOFError:
                return 1

        symbolic, geometric = perceive()
        found = find_board(geometric, board_substr)
        if found is None:
            names = [str(o.get("name")) for o in (geometric.get("objects") or [])]
            print(f"没看到棋盘 {board_substr!r}。当前物体：{names}")
            return 1
        board_name, board_poly = found
        occ = occupied_cells(geometric, board_poly, board_name)
        new_human = sync_human_from_scene(board, occ)
        if new_human:
            logger.info("Human cells %s (occupied=%s)", new_human, occ)
        print(format_board(board))

        w = winner(board)
        if w == HUMAN:
            print("你赢了。")
            return 0
        if is_full(board):
            print("平局。")
            return 0

        if (
            not robot_to_move
            and not new_human
            and first.strip().lower() in {"human", "x"}
            and not dry_run
        ):
            print("棋盘上没有新的棋子，再放一枚然后回车。")
            continue

        cell = best_move(board)
        if cell is None:
            print("平局。")
            return 0

        piece = pick_offboard_piece(
            geometric,
            piece_substr=piece_substr,
            board=board_poly,
            board_name=board_name,
        )
        if piece is None:
            print(f"桌上没有可抓的 {piece_substr}（要在 stand 外面）。")
            return 1

        geo = rename_piece(geometric, piece, "ttt_piece")
        plan = plan_robot_move(
            piece_name="ttt_piece",
            board_name=board_name,
            cell=cell,
            arm=arm,
        )
        print(f"机器人下 {cell_name(cell)}（第{cell[0]}行第{cell[1]}列）抓 {piece.get('name')}")
        if dry_run:
            status = execute_plan(plan, geo)
            print(f"Dry-run bound status={status}")
            return 0 if status == "success" else 1

        status = execute_plan(plan, geo)
        if status != "success":
            print(f"机器人这一手失败：{status}")
            return 1
        board[cell[0] - 1][cell[1] - 1] = ROBOT
        print(format_board(board))
        if go_home is not None:
            go_home()

        w = winner(board)
        if w == ROBOT:
            print("机器人赢了。")
            return 0
        if is_full(board):
            print("平局。")
            return 0
        robot_to_move = False
