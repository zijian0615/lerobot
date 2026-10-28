"""Slot booking for the cell: organizers open 30-minute slots, teams book one for an approved request.

runs/calendar.json: {"open": ["2026-10-21T14:00", ...], "bookings": {"<request id>": "2026-10-21T14:00"}}. Times are
the start of a slot, in the cell's local time (America/Chicago). A slot holds up to CAPACITY requests from different
teams (one from SINGLE_FROM on); a request holds one slot (booking another moves it). A team books with a pending (or approved) request; the
organizers then approve or reject it on the calendar, and a rejection frees its slot. Only future slots.
"""
from __future__ import annotations

import collections
import json
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Chicago")
SLOT_MIN = 30
CAPACITY = 2
# From this slot on, a slot holds one request (2026-10-28: one team at a time from Wednesday).
SINGLE_FROM = "2026-10-28T00:00"
DAYS_AHEAD = 28
SLOT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:(00|30)$")
LOOKUPS_PER_HOUR = 60  # per source IP: request ids are short, so nobody may try them one after another


def now() -> datetime:
    return datetime.now(TZ)


def parse(t: str) -> datetime:
    if not isinstance(t, str) or not SLOT.match(t):
        raise ValueError("a slot is YYYY-MM-DDTHH:MM, on the hour or the half hour")
    return datetime.fromisoformat(t).replace(tzinfo=TZ)


class Calendar:
    def __init__(self, root: Path):
        self.path = root / "calendar.json"
        self.lock = threading.Lock()
        self.recent: dict[str, list[float]] = collections.defaultdict(list)
        self.data = json.loads(self.path.read_text()) if self.path.is_file() else {"open": [], "bookings": {}}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)

    def limit(self, ip: str) -> None:
        """Lookups and bookings by request id, per source IP."""
        t = time.time()
        with self.lock:
            self.recent[ip] = [x for x in self.recent[ip] if t - x < 3600] + [t]
            if len(self.recent[ip]) > LOOKUPS_PER_HOUR:
                raise RuntimeError("too many requests from this address; try again later")

    @staticmethod
    def capacity(t: str) -> int:
        return 1 if t >= SINGLE_FROM else CAPACITY

    def _taken(self) -> collections.Counter:
        return collections.Counter(self.data["bookings"].values())

    def view(self) -> dict:
        """Every open slot from now on, with how many places are left (no team names)."""
        start = (now() - timedelta(minutes=SLOT_MIN)).strftime("%Y-%m-%dT%H:%M")
        with self.lock:
            taken = self._taken()
            slots = [{"t": t, "left": max(0, self.capacity(t) - taken[t])} for t in sorted(self.data["open"]) if t >= start]
        return {"tz": "America/Chicago", "slot_min": SLOT_MIN, "capacity": CAPACITY, "now": now().strftime("%Y-%m-%dT%H:%M"),
                "slots": slots}

    def booking(self, request_id: str) -> str | None:
        return self.data["bookings"].get(request_id)

    def book(self, req: dict, t: str) -> str:
        if req["status"] not in ("pending", "approved"):
            raise ValueError(f"request {req['id']} was {req['status']}: submit again to book a slot")
        when = parse(t)
        if when <= now():
            raise ValueError("that slot has already started")
        with self.lock:
            if t not in self.data["open"]:
                raise ValueError("that slot is not open for booking")
            others = [r for r, s in self.data["bookings"].items() if s == t and r != req["id"]]
            if len(others) >= self.capacity(t):
                raise ValueError("that slot is full")
            self.data["bookings"][req["id"]] = t
            self._save()
        return t

    def release(self, request_id: str) -> None:
        """Free a request's slot, if it has one (a rejected request)."""
        with self.lock:
            if self.data["bookings"].pop(request_id, None) is not None:
                self._save()

    def cancel(self, request_id: str) -> None:
        with self.lock:
            if self.data["bookings"].pop(request_id, None) is None:
                raise ValueError(f"request {request_id} has no booking")
            self._save()

    # ---- organizers ----
    def set_open(self, times: list[str], is_open: bool) -> dict:
        """Open or close slots; a slot with bookings stays open (cancel those first)."""
        if not isinstance(times, list) or len(times) > 2000:
            raise ValueError("times must be a list of slots")
        for t in times:
            parse(t)
            if parse(t) > now() + timedelta(days=DAYS_AHEAD * 4):
                raise ValueError("that slot is too far ahead")
        with self.lock:
            taken, opened, kept = self._taken(), set(self.data["open"]), []
            for t in times:
                if is_open:
                    opened.add(t)
                elif taken[t]:
                    kept.append(t)
                else:
                    opened.discard(t)
            self.data["open"] = sorted(opened)
            self._save()
        return {"open": len(self.data["open"]), "kept_booked": kept}

    def org_view(self, requests_q, store) -> dict:
        """Every slot from the start of this week, open or booked, with who booked it."""
        start = (now() - timedelta(days=now().weekday())).strftime("%Y-%m-%dT00:00")
        with self.lock:
            open_ = [t for t in sorted(self.data["open"]) if t >= start]
            bookings = dict(self.data["bookings"])
        out = []
        for rid, t in sorted(bookings.items(), key=lambda kv: kv[1]):
            if t < start:
                continue
            try:
                req = requests_q.get(rid)
            except KeyError:
                req = {"id": rid, "submission": {}, "status": "?"}
            sub = req.get("submission", {})
            out.append({"t": t, "request_id": rid, "team": sub.get("team", "?"), "policy": sub.get("policy", "?"),
                        "run_id": req.get("run_id"), "status": req.get("status"), "rollouts": sub.get("rollouts", 10)})
        return {**self.view(), "open": open_, "bookings": out}
