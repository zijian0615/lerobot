"""Public relay for the cell's team API (Python standard library only; runs on our cloud VM).

    lab GB10 --(outbound SSH reverse tunnel)--> VM 127.0.0.1:9099 <-- relay_proxy.py <--HTTPS-- teams

Forwards /api/v1/* (teams, with their token) and /org/* (the site's Organizer view, with its password session) to
the tunnel's end; every other path (the operator API /admin and the cell's full /leaderboard included) gets 403, so
the arm is run and scored from the lab only.

Public pages, each served from the last copy while the tunnel is down:
    GET /, /leaderboard           the cell's /site: no leaderboard until the operator reveals it (--cache)
    GET /results/<key>.json       one team's runs, for its My results (--results; the cell also pushes them here)
    GET /results/<key>/rollouts/<run>/<trial>/<file>   that run's videos (--rollouts, pushed by the cell)
    GET /rollouts/<run>/<trial>/<file>                 any video, only once the leaderboard is revealed
    GET /viz/<key>/<run>/episode_<n>                   the dataset viewer (--viz: our lerobot-dataset-visualizer fork)
    GET /vizdata/download/<key>/<run>.tar               that run's LeRobotDataset in one file
    GET|HEAD /vizdata/datasets/<key>/<run>/resolve/main/<path>   that run's LeRobotDataset (--datasets, pushed by
                                                                 the cell), at the Hub's URL layout, with Range
TLS with --cert/--key (a self-signed pair is fine: teams pass it to run.py --ca).

    python3 relay_proxy.py --port 8443 --cert cert.pem --key key.pem
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import ssl
import tarfile
import sys
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 4 * 1024 * 1024
PASS_HEADERS = ("Authorization", "Content-Type", "Content-Length", "X-Request-Secret", "Cookie", "X-Requested-With")
PASS_BACK = ("Set-Cookie", "Cache-Control")  # the Organizer view's session cookie


SAFE = re.compile(r"^[A-Za-z0-9_-]+$")
KEY = re.compile(r"^[0-9a-f]{32}$")
KEY_ANY = re.compile(r"([0-9a-f]{6})[0-9a-f]{26}")  # view keys in the access log: first 6 characters only
REVEAL_TTL_S = 30
CONN_TIMEOUT_S = 30


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def handle_error(self, request, client_address):  # a failed handshake or a timeout: one line, not a traceback
        e = sys.exc_info()[1]
        if isinstance(e, (ssl.SSLError, OSError)):
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {client_address[0]} dropped: {type(e).__name__}", flush=True)
        else:
            super().handle_error(request, client_address)
STATIC_TYPES = {".zip": "application/zip", ".json": "application/json", ".md": "text/plain; charset=utf-8",
                ".pdf": "application/pdf", ".png": "image/png", ".step": "application/step", ".stl": "model/stl",
                ".jpg": "image/jpeg", ".usda": "application/octet-stream", ".mp4": "video/mp4"}


VIZ_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization",
           "proxy-authenticate", "host", "content-length", "x-forwarded-for"}
DATASET_FILE = re.compile(r"^(meta|data|videos)/[A-Za-z0-9_.\-/]+$")


def make_handler(upstream_host: str, upstream_port: int, files: dict[str, str] | None = None,
                 cache: Path = Path("site_cache.html"), static: Path | None = None,
                 rollouts: Path | None = None, results: Path = Path("results"),
                 datasets: Path | None = None, viz: tuple[str, int] | None = None, behind_proxy: bool = False,
                 redirect: str | None = None):
    files = files or {}  # exact URL path -> local file, for a one-off download (e.g. the submission package)
    results.mkdir(parents=True, exist_ok=True)
    reveal_file = cache.with_name(cache.stem + "_reveal.json")  # the cell's last answer, kept across restarts
    reveal = {"t": 0.0}

    def upstream(path: str, timeout: float = 10) -> tuple[int, bytes, str] | None:
        """GET from the cell: (status, body, content type), or None while it is unreachable."""
        conn = http.client.HTTPConnection(upstream_host, upstream_port, timeout=timeout)
        try:
            conn.request("GET", path)
            r = conn.getresponse()
            return r.status, r.read(), r.getheader("Content-Type", "application/octet-stream")
        except OSError:
            return None
        finally:
            conn.close()

    def revealed() -> bool:
        """Whether the operator has revealed the leaderboard (asked at most every REVEAL_TTL_S; hidden if unknown)."""
        if time.time() - reveal["t"] > REVEAL_TTL_S:
            reveal["t"] = time.time()
            got = upstream("/site.json", timeout=5)
            if got and got[0] == 200:
                reveal_file.write_bytes(got[1])
        try:
            return json.loads(reveal_file.read_text()).get("reveal") is True
        except (OSError, ValueError):
            return False

    def key_runs(key: str) -> set[str]:
        """The runs a view key opens: from the results the cell pushed, else asked from the cell."""
        if not KEY.match(key):
            return set()
        saved = results / f"{key}.json"
        if not saved.is_file():
            got = upstream(f"/results/{key}.json")
            if not got or got[0] != 200:
                return set()
            saved.write_bytes(got[1])
        try:
            return {r["run_id"] for r in json.loads(saved.read_text())["rows"]}
        except (OSError, ValueError, KeyError):
            return set()

    def rollout(parts: list[str]) -> Path | None:
        """rollouts/<run>/<trial>/<name>.mp4|json, or None: no other path can be reached."""
        if rollouts is None or len(parts) != 3 or not all(SAFE.match(x) for x in parts[:2]):
            return None
        stem, _, ext = parts[2].rpartition(".")
        f = rollouts / parts[0] / parts[1] / parts[2]
        return f if SAFE.match(stem) and ext in ("mp4", "json") and f.is_file() else None

    class Relay(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = CONN_TIMEOUT_S  # a client that stalls (in the TLS handshake too) loses its own thread, nothing else

        def log_message(self, fmt, *args):  # one access line per request, on stdout
            pass

        def _reply(self, code: int, body: bytes, ctype: str = "application/json",
                   extra: list[tuple[str, str]] | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            for k, v in extra or []:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _ip(self) -> str:
            """The client's address: with --behind-proxy, the one nginx sets in X-Real-IP (we listen on localhost only)."""
            if behind_proxy:
                real = self.headers.get("X-Real-IP", "").strip()
                if real:
                    return real
            return self.client_address[0]

        def _forward(self, method: str) -> None:
            t0 = time.time()
            status, size = 403, 0
            try:
                if redirect and not self.path.startswith("/api/v1/"):
                    # the site moved: send browsers there (308 keeps the method). /api/v1 still goes to the cell, since
                    # an HTTP client drops its Authorization header when a redirect changes the host.
                    status = 308
                    self._reply(308, b"", "text/plain", [("Location", redirect + self.path)])
                    return
                if method == "GET" and self.path in files:
                    with open(files[self.path], "rb") as f:
                        body = f.read()
                    status, size = 200, len(body)
                    self._reply(200, body, "application/octet-stream")
                    return
                if method == "GET" and self.path.split("?")[0].rstrip("/") in ("", "/leaderboard"):
                    status, size = self._leaderboard()
                    return
                if method == "GET" and self.path.startswith("/results/"):
                    status, size = self._results(self.path.split("?")[0].split("/")[2:])
                    return
                if method == "GET" and self.path.startswith("/vizdata/download/"):
                    status, size = self._download(self.path.split("?")[0].split("/")[3:])
                    return
                if method in ("GET", "HEAD") and self.path.startswith("/vizdata/datasets/"):
                    status, size = self._vizdata(self.path.split("?")[0].split("/")[3:])
                    return
                if method == "GET" and (self.path == "/viz" or self.path.startswith("/viz/")):
                    status, size = self._viz()
                    return
                if method == "GET" and self.path.startswith("/rollouts/"):
                    f = rollout(self.path.split("?")[0].split("/")[2:]) if revealed() else None
                    status, size = self._file(f)
                    return
                name = self.path.split("?")[0][len("/files/"):] if self.path.startswith("/files/") else None
                if method == "GET" and name is not None and static is not None:
                    # only a plain file name that is in the static folder: no subfolders, no "..", no hidden files
                    f = static / name
                    if "/" in name or name.startswith(".") or not f.is_file() or f.suffix not in STATIC_TYPES:
                        status, body = 404, b'{"error": "not found"}'
                        self._reply(404, body)
                    elif f.suffix == ".mp4":
                        status, size = self._ranged(f, "video/mp4", "public, max-age=3600")
                        return
                    else:
                        body = f.read_bytes()
                        status = 200
                        self._reply(200, body, STATIC_TYPES[f.suffix])
                    size = len(body)
                    return
                if not (self.path.startswith("/api/v1/") or self.path.startswith("/org/")):
                    body = b'{"error": "not available through the relay"}'
                    self._reply(403, body)
                    size = len(body)
                    return
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    status = 413
                    self._reply(413, b'{"error": "request too large"}')
                    return
                data = self.rfile.read(n) if n else None
                conn = http.client.HTTPConnection(upstream_host, upstream_port, timeout=30)
                headers = {k: self.headers[k] for k in PASS_HEADERS if self.headers.get(k)}
                headers["X-Forwarded-For"] = self._ip()  # set here, never passed through from the team
                try:
                    conn.request(method, self.path, body=data, headers=headers)
                    r = conn.getresponse()
                    out = r.read()
                    status, size = r.status, len(out)
                    self._reply(r.status, out, r.getheader("Content-Type", "application/json"),
                                [(k, v) for k, v in r.getheaders() if k in PASS_BACK])
                except OSError as e:  # tunnel down / cell server stopped
                    status = 502
                    self._reply(502, f'{{"error": "cell unreachable: {type(e).__name__}"}}'.encode())
                finally:
                    conn.close()
            finally:
                path = KEY_ANY.sub(r"\1…", self.path.split("?")[0])
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {self._ip()} {method} {path} "
                      f"{status} {size}B {1000 * (time.time() - t0):.0f}ms", flush=True)

        def _file(self, f: Path | None) -> tuple[int, int]:
            if f is None:
                body = b'{"error": "not found"}'
                self._reply(404, body)
                return 404, len(body)
            if f.suffix == ".mp4":
                return self._ranged(f, "video/mp4", "private, max-age=60")
            body = f.read_bytes()
            self._reply(200, body, "application/json")
            return 200, len(body)

        def _results(self, parts: list[str]) -> tuple[int, int]:
            """A team's runs (results/<key>.json) and their videos (results/<key>/rollouts/<run>/<trial>/<file>)."""
            key = parts[0].removesuffix(".json") if parts else ""
            if not KEY.match(key) or (len(parts) == 1) != parts[0].endswith(".json"):
                return self._file(None)
            saved = results / f"{key}.json"
            if len(parts) == 1:  # the cell's current copy; while it is unreachable, the last one it pushed or we saw
                got = upstream(f"/results/{key}.json")
                if got and got[0] == 200:
                    saved.write_bytes(got[1])
                elif got and got[0] == 404:
                    saved.unlink(missing_ok=True)
                    return self._file(None)
                elif not saved.is_file():
                    body = b'{"error": "cell unreachable"}'
                    self._reply(503, body)
                    return 503, len(body)
                return self._file(saved)
            if len(parts) != 5 or parts[1] != "rollouts":
                return self._file(None)
            if not saved.is_file():  # neither pushed nor seen yet: ask the cell which run this key opens
                got = upstream(f"/results/{key}.json")
                if not got or got[0] != 200:
                    return self._file(None)
                saved.write_bytes(got[1])
            try:
                runs = {r["run_id"] for r in json.loads(saved.read_text())["rows"]}
            except (OSError, ValueError, KeyError):
                runs = set()
            if parts[2] not in runs:  # this key opens only its own run's videos
                return self._file(None)
            f = rollout(parts[2:])
            if f is not None:
                return self._file(f)
            got = upstream("/" + "/".join(["results", *parts]), timeout=30)  # not pushed (yet): ask the cell
            if not got or got[0] != 200:
                return self._file(None)
            self._reply(200, got[1], got[2])
            return 200, len(got[1])

        def _ranged(self, f: Path, ctype: str, cache: str) -> tuple[int, int]:
            """Send a file, or the byte range the request asks for (video players seek with ranges; Safari needs them)."""
            size = f.stat().st_size
            start, end, code = 0, size - 1, 200
            m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start, end = int(m.group(1)), min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:  # the last N bytes
                    start = max(0, size - int(m.group(2)))
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return 416, 0
                code = 206
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", cache)
            self.send_header("Content-Length", str(end - start + 1))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return code, 0
            with open(f, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            return code, end - start + 1

        def _vizdata(self, parts: list[str]) -> tuple[int, int]:
            """<key>/<run>/resolve/main/<path>: a file of that run's dataset, if the key opens the run. Range-aware."""
            if datasets is None or len(parts) < 5 or parts[2:4] != ["resolve", "main"] or not SAFE.match(parts[1]):
                return self._file(None)
            rel = "/".join(parts[4:])
            if ".." in rel or not DATASET_FILE.match(rel) or parts[1] not in key_runs(parts[0]):
                return self._file(None)
            f = datasets / parts[1] / rel
            if not f.is_file():
                return self._file(None)
            ctype = {".json": "application/json", ".parquet": "application/octet-stream", ".mp4": "video/mp4"}.get(
                f.suffix, "application/octet-stream")
            return self._ranged(f, ctype, "private, max-age=60")

        def _download(self, parts: list[str]) -> tuple[int, int]:
            """<key>/<run>.tar: the run's LeRobotDataset as <run>/{meta,data,videos,manipbench.json}, streamed."""
            run = parts[1].removesuffix(".tar") if len(parts) == 2 and parts[1].endswith(".tar") else ""
            src = datasets / run if datasets is not None and SAFE.match(run) else None
            if src is None or not src.is_dir() or run not in key_runs(parts[0]):
                return self._file(None)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-tar")
            self.send_header("Content-Disposition", f'attachment; filename="{run}.tar"')
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            sent = [0]

            class Chunked:  # tarfile writes here; each write goes out as one HTTP chunk
                def write(_, b):
                    if b:
                        self.wfile.write(f"{len(b):x}\r\n".encode() + bytes(b) + b"\r\n")
                        sent[0] += len(b)
                    return len(b)

            with tarfile.open(fileobj=Chunked(), mode="w|", bufsize=1 << 20) as tar:
                for f in sorted(src.rglob("*")):
                    if f.is_file() and (f.name == "manipbench.json" or f.relative_to(src).parts[0] in ("meta", "data", "videos")):
                        tar.add(f, arcname=f"{run}/{f.relative_to(src)}", recursive=False)
            self.wfile.write(b"0\r\n\r\n")
            return 200, sent[0]

        def _viz(self) -> tuple[int, int]:
            """The dataset viewer: its static assets for everyone, a run's pages only with a key that opens it."""
            parts = [p for p in self.path.split("?")[0].split("/") if p][1:]  # after "viz"
            public = parts[:1] == ["_next"] or parts in (["lerobot-logo.png"], ["favicon.ico"])
            if viz is None or not (public or (len(parts) >= 2 and parts[1] in key_runs(parts[0]))):
                return self._file(None)
            conn = http.client.HTTPConnection(*viz, timeout=60)
            try:
                headers = {k: v for k, v in self.headers.items() if k.lower() not in VIZ_HOP}
                conn.request("GET", self.path, headers=headers)
                r = conn.getresponse()
                body = r.read()
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in VIZ_HOP:
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return r.status, len(body)
            except OSError as e:
                body = f'{{"error": "dataset viewer unavailable: {type(e).__name__}"}}'.encode()
                self._reply(502, body)
                return 502, len(body)
            finally:
                conn.close()

        def _leaderboard(self) -> tuple[int, int]:
            """The cell's public site; when the cell is unreachable, the last good copy."""
            got = upstream("/site")
            body = got[1] if got and got[0] == 200 else b""
            if body:
                cache.write_bytes(body)
            if not body and cache.exists():
                body = cache.read_bytes()
            if not body:
                body = b"<p>The leaderboard is not available yet.</p>"
            self._reply(200, body, "text/html; charset=utf-8")
            return 200, len(body)

        def do_GET(self):
            self._forward("GET")

        def do_POST(self):
            self._forward("POST")

        def do_HEAD(self):
            self._forward("HEAD")

    return Relay


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8443)
    p.add_argument("--upstream", default="127.0.0.1:9099", help="the SSH tunnel's end on this machine")
    p.add_argument("--cert")
    p.add_argument("--key")
    p.add_argument("--cache", type=Path, default=Path("site_cache.html"),
                   help="last copy of the site, served while the tunnel is down")
    p.add_argument("--rollouts", type=Path, help="folder of recorded trials (the cell pushes them), served read-only to the "
                   "team with the run's key, and to everyone once the leaderboard is revealed")
    p.add_argument("--behind-proxy", action="store_true",
                   help="behind a local reverse proxy (nginx) that ends TLS: take the client's address from X-Real-IP. "
                        "Only with --bind 127.0.0.1, so nobody else can set that header")
    p.add_argument("--datasets", type=Path, help="each run's LeRobotDataset (the cell rsyncs them), for the dataset viewer")
    p.add_argument("--viz", help="host:port of the dataset viewer (lerobot-dataset-visualizer, basePath /viz)")
    p.add_argument("--results", type=Path, default=Path("results"),
                   help="each run's results/<key>.json (the cell pushes them), served while the tunnel is down")
    p.add_argument("--static", type=Path, help="folder served read-only at /files/<name> (downloads for the site)")
    p.add_argument("--redirect-to", metavar="URL", help="the site moved here (e.g. https://hackathon.sidclab.org): "
                   "redirect everything but /api/v1, which still reaches the cell for clients that use this address")
    p.add_argument("--file", action="append", default=[], metavar="URL_PATH=LOCAL_FILE",
                   help="also serve one file at exactly this (unguessable) path; remove it after the download")
    args = p.parse_args()
    if args.behind_proxy and args.bind not in ("127.0.0.1", "localhost"):
        raise SystemExit("--behind-proxy trusts X-Real-IP: use it only with --bind 127.0.0.1")
    files = dict(f.split("=", 1) for f in args.file)
    host, port = args.upstream.rsplit(":", 1)
    srv = Server((args.bind, args.port), make_handler(
        host, int(port), files, args.cache, args.static, args.rollouts, args.results, args.datasets,
        (args.viz.rsplit(":", 1)[0], int(args.viz.rsplit(":", 1)[1])) if args.viz else None, args.behind_proxy,
        args.redirect_to.rstrip("/") if args.redirect_to else None))
    scheme = "http"
    if args.cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.cert, args.key)
        # The handshake runs in each connection's thread, under its timeout: done in accept(), one client that
        # connects and never speaks TLS (a scanner) blocks every other connection.
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True, do_handshake_on_connect=False)
        scheme = "https"
    elif args.bind not in ("127.0.0.1", "localhost"):
        print("[relay] WARNING: no --cert: team tokens cross the internet in clear text", file=sys.stderr)
    print(f"[relay] {scheme}://{args.bind}:{args.port}/api/v1/* -> http://{args.upstream} (everything else 403)",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
