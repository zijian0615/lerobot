"""Egress proxy of the sandbox: the agent's only way out of its network.

    python egress_proxy.py --listen 0.0.0.0:3128 \
        --allow api.anthropic.com --allow pypi.org --allow files.pythonhosted.org

HTTPS only: a CONNECT to port 443 of an allowed host (or a subdomain of one)
is tunnelled; everything else gets 403. The agent's container sits on an
internal Docker network whose only other members are this proxy and the
development simulator, and reaches it through HTTPS_PROXY.
"""

from __future__ import annotations

import argparse
import asyncio

MAX_HEADER = 16 * 1024


def allowed(host: str, allow: list[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == a or host.endswith("." + a) for a in allow)


async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        w.close()


async def handle(cr: asyncio.StreamReader, cw: asyncio.StreamWriter, allow: list[str]):
    peer = cw.get_extra_info("peername")
    try:
        head = await asyncio.wait_for(cr.readuntil(b"\r\n\r\n"), 30)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError,
            asyncio.TimeoutError, ConnectionError):
        cw.close()
        return
    line = head.split(b"\r\n", 1)[0].decode("latin-1")
    parts = line.split()
    host, port = None, None
    if len(parts) == 3 and parts[0].upper() == "CONNECT":
        h, _, p = parts[1].rpartition(":")
        host, port = h.strip("[]"), p
    if host is None or port != "443" or not allowed(host, allow):
        print(f"[proxy] DENY {peer[0]} {line}", flush=True)
        cw.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        await cw.drain()
        cw.close()
        return
    try:
        ur, uw = await asyncio.wait_for(asyncio.open_connection(host, 443), 30)
    except (OSError, asyncio.TimeoutError):
        cw.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
        await cw.drain()
        cw.close()
        return
    print(f"[proxy] ALLOW {peer[0]} {host}", flush=True)
    cw.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await cw.drain()
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:3128")
    ap.add_argument("--allow", action="append", default=[])
    a = ap.parse_args()
    allow = [d.lower().lstrip(".") for d in a.allow]
    host, port = a.listen.rsplit(":", 1)
    srv = await asyncio.start_server(lambda r, w: handle(r, w, allow), host, int(port),
                                     limit=MAX_HEADER)
    print(f"[proxy] {a.listen}, allow: {', '.join(allow)}", flush=True)
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
