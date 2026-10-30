#!/usr/bin/env python3
"""A tiny HTTPS (CONNECT) forward proxy that only lets through an allowlist of hosts, and logs every attempt.

The synthesis agent and the policy runs point HTTPS_PROXY/HTTP_PROXY here, so the only places they can
reach are the model API (to think) and the Python package index (to install public packages). Anything else
(web pages, GitHub, benchmark sites) is refused and recorded in the log, which the audit reads afterwards.

    python3 egress_proxy.py --socket PATH --log PATH [--allow-api]

It listens on a Unix socket, not a TCP port: the client side runs in a network namespace with no route to
the host (see netjail.py), and a socket file on a bound directory is the one channel across.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

PYPI = ("pypi.org", "files.pythonhosted.org")
# Exact hosts: the model API and the login-refresh endpoints. Not a domain suffix, which would also admit
# mcp-proxy.anthropic.com, the channel to the account's claude.ai connectors (mail, drive, docs).
API = ("api.anthropic.com", "console.anthropic.com", "platform.claude.com", "claude.ai")


def allowed(host: str, allow: tuple[str, ...]) -> bool:
    return host.lower().rstrip(".") in allow


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def handle(reader, writer, allow, log) -> None:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        writer.close()
        return
    first = head.split(b"\r\n", 1)[0].decode("latin-1")
    parts = first.split()
    method, target = (parts + ["", ""])[:2]
    if method.upper() == "CONNECT":
        host, _, port = target.rpartition(":")
    else:   # plain HTTP: only ever refused, logged so the audit sees it
        host, port = target.split("/")[2].split(":")[0] if "://" in target else target, "80"
    ok = method.upper() == "CONNECT" and allowed(host, allow)
    log.write(json.dumps({"t": time.time(), "method": method, "host": host, "port": port, "allowed": ok}) + "\n")
    log.flush()
    if not ok:
        writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    try:
        up_r, up_w = await asyncio.open_connection(host, int(port or 443))
    except OSError:
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await writer.drain()
    await asyncio.gather(pipe(reader, up_w), pipe(up_r, writer))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True, help="Unix socket path to listen on (netjail.py connects here)")
    ap.add_argument("--log", required=True)
    ap.add_argument("--allow-api", action="store_true", help="also allow the model API (synthesis); off for policy runs")
    args = ap.parse_args()
    allow = PYPI + (API if args.allow_api else ())
    log = open(args.log, "a")
    server = await asyncio.start_unix_server(lambda r, w: handle(r, w, allow, log), path=args.socket)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
