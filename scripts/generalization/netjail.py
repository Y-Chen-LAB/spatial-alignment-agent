#!/usr/bin/env python3
"""Run a command with no network except an allowlisting proxy (run inside the container).

    unshare --map-root-user --net -- python3 netjail.py SOCKET UID GID -- command ...

Inside a fresh network namespace there is no interface but loopback, so nothing can reach the outside
directly, whatever proxy settings it ignores. This brings loopback up, forwards 127.0.0.1:<port> to the host's
egress_proxy.py through a Unix socket (a file on a bound directory, so it crosses the namespace), sets
HTTPS_PROXY/HTTP_PROXY to that port, and runs the command. Bringing loopback up needs root in the namespace,
so the outer unshare maps the caller to root; the command itself runs in a nested user namespace as the
caller's own uid (UID GID), without capabilities (and the Claude CLI refuses to bypass permissions as root).
Standard library only.
"""
from __future__ import annotations

import fcntl
import os
import socket
import struct
import sys
import threading


def loopback_up() -> None:
    SIOCGIFFLAGS, SIOCSIFFLAGS, IFF_UP = 0x8913, 0x8914, 0x1
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ifr = struct.pack("16sH14s", b"lo", 0, b"")
    flags = struct.unpack("16sH14s", fcntl.ioctl(s, SIOCGIFFLAGS, ifr))[1]
    fcntl.ioctl(s, SIOCSIFFLAGS, struct.pack("16sH14s", b"lo", flags | IFF_UP, b""))
    s.close()


def relay(a: socket.socket, b: socket.socket) -> None:
    try:
        while data := a.recv(65536):
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def serve(listener: socket.socket, path: str) -> None:
    while True:
        conn, _ = listener.accept()
        up = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            up.connect(path)
        except OSError:
            conn.close()
            continue
        threading.Thread(target=relay, args=(conn, up), daemon=True).start()
        threading.Thread(target=relay, args=(up, conn), daemon=True).start()


def main() -> None:
    path, uid, gid, sep, *cmd = sys.argv[1:]
    assert sep == "--" and cmd, __doc__
    cmd = ["unshare", "--user", f"--map-user={uid}", f"--map-group={gid}", "--", *cmd]
    loopback_up()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(64)
    port = listener.getsockname()[1]
    proxy = f"http://127.0.0.1:{port}"
    env = dict(os.environ, HTTPS_PROXY=proxy, HTTP_PROXY=proxy, https_proxy=proxy, http_proxy=proxy,
               NO_PROXY="", no_proxy="")
    pid = os.fork()
    if pid == 0:
        listener.close()
        os.execvpe(cmd[0], cmd, env)
    threading.Thread(target=serve, args=(listener, path), daemon=True).start()
    _, status = os.waitpid(pid, 0)
    sys.exit(os.waitstatus_to_exitcode(status))


if __name__ == "__main__":
    main()
