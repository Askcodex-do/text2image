"""Single-instance guarding for the desktop launcher.

The launcher binds a dedicated loopback port as a lock. Binding, unlike a
lockfile, is released automatically by the OS when the process dies, so a crash
can never leave a stale lock behind. Stdlib only, no display required, so this
module is importable and testable on a headless machine.
"""

from __future__ import annotations

import socket

#: Default loopback port used purely as a mutex; it serves no traffic.
DEFAULT_LOCK_PORT = 47999


def is_port_serving(host: str, port: int, timeout: float = 0.6) -> bool:
    """True if something accepts a TCP connection at ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class InstanceLock:
    """An exclusive lock backed by a bound loopback socket.

    ``acquire()`` returns False when another process already holds the lock,
    which the launcher reports instead of starting a second server.
    """

    def __init__(self, port: int = DEFAULT_LOCK_PORT, host: str = "127.0.0.1") -> None:
        self.port = port
        self.host = host
        self._socket: socket.socket | None = None

    @property
    def held(self) -> bool:
        return self._socket is not None

    def acquire(self) -> bool:
        if self._socket is not None:
            return True
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen(1)
        except OSError:
            sock.close()
            return False
        self._socket = sock
        return True

    def release(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None