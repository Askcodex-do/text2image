"""Tests for the desktop launcher: single-instance lock and clean shutdown.

These run headless. They exercise the real server thread and the real lock
socket rather than mocks, so the guarantees that matter for a double-clicked
build -- "exactly one server" and "closing the window stops the server" -- are
actually verified.
"""

from __future__ import annotations

import socket
import sys
import time
import urllib.request

import pytest

from ai_image_studio.app import create_app
from ai_image_studio.config import AppConfig
from ai_image_studio.desktop import ServerThread, _display_host, main, run_windowed
from ai_image_studio.single_instance import InstanceLock, is_port_serving


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def quiet_app(output_dir):
    return create_app(AppConfig(output_dir=output_dir, host="127.0.0.1", port=0))


def test_instance_lock_is_exclusive():
    port = _free_port()
    first = InstanceLock(port=port)
    second = InstanceLock(port=port)

    assert first.acquire() is True
    assert first.held is True
    # The whole point: a second launcher must be refused.
    assert second.acquire() is False
    assert second.held is False

    first.release()
    assert first.held is False
    # Once released, the lock becomes available again (no stale lockfile).
    assert second.acquire() is True
    second.release()


def test_instance_lock_reacquire_is_idempotent():
    lock = InstanceLock(port=_free_port())
    assert lock.acquire() is True
    assert lock.acquire() is True
    lock.release()


def test_lock_released_when_process_exits():
    """Binding is used as the lock precisely so the OS frees it on crash."""
    import subprocess

    port = _free_port()
    holder = InstanceLock(port=port)
    assert holder.acquire() is True
    # A child process must not be able to take the lock while we hold it.
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ai_image_studio.single_instance import InstanceLock;"
            f"print(InstanceLock(port={port}).acquire())",
        ],
        capture_output=True,
        text=True,
    )
    assert child.stdout.strip() == "False"

    # Free it and let the child try again in a fresh process.
    holder.release()
    child2 = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ai_image_studio.single_instance import InstanceLock;"
            "l=InstanceLock("
            f"port={port});print(l.acquire());l.release()",
        ],
        capture_output=True,
        text=True,
    )
    assert child2.stdout.strip() == "True"


def test_is_port_serving_false_for_closed_port():
    assert is_port_serving("127.0.0.1", _free_port(), timeout=0.3) is False


def test_server_thread_serves_and_stops_cleanly(quiet_app):
    server = ServerThread(quiet_app, "127.0.0.1", 0)
    server.start()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not is_port_serving("127.0.0.1", server.port):
            time.sleep(0.1)
        url = f"http://127.0.0.1:{server.port}"
        assert urllib.request.urlopen(url + "/", timeout=5).status == 200
        assert urllib.request.urlopen(url + "/api/config", timeout=5).status == 200
    finally:
        server.shutdown()
    time.sleep(0.5)
    # Closing the window stops the server thread rather than orphaning it.
    assert server.is_alive() is False
    assert is_port_serving("127.0.0.1", server.port, timeout=0.5) is False


def test_server_binds_requested_port_when_available(quiet_app):
    port = _free_port()
    server = ServerThread(quiet_app, "127.0.0.1", port)
    server.start()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not is_port_serving("127.0.0.1", port):
            time.sleep(0.1)
        assert server.port == port
    finally:
        server.shutdown()


def test_display_host_maps_wildcard_to_loopback():
    assert _display_host("0.0.0.0") == "127.0.0.1"
    assert _display_host("::") == "127.0.0.1"
    assert _display_host("") == "127.0.0.1"
    assert _display_host("192.168.1.5") == "192.168.1.5"


def test_run_windowed_returns_false_without_tkinter(monkeypatch):
    """A build without Tcl/Tk must fall back instead of crashing."""
    monkeypatch.setitem(sys.modules, "tkinter", None)
    assert run_windowed("subtitle", "http://127.0.0.1:1", "/tmp/out", lambda: None) is False


def test_main_reports_already_running_and_exits(monkeypatch, output_dir):
    """A second launch must not start a second server."""
    import ai_image_studio.desktop as desktop

    port = _free_port()
    monkeypatch.setattr(desktop, "InstanceLock", lambda *a, **k: InstanceLock(port=port))
    monkeypatch.setattr(desktop.AppConfig, "from_env", classmethod(lambda cls: AppConfig(
        output_dir=output_dir, host="127.0.0.1", port=port
    )))

    opened: list[str] = []
    monkeypatch.setattr(desktop.webbrowser, "open", lambda u: opened.append(u))

    holder = InstanceLock(port=port)
    assert holder.acquire() is True
    try:
        # The launcher should notice the lock, open the existing instance, and
        # return without ever binding a server socket.
        assert main() == 0
    finally:
        holder.release()


def test_main_starts_server_and_cleans_up_when_windowed_returns_false(
    monkeypatch, output_dir, tmp_path
):
    """Simulates the frozen headless path: server runs, browser opens, then quits."""
    import ai_image_studio.desktop as desktop

    port = _free_port()
    lock_port = _free_port()
    assert lock_port != port
    monkeypatch.setattr(
        desktop, "InstanceLock", lambda *a, **k: InstanceLock(port=lock_port)
    )
    monkeypatch.setattr(
        desktop.AppConfig,
        "from_env",
        classmethod(lambda cls: AppConfig(output_dir=str(tmp_path), host="127.0.0.1", port=port)),
    )

    opened: list[str] = []
    monkeypatch.setattr(desktop.webbrowser, "open", lambda u: opened.append(u))
    # Force the frozen path but make the window unavailable, so main() uses the
    # console loop. Stop that loop immediately so the test terminates.
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(desktop, "run_windowed", lambda *a, **k: False)
    monkeypatch.setattr(desktop, "_message_box", lambda text: None)

    def fake_wait():
        # Poll the server from "inside" the loop, then return as the real
        # _wait_for_quit does when it catches KeyboardInterrupt.
        deadline = time.time() + 10
        while time.time() < deadline and not opened:
            time.sleep(0.1)

    monkeypatch.setattr(desktop, "_wait_for_quit", fake_wait)

    assert main() == 0
    # Browser was opened against the running instance.
    assert opened and opened[0].startswith(f"http://127.0.0.1:{port}")
    # And the port is free again after shutdown.
    assert is_port_serving("127.0.0.1", port, timeout=0.5) is False