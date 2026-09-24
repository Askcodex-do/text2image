"""Desktop (windowed) launcher for AI Image Studio.

This is the entry point used by the windowed executable. A double-clicked build
shows a native window that:

* reports the local URL and output directory,
* opens the GUI in the default browser,
* shuts the server down cleanly when the window closes,

so closing the window never leaves an orphaned background process.

The Flask app is served with ``werkzeug.serving.make_server`` on a background
thread instead of ``app.run()``, which keeps shutdown under our control rather
than relying on signal handlers.  A second launch is detected through
:mod:`ai_image_studio.single_instance` and simply opens the running instance.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import webbrowser
from typing import Callable

from werkzeug.serving import make_server

from . import __version__
from .config import AppConfig, open_browser_enabled
from .single_instance import InstanceLock, is_port_serving

APP_TITLE = "AI Image Studio"


def _configure_logging(output_dir: str) -> None:
    """Send logs to a file instead of stderr.

    A windowed executable has no console, and writing to a missing stderr can
    raise during request handling.  Keeping a log file also gives users
    something to attach to a bug report.
    """
    handlers = []
    try:
        os.makedirs(output_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(output_dir, "app.log"), encoding="utf-8"))
    except Exception:
        pass
    if sys.stderr is not None and not getattr(sys, "frozen", False):
        handlers.append(logging.StreamHandler(sys.stderr))
    if not handlers:
        handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    for name in ("werkzeug", "flask", "ai_image_studio"):
        logger = logging.getLogger(name)
        logger.handlers = handlers
        logger.setLevel(logging.INFO)
        logger.propagate = False


def _display_host(host: str) -> str:
    if host in ("0.0.0.0", "::", ""):
        return "127.0.0.1"
    return host


class ServerThread(threading.Thread):
    """Serves the Flask app until :meth:`shutdown` is called."""

    def __init__(self, app, host: str, port: int) -> None:
        super().__init__(name="ai-image-studio-server", daemon=True)
        self._server = make_server(host, port, app, threaded=True)
        self.error: BaseException | None = None

    @property
    def port(self) -> int:
        return self._server.server_port

    def run(self) -> None:
        try:
            self._server.serve_forever()
        except BaseException as exc:  # surfaced to the launcher for a clean exit
            self.error = exc

    def shutdown(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass


def _parse_port(url: str) -> int:
    return int(url.rsplit(":", 1)[1])


def _open_browser_when_ready(
    url: str, stop: threading.Event, opener: Callable[[str], bool]
) -> None:
    """Open the GUI once the port is listening and the user has not beaten us to it."""
    port = _parse_port(url)
    for _ in range(150):  # up to ~30s
        if stop.wait(0.2):
            return
        if is_port_serving("127.0.0.1", port):
            break
    else:
        return
    if stop.wait(0.4):
        return
    try:
        opener(url)
    except Exception:  # a missing browser must never take the app down
        pass


def _wait_for_quit() -> None:
    """Console fallback: block until interrupted."""
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


def _message_box(text: str) -> None:
    """OS-level info box, used when no window toolkit is importable."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, APP_TITLE, 0x40)  # type: ignore[attr-defined]
    except Exception:
        print(text)


def run_windowed(subtitle: str, url: str, output_dir: str, on_quit: Callable[[], None]) -> bool:
    """Show a visible window. Returns False when Tkinter is unavailable."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        return False

    try:
        root = tk.Tk()
    except Exception:
        return False

    root.title(APP_TITLE)
    root.minsize(520, 290)
    root.configure(padx=22, pady=18)

    ttk.Label(root, text=APP_TITLE, font=("Segoe UI", 17, "bold")).pack(anchor="w")
    ttk.Label(root, text=subtitle, foreground="#555").pack(anchor="w", pady=(2, 14))

    ttk.Label(root, text="Address", font=("Segoe UI", 9, "bold")).pack(anchor="w")
    entry = ttk.Entry(root, width=48)
    entry.insert(0, url)
    entry.configure(state="readonly")
    entry.pack(fill="x", pady=(2, 12))

    if output_dir:
        ttk.Label(root, text="Generated images", font=("Segoe UI", 9, "bold")).pack(
            anchor="w"
        )
        ttk.Label(root, text=output_dir, wraplength=470, foreground="#555").pack(
            anchor="w", pady=(2, 12)
        )

    ttk.Label(
        root,
        text=(
            "The editor is open in your browser. Keep this window open while you "
            "work, and close it when you are finished."
        ),
        wraplength=470,
        foreground="#555",
    ).pack(anchor="w", pady=(0, 14))

    def copy_address() -> None:
        try:
            root.clipboard_clear()
            root.clipboard_append(url)
        except Exception:
            pass

    buttons = ttk.Frame(root)
    buttons.pack(anchor="w")
    ttk.Button(buttons, text="Open in Browser", command=lambda: webbrowser.open(url)).pack(
        side="left"
    )
    ttk.Button(buttons, text="Copy Address", command=copy_address).pack(
        side="left", padx=8
    )
    ttk.Button(buttons, text="Quit", command=on_quit).pack(side="left")

    root.update_idletasks()
    try:
        root.eval("tk::PlaceWindow . center")
    except Exception:
        pass
    # Closing the window is the supported way to stop the server.
    root.protocol("WM_DELETE_WINDOW", on_quit)
    root.mainloop()
    return True


def main() -> int:
    config = AppConfig.from_env()
    _configure_logging(config.output_dir)
    lock = InstanceLock()
    if not lock.acquire():
        _report_already_running(config)
        return 0

    from .app import create_app

    app = create_app(config)
    server = ServerThread(app, config.host, config.port)
    server.start()

    deadline = time.time() + 30
    while time.time() < deadline:
        if server.error is not None:
            raise server.error
        if is_port_serving("127.0.0.1", server.port, timeout=0.3):
            break
        time.sleep(0.1)
    if server.error is not None:
        raise server.error

    url = f"http://{_display_host(config.host)}:{server.port}"
    stop = threading.Event()

    def quit_() -> None:
        stop.set()
        try:
            import tkinter as tk

            if tk._default_root is not None:
                tk._default_root.destroy()
        except Exception:
            pass
        server.shutdown()

    subtitle = f"Local image editor and face-preserving pipeline \u2014 v{__version__}"
    if open_browser_enabled(default=True):
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url, stop, webbrowser.open),
            daemon=True,
        ).start()

    windowed = False
    if getattr(sys, "frozen", False):
        windowed = run_windowed(subtitle, url, config.output_dir, quit_)
        if not windowed:
            _message_box(
                f"{APP_TITLE} is running at {url}\n\n"
                f"Generated images: {config.output_dir}\n\n"
                "Close this dialog after you finish; the server stops when the "
                "process exits."
            )
    if not windowed:
        # Source runs (and any build without Tkinter) stay in the terminal.
        _wait_for_quit()

    server.shutdown()
    lock.release()
    return 0


def _report_already_running(config: AppConfig) -> None:
    url = f"http://{_display_host(config.host)}:{config.port}"
    # Focus the running instance first: a modal dialog would otherwise block the
    # browser from opening until the user dismisses it.
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            APP_TITLE,
            f"{APP_TITLE} is already running.\n\nOpened {url} in your browser.",
        )
        root.destroy()
    except Exception:
        print(f"{APP_TITLE} is already running: {url}")


if __name__ == "__main__":
    raise SystemExit(main())