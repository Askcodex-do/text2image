# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the AI Image Studio desktop executable.

Build (from the project root):

    pyinstaller packaging/ai_image_studio.spec --noconfirm

Produces a windowed (``console=False``) executable: double-clicking it opens a
native window and the browser, with no terminal window. Set ``AIS_CONSOLE=1``
to build the console variant instead:

    AIS_CONSOLE=1 pyinstaller packaging/ai_image_studio.spec --noconfirm

Notes
-----
* The entry point is ``run_desktop.py``, which shows a Tkinter window and
  shuts the server down when that window closes.  ``console=False`` means there
  is no stderr, so the launcher logs to ``AIImageStudio/app.log``.
* Tkinter is imported lazily inside a function (so a build without Tcl/Tk can
  fall back), which PyInstaller's static analysis cannot see; the Tk modules are
  therefore listed as hidden imports so the toolkit and its Tcl data still get
  bundled.
* Flask discovers templates and static files through filesystem paths, so they
  are collected as data and resolved from ``sys._MEIPASS`` at runtime.
* OpenCV's Haar cascade data is collected explicitly.  Detection is optional in
  the pipeline, but bundling the cascade means a frozen build behaves exactly
  like a source run instead of silently degrading to "no faces detected".
* ``AIImageStudio`` output never lives in the bundle; the app resolves a
  writable directory next to the executable (see ``config.default_output_dir``).
"""

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = os.path.dirname(os.path.abspath(SPECPATH))
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "ai_image_studio")

#: Console build when truthy; windowed (GUI) build otherwise.
CONSOLE = os.environ.get("AIS_CONSOLE", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
    "",
)

# GUI assets shipped inside the bundle.
datas = [
    (os.path.join(PACKAGE_DIR, "templates"), "ai_image_studio/templates"),
    (os.path.join(PACKAGE_DIR, "static"), "ai_image_studio/static"),
]

# Haar cascade data; the app checks sys._MEIPASS/cv2/data as a fallback.
datas += collect_data_files("cv2", includes=["data/*.xml"])

hiddenimports = collect_submodules("click")
# Bundle every application submodule so a newly added provider or service is
# never missed by static analysis.
hiddenimports += collect_submodules("ai_image_studio")
# Tkinter is imported lazily inside a function; make sure it is always bundled
# even though static analysis cannot see the import.
hiddenimports += [
    "tkinter",
    "tkinter.ttk",
    "tkinter.messagebox",
    "tkinter.filedialog",
    "tkinter.font",
]
hiddenimports += collect_submodules("werkzeug")

a = Analysis(
    [os.path.join(PROJECT_ROOT, "run_desktop.py")],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Never bundle test-only or interactive tooling into the shipped exe.
        "pytest",
        "pyinstaller",
        "IPython",
        "matplotlib",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ai_image_studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=CONSOLE,
    # Windowed builds raise a native error dialog with the traceback instead of
    # dying silently when there is no console to print to.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)