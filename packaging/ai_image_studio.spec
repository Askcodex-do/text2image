# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the AI Image Studio one-file executable.

Build (from the project root):

    pyinstaller packaging/ai_image_studio.spec --noconfirm

Notes
-----
* Flask discovers its templates and static files through filesystem paths, so
  they are collected as data and the Jinja loader is told to look inside the
  bundle (``sys._MEIPASS``) at runtime.
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

# GUI assets shipped inside the bundle.
datas = [
    (os.path.join(PACKAGE_DIR, "templates"), "ai_image_studio/templates"),
    (os.path.join(PACKAGE_DIR, "static"), "ai_image_studio/static"),
]

# Haar cascade data; the app checks sys._MEIPASS/cv2/data as a fallback.
datas += collect_data_files("cv2", includes=["data/*.xml"])

hiddenimports = collect_submodules("click")

a = Analysis(
    [os.path.join(PROJECT_ROOT, "run.py")],
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
        "tkinter",
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)