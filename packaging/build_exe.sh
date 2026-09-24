#!/usr/bin/env bash
# Cross-build the windowed Windows executable on Linux using Wine.
#
# Verified on Debian 13 with Wine 10.0.
#
#  1. Installs wine64, 7z, cabextract and msitools if missing.
#  2. Builds a Windows Python by extracting the official CPython installer's
#     payload (not the embeddable zip, which ships without Tcl/Tk).
#  3. Installs the pinned Windows dependency set.
#  4. Runs PyInstaller with the project spec (windowed by default).
#
# Usage:
#   bash packaging/build_exe.sh [python-version]
#
# Environment:
#   EXE_BUILD_DIR   workspace for the download/extract/build (default ~/.ai_image_studio-build)
#   AIS_CONSOLE=1   build the console variant instead of the windowed GUI build
set -euo pipefail

# Tkinter is required for the GUI window and only ships with the full CPython
# installer, so 3.10.11 is the pinned toolchain version.
PYVER="${1:-3.10.11}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${EXE_BUILD_DIR:-$HOME/.ai_image_studio-build}"
PYDIR="$WORK/pywin"
EXE_NAME="ai_image_studio.exe"

export WINEPREFIX="$WORK/wineprefix"
export WINEARCH=win64
export WINEDEBUG=-all
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg}"
mkdir -p "$XDG_RUNTIME_DIR"

# numpy<2 keeps the runtime free of symbols Wine's ucrtbase lacks; cv2 4.10 is
# the matching wheel that still ships the Haar cascade data. On a real Windows
# host newer versions work fine -- these pins exist only for the Wine cross-build.
DEPS=(
  "Flask>=3.0"
  "numpy==1.26.4"
  "opencv-python-headless==4.10.0.84"
  "pyinstaller==6.22.3"
)

have() { command -v "$1" >/dev/null 2>&1; }

if ! have wine || ! have 7z || ! have cabextract || ! have msiextract; then
  echo "==> Installing build prerequisites (wine64, 7z, cabextract, msitools)"
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wine64 wine p7zip-full cabextract msitools
fi

# Copy the Tcl/Tk runtime next to the interpreter, where CPython and PyInstaller
# both expect it (Lib\tkinter plus tcl/ and DLLs/_tkinter.pyd + tcl86t/tk86t).
extract_python() {
  echo "==> Building Windows Python $PYVER in $PYDIR"
  mkdir -p "$WORK"
  local installer="$WORK/python-$PYVER-amd64.exe"
  [ -f "$installer" ] || curl -fsSL -o "$installer" \
    "https://www.python.org/ftp/python/$PYVER/python-$PYVER-amd64.exe"

  # The installer bootstrapper is 32-bit and needs WoW64, so its payload is
  # extracted directly instead of run. The MSIs inside the embedded cabinet are
  # 64-bit and portable.
  local unpack="$WORK/unpack"
  rm -rf "$unpack" && mkdir -p "$unpack"
  # 7z usually exposes the payload files directly; older p7zip does not, so fall
  # back to carving the embedded cabinet and letting cabextract list it.
  if ! 7z x -y "$installer" "-o$unpack" >/dev/null 2>&1 || [ ! -f "$unpack/a0" ]; then
    local cab="$WORK/python-$PYVER-payload.cab"
    python3 - "$installer" "$cab" <<'PY'
import sys

src, dst = sys.argv[1], sys.argv[2]
data = open(src, "rb").read()
offset = data.find(b"MSCF")  # cabinet magic
if offset < 0:
    raise SystemExit("no embedded cabinet found")
with open(dst, "wb") as fh:
    fh.write(data[offset:])
print(offset)
PY
    cabextract -q -d "$unpack" "$cab"
  fi

  mkdir -p "$PYDIR"
  # a0=core dlls  a2=python.exe  a6=Lib  a14=tkinter+tcl/tk
  local msi
  for msi in "$unpack"/a0 "$unpack"/a2 "$unpack"/a6 "$unpack"/a14; do
    [ -f "$msi" ] || { echo "missing payload component: $msi" >&2; exit 1; }
    msiextract -C "$PYDIR" "$msi" >/dev/null 2>&1 || true
  done
  mkdir -p "$PYDIR/Lib/site-packages"

  wine "$PYDIR/python.exe" -c "import sys, tkinter; print('python', sys.version.split()[0], 'tk', tkinter.TkVersion)"
}

if [ ! -x "$PYDIR/python.exe" ] || ! wine "$PYDIR/python.exe" -c "import tkinter" >/dev/null 2>&1; then
  extract_python
fi

if ! wine "$PYDIR/python.exe" -m pip --version >/dev/null 2>&1; then
  echo "==> Bootstrapping pip"
  curl -fsSL -o "$WORK/get-pip.py" https://bootstrap.pypa.io/get-pip.py
  wine "$PYDIR/python.exe" "$WORK/get-pip.py" --no-warn-script-location
fi

echo "==> Installing Windows build dependencies"
wine "$PYDIR/python.exe" -m pip install --no-warn-script-location "${DEPS[@]}"

if [ "${AIS_CONSOLE:-0}" != "0" ]; then
  echo "==> Building CONSOLE executable"
else
  echo "==> Building windowed (GUI) executable"
fi
rm -rf "$WORK/build" "$WORK/dist"
wine "$PYDIR/python.exe" -m PyInstaller "$ROOT/packaging/ai_image_studio.spec" \
  --noconfirm --distpath "$WORK/dist" --workpath "$WORK/build"

echo
echo "Built: $WORK/dist/$EXE_NAME"
ls -lh "$WORK/dist/$EXE_NAME"