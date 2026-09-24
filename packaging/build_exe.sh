#!/usr/bin/env bash
# Cross-build the Windows executable on Linux using Wine + an embeddable
# Windows Python. Verified on Debian 13 with Wine 10.0.
#
#   1. Installs wine64 if missing.
#   2. Downloads the embeddable Python and adds site/pip.
#   3. Installs the pinned Windows dependency set.
#   4. Runs PyInstaller with the project spec.
#
# Usage:  bash packaging/build_exe.sh [python-version]
set -euo pipefail

PYVER="${1:-3.12.8}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${EXE_BUILD_DIR:-$HOME/.ai_image_studio-build}"
EMBED="$WORK/pyembed"

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
  "Pillow>=10"
  "pyinstaller==6.22.3"
)

command -v wine >/dev/null 2>&1 || {
  echo "wine is not installed; installing wine64..."
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends wine64 wine
}

if [ ! -x "$EMBED/python.exe" ]; then
  echo "==> Preparing embeddable Windows Python $PYVER in $EMBED"
  mkdir -p "$WORK"
  curl -fsSL -o "$WORK/python-embed.zip" \
    "https://www.python.org/ftp/python/$PYVER/python-$PYVER-embed-amd64.zip"
  mkdir -p "$EMBED"
  python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    "$WORK/python-embed.zip" "$EMBED"
fi

# The embeddable distribution ships without site-packages on the path.
# Rewrite its ._pth so `import site` and Lib\site-packages take effect.
MM="${PYVER%.*}"                       # 3.12
ZIPNAME="python${MM/./}.zip"           # python312.zip
PTH="$(ls "$EMBED"/python3*._pth | head -1)"
printf '%s\n.\nLib\\site-packages\nimport site\n' "$ZIPNAME" > "$PTH"

if ! wine "$EMBED/python.exe" -m pip --version >/dev/null 2>&1; then
  echo "==> Bootstrapping pip"
  curl -fsSL -o "$WORK/get-pip.py" https://bootstrap.pypa.io/get-pip.py
  wine "$EMBED/python.exe" "$WORK/get-pip.py" --no-warn-script-location
fi

echo "==> Installing Windows build dependencies"
wine "$EMBED/python.exe" -m pip install --no-warn-script-location "${DEPS[@]}"

echo "==> Building executable"
rm -rf "$WORK/build" "$WORK/dist"
wine "$EMBED/python.exe" -m PyInstaller "$ROOT/packaging/ai_image_studio.spec" \
  --noconfirm --distpath "$WORK/dist" --workpath "$WORK/build"

echo
echo "Built: $WORK/dist/ai_image_studio.exe"
ls -lh "$WORK/dist/ai_image_studio.exe"