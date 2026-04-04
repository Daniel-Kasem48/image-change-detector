#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Installing Video Compare Desktop App"
echo "Project: $PROJECT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found. Please install Python 3.10+ first."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtual environment..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "==> Installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements-desktop.txt"

echo "==> Writing launcher script..."
cat > "$PROJECT_DIR/run_video_compare_desktop.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/video_compare_desktop.py"
EOF
chmod +x "$PROJECT_DIR/run_video_compare_desktop.sh"

DESKTOP_FILE_CONTENT="[Desktop Entry]
Version=1.0
Type=Application
Name=Video Change Detector
Comment=Compare two videos for scene changes
Exec=$PROJECT_DIR/run_video_compare_desktop.sh
Path=$PROJECT_DIR
Terminal=false
Categories=Utility;Graphics;
"

echo "==> Creating app launcher..."
mkdir -p "$HOME/.local/share/applications"
printf "%s" "$DESKTOP_FILE_CONTENT" > "$HOME/.local/share/applications/video-change-detector.desktop"
chmod +x "$HOME/.local/share/applications/video-change-detector.desktop"

if [ -d "$HOME/Desktop" ]; then
  printf "%s" "$DESKTOP_FILE_CONTENT" > "$HOME/Desktop/Video Change Detector.desktop"
  chmod +x "$HOME/Desktop/Video Change Detector.desktop"
fi

echo ""
echo "Done."
echo "You can now launch the app using either:"
echo "1) App menu: Video Change Detector"
echo "2) Desktop shortcut: Video Change Detector.desktop"
echo "3) Terminal: $PROJECT_DIR/run_video_compare_desktop.sh"
