#!/bin/bash
cd "$(dirname "$0")"

# Check for pdfimages (poppler)
if ! command -v pdfimages &> /dev/null; then
    echo "ERROR: pdfimages not found. Install poppler first:"
    echo "  macOS:          brew install poppler"
    echo "  Ubuntu/Debian:  sudo apt-get install poppler-utils"
    exit 1
fi

source venv/bin/activate
pip install -r requirements.txt -q
(sleep 1 && open http://127.0.0.1:5200/) &
python app.py
