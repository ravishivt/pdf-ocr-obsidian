#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
pip install -r requirements.txt -q
(sleep 1 && open http://127.0.0.1:5200/) &
python app.py
