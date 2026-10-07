#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/src"
python3 01_build.py
python3 02_run.py
python3 03_evaluate.py
python3 04_report.py
python3 07_impact.py
python3 08_feedback.py
python3 05_figures.py
python3 06_slides.py
python3 09_final_slides.py
