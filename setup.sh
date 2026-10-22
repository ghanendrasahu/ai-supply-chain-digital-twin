#!/usr/bin/env bash
# Streamlit Cloud entry point.
#
# The UI imports the `supply_twin` package (src-layout), so it must be
# installed in editable mode before `streamlit run`.
set -euo pipefail
cd "$(dirname "$0")"

pip install -r requirements.txt
pip install --no-deps -e .

exec streamlit run src/supply_twin/ui/app.py
