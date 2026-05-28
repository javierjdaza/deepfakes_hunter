#!/usr/bin/env bash
set -e

pip install --upgrade build twine
python -m build
twine upload dist/*

rm -rf build dist deepfakes_hunter.egg-info
