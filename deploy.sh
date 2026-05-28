#!/usr/bin/env bash
set -e

pip install wheel twine
python setup.py sdist bdist_wheel
twine upload dist/*

rm -rf build
rm -rf dist
rm -rf deepfakes_hunter.egg-info
