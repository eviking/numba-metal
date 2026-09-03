#!/usr/bin/env bash
# Build and validate a distributable numba-metal release: sdist + wheel,
# `twine check`, then a full install-and-smoke-test cycle in a throwaway
# clean virtualenv (never the developer's own .venv), so a release is
# only produced after proving it installs and runs from scratch with no
# leftover state from the development environment.
#
# Usage: scripts/build_release.sh
#
# Exits non-zero on the first failure. Requires: python3, a working
# Apple-silicon Metal device for the Metal-specific smoke-test steps
# (skipped with a clear message if unavailable, matching this project's
# "explicit skip, never silent fallback" convention elsewhere).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== 1/7: Cleaning previous build artifacts ==="
rm -rf dist/ build/ ./*.egg-info src/*.egg-info
find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true

echo "=== 2/7: Building sdist + wheel (python -m build) ==="
python3 -m pip install --quiet --upgrade build twine >/dev/null
python3 -m build

echo "=== 3/7: Checking distribution metadata (twine check) ==="
python3 -m twine check dist/*

echo "=== 4/7: Verifying distribution contents (no venv/caches/macOS metadata) ==="
FORBIDDEN_PATTERN='(^|/)(\.venv|\.git|\.DS_Store|__MACOSX|__pycache__|\.pytest_cache|\.ruff_cache|\.numba_cache)(/|$)|\.nbi$|\.nbc$|\.metallib$'
DIST_LISTING_FILE="$(mktemp)"
trap 'rm -f "$DIST_LISTING_FILE"' EXIT

for artifact in dist/*.whl; do
  echo "--- wheel contents: $artifact ---"
  python3 -m zipfile -l "$artifact" | tee -a "$DIST_LISTING_FILE"
done
for artifact in dist/*.tar.gz; do
  echo "--- sdist contents: $artifact ---"
  tar tzf "$artifact" | tee -a "$DIST_LISTING_FILE"
done

if grep -Eq "$FORBIDDEN_PATTERN" "$DIST_LISTING_FILE"; then
  echo "FAIL: distribution contains forbidden paths (venv/cache/macOS metadata):"
  grep -E "$FORBIDDEN_PATTERN" "$DIST_LISTING_FILE"
  exit 1
fi
echo "OK: no forbidden paths found in sdist or wheel."

echo "=== 5/7: Installing wheel into a clean, throwaway virtualenv ==="
CLEAN_VENV="$(mktemp -d)/clean-install-venv"
python3 -m venv "$CLEAN_VENV"
# shellcheck disable=SC1091
source "$CLEAN_VENV/bin/activate"
python3 -m pip install --quiet --upgrade pip
WHEEL_FILE="$(ls dist/*.whl | head -n1)"
python3 -m pip install --quiet "$WHEEL_FILE"

echo "=== 6/7: Smoke-testing the clean install ==="
echo "--- import check ---"
python3 -c "import numba_metal; print('numba_metal', numba_metal.__version__)"

echo "--- capability check ---"
python3 -c "
from numba_metal import metal
available = metal.is_available()
print('metal.is_available():', available)
if available:
    info = metal.get_device_info()
    print('device:', info.name)
else:
    print('No Metal device on this machine; skipping GPU-dependent smoke steps.')
"

METAL_AVAILABLE="$(python3 -c "from numba_metal import metal; print(metal.is_available())")"

if [ "$METAL_AVAILABLE" = "True" ]; then
  echo "--- vector-add example ---"
  python3 "$REPO_ROOT/examples/vector_add.py"
else
  echo "SKIP: vector-add example (no Metal device available in this environment)."
fi

deactivate

echo "=== 7/7: Running unit test suite against the repo checkout (not the installed wheel) ==="
if [ -x "$REPO_ROOT/.venv/bin/pytest" ]; then
  "$REPO_ROOT/.venv/bin/pytest" "$REPO_ROOT/tests" -q --ignore="$REPO_ROOT/tests/differential" -m "not metal"
  if [ "$METAL_AVAILABLE" = "True" ]; then
    echo "--- Metal integration tests ---"
    "$REPO_ROOT/.venv/bin/pytest" "$REPO_ROOT/tests" -q --ignore="$REPO_ROOT/tests/differential" -m metal
  else
    echo "SKIP: Metal integration tests (no Metal device available)."
  fi
else
  echo "SKIP: repo .venv not found; run 'pytest tests/' manually to validate the checkout separately from this release build."
fi

rm -rf "$(dirname "$CLEAN_VENV")"

echo ""
echo "=== Release build validated ==="
echo "Distributions in dist/:"
ls -la dist/
