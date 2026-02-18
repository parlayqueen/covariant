#!/data/data/com.termux/files/usr/bin/sh
set -e

echo "Running Covariant test suite..."
for f in tests/test_*.py; do
    echo "-> $f"
    python "$f"
done

echo "ALL TESTS PASSED"
