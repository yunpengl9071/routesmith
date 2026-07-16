#!/usr/bin/env bash
set -e
fail=0
check_absent() {
  if grep -rn "$1" README.md docs/ --include='*.md' 2>/dev/null | grep -v ROADMAP | grep -v plans/; then
    echo "STALE CLAIM FOUND: $1"; fail=1
  fi
}
check_absent "27-dimensional"
check_absent "LinTS-27d"
check_absent "rs.complete("
check_absent "response.request_id"
check_absent "_detect_provider"
check_absent "LinTS-27d"
exit $fail
