#!/usr/bin/env bash
# Refuses to commit files that contain words from a local, untracked block list.
#
# The list lives in .git/info/blocked-words (one case-insensitive word per line), so it never
# ends up in the repository itself. If the file doesn't exist, the check passes.
set -euo pipefail

list="$(git rev-parse --git-dir)/info/blocked-words"
[[ -s "$list" ]] || exit 0
[[ $# -gt 0 ]] || exit 0

if matches="$(grep -H -I -n -i -F -w -f "$list" -- "$@")"; then
    echo "Commit blocked: staged files contain blocked words:" >&2
    echo "$matches" >&2
    exit 1
fi
exit 0
