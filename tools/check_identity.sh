#!/usr/bin/env bash
# Refuses to commit unless the author email is one of the allowed personal addresses.
# Run by pre-commit before every commit.
set -euo pipefail

allowed=(
    "157473246+jordimatagarcia@users.noreply.github.com"
)

email="${GIT_AUTHOR_EMAIL:-$(git config user.email || true)}"

for a in "${allowed[@]}"; do
    if [[ "${email,,}" == "${a,,}" ]]; then
        exit 0
    fi
done

cat >&2 <<EOF
Commit blocked: author email '${email:-<unset>}' is not an allowed personal address.
Set it for this repository with:
    git config user.email "${allowed[0]}"
EOF
exit 1
