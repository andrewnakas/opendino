#!/usr/bin/env bash
# hf_upload_retry.sh <repo> <local path> <path in repo>
# Upload to a HF dataset repo, retrying on transient errors (concurrent commits from many runners
# can collide). Exits non-zero if every attempt fails, so the CI job is marked as failed.
set -u
repo="$1"; src="$2"; dst="$3"
for attempt in 1 2 3 4 5; do
  if hf upload "$repo" "$src" "$dst" --repo-type dataset --commit-message "upload $dst"; then
    exit 0
  fi
  echo "upload attempt $attempt failed; retrying" >&2
  sleep $((RANDOM % 60 + 30))
done
echo "upload of $src to $repo/$dst failed after 5 attempts" >&2
exit 1
