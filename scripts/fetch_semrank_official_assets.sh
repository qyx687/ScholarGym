#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="$(cd "$REPO_DIR/../third_party/SemRank/classifier" && pwd)"
CHECKPOINT="$ASSET_DIR/topic_classifier_specter2.pt"
PARTIAL="$CHECKPOINT.part"
URL='https://www.dropbox.com/scl/fi/tzg189k3n6tfxr2lzvjqj/topic_classifier_specter2.pt?rlkey=hnp2kfkxezubqeblpq4ym8kkd&st=btgz2a4s&dl=1'

if [[ ! -f "$ASSET_DIR/labels.txt" ]]; then
  echo "Official labels are missing from $ASSET_DIR/labels.txt" >&2
  exit 2
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  curl --fail --location --retry 5 --continue-at - \
    --output "$PARTIAL" "$URL"
  mv "$PARTIAL" "$CHECKPOINT"
fi

ls -lh "$CHECKPOINT" "$ASSET_DIR/labels.txt"
sha256sum "$CHECKPOINT" "$ASSET_DIR/labels.txt"

PYTHON_BIN="${SCHOLARGYM_PYTHON:-/home/quan/miniconda3/envs/scholargym-official/bin/python}"
"$PYTHON_BIN" -c \
  "from huggingface_hub import snapshot_download; print(snapshot_download('allenai/specter2_base', revision='3447645e1def9117997203454fa4495937bfbd83'))"
