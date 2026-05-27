#!/bin/bash
# Build the SimpleVLA-RL CUDA image, convert to an Enroot squashfs, and
# (optionally) upload to the H100 cluster.
#
# Usage (run from the openvla-oft-yhs repo root):
#   bash SimpleVLA-RL/docker/build_h100.sh                      # build + import
#   UPLOAD=1 REMOTE_HOST=h100-login REMOTE_DIR=/scratch/yuhang/sqsh \
#       bash SimpleVLA-RL/docker/build_h100.sh                  # build + import + scp

set -euo pipefail

# ----- Config (override via env) ---------------------------------------------
IMAGE_TAG="${IMAGE_TAG:-simplevla-rl-cuda:saga}"
SQSH_NAME="${SQSH_NAME:-simplevla-rl-cuda-saga.sqsh}"
DOCKERFILE="${DOCKERFILE:-SimpleVLA-RL/docker/Dockerfile.h100}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"

UPLOAD="${UPLOAD:-0}"
REMOTE_HOST="${REMOTE_HOST:-}"           # e.g. yuhang@h100-login.example
REMOTE_DIR="${REMOTE_DIR:-}"             # e.g. /scratch/yuhang/containers

# ----- Sanity checks ----------------------------------------------------------
if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found at $DOCKERFILE."
    echo "Run this script from the openvla-oft-yhs repo root."
    exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker is not installed / not on PATH."
    exit 1
fi
if ! command -v enroot >/dev/null 2>&1; then
    echo "Warning: enroot not on PATH on this host. Skipping local sqsh export."
    echo "         You can run 'enroot import dockerd://$IMAGE_TAG' on a host that has enroot,"
    echo "         or push to a registry and use 'enroot import docker://...' on the H100 side."
    SKIP_ENROOT=1
else
    SKIP_ENROOT=0
fi

# Pull base first (better progress UX if it isn't cached)
echo ">>> Pulling base image: $BASE_IMAGE"
docker pull "$BASE_IMAGE"

# ----- Docker build -----------------------------------------------------------
echo ">>> Building docker image: $IMAGE_TAG"
docker build \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t "$IMAGE_TAG" \
    -f "$DOCKERFILE" \
    .

# ----- Enroot import (squashfs) ----------------------------------------------
if [ "$SKIP_ENROOT" -eq 0 ]; then
    if [ -f "$SQSH_NAME" ]; then
        echo ">>> Removing existing $SQSH_NAME"
        rm -f "$SQSH_NAME"
    fi
    echo ">>> Exporting to enroot squashfs: $SQSH_NAME"
    enroot import -o "$SQSH_NAME" "dockerd://$IMAGE_TAG"
    ls -lh "$SQSH_NAME"
fi

# ----- Optional upload --------------------------------------------------------
if [ "$UPLOAD" = "1" ]; then
    if [ -z "$REMOTE_HOST" ] || [ -z "$REMOTE_DIR" ]; then
        echo "Error: UPLOAD=1 requires REMOTE_HOST and REMOTE_DIR."
        exit 1
    fi
    if [ ! -f "$SQSH_NAME" ]; then
        echo "Error: $SQSH_NAME not found (enroot import skipped or failed?)."
        exit 1
    fi
    echo ">>> Uploading $SQSH_NAME to $REMOTE_HOST:$REMOTE_DIR/"
    ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
    rsync -avh --progress "$SQSH_NAME" "$REMOTE_HOST:$REMOTE_DIR/"
    echo ">>> Done. On the H100 cluster, reference it as:"
    echo "      #SBATCH --container-image=$REMOTE_DIR/$SQSH_NAME"
fi

echo ">>> All done."
