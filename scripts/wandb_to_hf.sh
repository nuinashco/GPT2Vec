#!/usr/bin/env bash

# Check if required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <RUN_ID> <VERSION>"
    echo "Example: $0 hrui1936 v9"
    exit 1
fi

# ─── CONFIG ────────────────────────────────────────────────────────────────
# Weights & Biases config
ENTITY="havlytskyi-thesis"                      
PROJECT="bidirectional-decoder"        

RUN_ID="$1"
VERSION="$2"
ARTIFACT="model-${RUN_ID}:${VERSION}"         # replace artifact name:version
                   

BASE_DIR="data/pretrained_models" # replace with your base directory

# Kaggle config for saving old versions to locan main folder
HF_REPO_ID="GPT2Vec"
HF_PATH="."


# ─── 1) DERIVE DESTINATION FOLDER FROM RUN NAME ────────────────────────────
RUN_NAME=$(python3 <<EOF
import wandb, re
ENTITY="${ENTITY}"
PROJECT="${PROJECT}"
RUN_ID="${RUN_ID}"
api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
name = run.name or run.id
# slugify: replace non‑alphanumeric with underscore
slug = re.sub(r"\W+", "_", name)
print(slug)
EOF
)

DEST_DIR="${BASE_DIR}/${RUN_ID}_${VERSION}_${RUN_NAME}"

# # ─── 2) PREP ───────────────────────────────────────────────────────────────
# echo ">>> Preparing directory structure"
# rm -rf "$DEST_DIR"
# mkdir -p "$DEST_DIR"

# ─── 3) DOWNLOAD ARTIFACT ─────────────────────────────────────────────────
echo ">>> Downloading artifact into $DEST_DIR"
wandb artifact get "$ENTITY/$PROJECT/$ARTIFACT" --root "$DEST_DIR"


# ─── 5) UPLOAD TO HUGGINGFACE ──────────────────────────────────────────────
echo ">>> Uploading to Hugging Face"
huggingface-cli upload $HF_REPO_ID $DEST_DIR "${RUN_NAME}/" --commit-message "${RUN_ID}:${VERSION}" --private


echo "RUN_NAME=${RUN_NAME}"