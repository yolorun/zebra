#!/bin/bash

DALL=false
REMOTE="v@100.79.4.66:/home/v/proj/zebra"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dall)
            DALL=true
            shift
            ;;
        *)
            REMOTE="$1"
            shift
            ;;
    esac
done

echo "Downloading models from: $REMOTE"

mkdir -p checkpoints

if [ "$DALL" = true ]; then
    echo "Downloading all checkpoints..."
    rsync -avz --progress --include="massive-rnn_*/" --include="massive-rnn_*/**" --exclude="*" "$REMOTE/checkpoints/" ./checkpoints/
else
    echo "Downloading only last.ckpt from each model..."
    rsync -avz --progress --include="massive-rnn_*/" --include="massive-rnn_*/last.ckpt" --exclude="*" "$REMOTE/checkpoints/" ./checkpoints/
fi

echo "Download complete."
