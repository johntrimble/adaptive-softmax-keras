#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SOURCE_DIR=$(dirname "$DIR")

docker build -t adaptive-softmax-keras "$SOURCE_DIR"
