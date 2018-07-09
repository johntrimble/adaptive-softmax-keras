#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SOURCE_DIR=$(dirname "$DIR")

exec docker run --runtime=nvidia \
                --rm \
                --interactive \
                --tty \
                --user $(id -u):$(id -g) \
                --env PYTHONPATH="\$PYTHONPATH:/code" \
                --mount type=bind,source="$SOURCE_DIR",target=/code \
                adaptive-softmax-keras \
                python3 examples/text8_benchmark.py "$@"
