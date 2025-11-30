#!/usr/bin/env bash

# Default memory to 64G if no input is provided
MEMORY="${1:-64}G"

sinteractive -p interactive --time=01:00:00 --cpus-per-task=8 --mem "$MEMORY"
