#!/bin/bash
# List and attach to an existing Claude tmux session
sessions=$(tmux ls -F '#{session_name}' 2>/dev/null | grep '^claude-')

if [ -z "$sessions" ]; then
  echo "No claude sessions found."
  exit 1
fi

echo "Available sessions:"
echo "$sessions"
echo ""
read -p "Enter session number: " num
tmux attach -t "claude-$num"
