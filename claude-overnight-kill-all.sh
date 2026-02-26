#!/bin/bash
# Kill all Claude tmux sessions
sessions=$(tmux ls -F '#{session_name}' 2>/dev/null | grep '^claude-')

if [ -z "$sessions" ]; then
  echo "No claude sessions found."
  exit 0
fi

count=$(echo "$sessions" | wc -l)
echo "Killing $count claude session(s): $(echo $sessions | tr '\n' ' ')"

echo "$sessions" | while read -r s; do
  tmux kill-session -t "$s"
done

echo "Done."
