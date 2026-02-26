#!/bin/bash
# Launch a persistent Claude Code tmux session (survives SSH disconnects)
n=0
while tmux has-session -t "claude-$n" 2>/dev/null; do
  ((n++))
done
echo "Launching claude-$n"
tmux new-session -d -s "claude-$n" "claude --dangerously-skip-permissions; bash"
tmux set-option -t "claude-$n" mouse on
tmux attach -t "claude-$n"
