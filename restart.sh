#!/bin/bash
cd ~/pose-demo
pkill -f 'python3 server.py' 2>/dev/null
sleep 2
export PATH=$HOME/.deno/bin:$HOME/.local/bin:$PATH
nohup python3 server.py > server.log 2>&1 &
sleep 5
tail -5 server.log
curl -s -o /dev/null -w 'HTTP %{http_code}\n' http://localhost:5001/
