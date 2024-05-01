#!/bin/bash

sudo su
sudo killall python
sudo killall python3
sudo sync; echo 1 > /proc/sys/vm/drop_caches
pkill -f .vscode