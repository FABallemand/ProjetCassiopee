#!/bin/bash

# Free resources
sudo killall python
sudo killall python3
pkill -f .vscode
# sudo sync; echo 1 > /proc/sys/vm/drop_caches

# Restart GPU
# https://discuss.pytorch.org/t/cuda-fails-to-reinitialize-after-system-suspend/158108/3
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm