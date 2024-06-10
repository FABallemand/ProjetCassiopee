#!/bin/bash

# This script can be executed before executing a model training.
# It release a lot of computer resources that can be utilised for training.
# This script was used on a remote node before training models
# on the RGB-D Object dataset.

# Free resources
sudo killall python
sudo killall python3
sudo pkill -f .vscode
# sudo sync; echo 1 > /proc/sys/vm/drop_caches

# Restart GPU
# https://discuss.pytorch.org/t/cuda-fails-to-reinitialize-after-system-suspend/158108/3
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm