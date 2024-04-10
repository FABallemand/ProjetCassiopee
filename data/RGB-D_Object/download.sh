#!/bin/bash

# nohup ./download.sh &

wget https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_full/README.txt
wget https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_full/rgbd-dataset_full.tar
tar -xvf rgbd-dataset_full.tar