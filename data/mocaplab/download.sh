#!/bin/bash

# pip3 install gdown

# https://unix.stackexchange.com/questions/183452/error-trying-to-unzip-file-need-pk-compat-v6-1-can-do-v4-6
# sudo apt-get install -y p7zip-full p7zip-rar

# nohup ./download.sh &

gdown --folder https://drive.google.com/drive/folders/1xos4pybtOfltFU0_YPhWLEyWKH5h6SoO
mkdir Cassiopée/Cassiopée
7za x Cassiopée/Cassiopée.zip -oCassiopée/Cassiopée
mkdir Cassiopée/Cassiopée_Allbones
7za x Cassiopée/Cassiopée_Allbones.zip -oCassiopée/Cassiopée_Allbones