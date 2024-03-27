# 📝 Notes

## 🛠️ Set-up
- https://vulkan.telecom-sudparis.eu/help/
- Python 3.8.15
- CUDA 11.5
- pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
- https://pytorch.org/blog/deprecation-cuda-python-support/
- https://pytorch.org/get-started/previous-versions/
- Activate Python virtual environment:
```bash
cd ../..
source start
cd dev/ProjetCassiopee
```

## 💾 Useful Commands
```bash
# Copy folder/file from local to server
scp -r /path/to/local/dir user@remotehost:/path/to/remote/dir

# Download Google Drive folder/file
gdown --folder link/to/folder -O /path/to/dir

nohup python3 -u main.py &

# https://stackoverflow.com/questions/17385794/how-to-get-the-process-id-to-kill-a-nohup-process

tar -xvf file.tar
tar -xzvf file.tar.gz
```

## 🗂️ Dataset
- [RGBD-SOD Dataset](https://www.kaggle.com/datasets/thinhhuynh3108/rgbdsod-set1)
- [How2Sign (A Large-scale Multimodal Dataset for Continuous American Sign Language)](https://paperswithcode.com/dataset/how2sign)
- [HIC (Hands in Action)](https://paperswithcode.com/dataset/hic)
- [RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset.html)
- [List of RGBD datasets](http://www.michaelfirman.co.uk/RGBDdatasets/)
- [Mocaplab Google Drive](https://drive.google.com/drive/folders/1xos4pybtOfltFU0_YPhWLEyWKH5h6SoO)
- [Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset](https://motion-x-dataset.github.io/)

## 🖥️ Code
- [PyTorch v1.12 Documentation](https://pytorch.org/docs/1.12/)
- [PyTorch Tutorials](https://github.com/yunjey/pytorch-tutorial/tree/master)
- [SimCLR Article](https://arxiv.org/pdf/2002.05709.pdf)
- [SimCLR in PyTorch](https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7)
- [SimCLR Tutorial](https://deeplearning.neuromatch.io/tutorials/W3D3_UnsupervisedAndSelfSupervisedLearning/student/W3D3_Tutorial1.html)
- [SimCLR Tutorial Videos](https://www.youtube.com/playlist?list=PLkBQOLLbi18NYb71nfD5gwwnZY4DPMCXu)

## 🧠 AI
- [MMAction](https://github.com/open-mmlab/mmaction2)
- [MMAction graph-based action recognition](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/2s-agcn/README.md)
- [Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action
Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf)

## 📋 ToDo
- Add cache files for datasets
- Adjust TSNE perplexity (depending of the number of samples)
- Log confusion matrix during training