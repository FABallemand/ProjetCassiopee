import os
import numpy as np
import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

###############################################################################
## RGB-D Object Dataset #######################################################
###############################################################################

class RGBDObjectDataset(Dataset):

    def __init__(self, path, mode, modalities=["rgb"], transform=None, nb_imgs=None):
        super().__init__()
        self.path = path

        self.mode = mode
        self.modalities = modalities
        self.train_test_ratio = 4
        self.transform = transform
        self.nb_imgs = nb_imgs

        self.class_dict = None

        self.x = []
        self.y = []
        
        self.create_labels_dict()
        self.load_data()

        if nb_imgs is not None:
            self.x = self.x[:nb_imgs]
            self.y = self.y[:nb_imgs]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.path,
                                 "_".join(self.x[idx].split("_")[:-3]),
                                 "_".join(self.x[idx].split("_")[:-2]),
                                 self.x[idx])

        load_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        # RGB Data
        img = -1
        if "rgb" in self.modalities:
            rgb = cv2.imread(data_path + ".png")
            rgb = load_transform(rgb)
            if self.transform:
                rgb = self.transform(img)

        # Depth Data
        depth = -1
        if "depth" in self.modalities:
            depth = cv2.imread(data_path + "_depth.png")
            depth = load_transform(depth)

        # Mask Data
        mask = -1
        if "mask" in self.modalities:
            mask = cv2.imread(data_path + "_mask.png")
            mask = load_transform(mask)
        
        # Location Data
        loc = -1
        if "loc" in self.modalities:
            with open(data_path + "_loc.txt", "r") as loc_file:
                loc_x, loc_y = loc_file.readlines()[0].split(",")
                loc_x = int(loc_x)
                loc_y = int(loc_y)

        # Label
        label = self.y[idx]

        return rgb, depth, mask, loc_x, loc_y, label
    
    def create_labels_dict(self):
        classes = os.listdir(self.path)
        classes = [c for c in classes if os.path.isdir(os.path.join(self.path, c))]
        classes = sorted(classes)
        self.class_dict = {c: i for i,c in enumerate(classes)}
        return self.class_dict
    
    def load_data(self):
        disc = ["loc", "depth", "mask"]
        for c in list(self.class_dict.keys()):
            for sc in os.listdir(os.path.join(self.path, c)):
                data = [f[:-4] for f in os.listdir(os.path.join(self.path, c, sc)) if not any(d in f for d in disc)]
                
                nb_new = 0
                if self.mode == "train":
                    new = [img for img in data if int(img.split("_")[-1]) % self.train_test_ratio != 0]
                    nb_new = len(new)
                    self.x += new
                elif self.mode == "test":
                    new = [img for img in data if int(img.split("_")[-1]) % self.train_test_ratio == 0]
                    nb_new = len(new)
                    self.x += new
                else:
                    print(f"ERROR: Invalid dataset mode {self.mode}")

                self.y += [self.class_dict[c]] * nb_new