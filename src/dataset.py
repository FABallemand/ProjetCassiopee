import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

###############################################################################
## RGB-D Object Dataset #######################################################
###############################################################################

class RGBDObjectDataset(Dataset):

    def __init__(self, path, train):
        super().__init__()
        self.path = path

        self.train = train
        self.train_test_ratio = 4
        self.class_dict = None

        self.x = []
        self.y = []
        
        self.create_labels_dict()
        self.load_data()

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.path,
                                "_".join(self.x[idx].split("_")[:-3]),
                                "_".join(self.x[idx].split("_")[:-2]),
                                self.x[idx])
        img = torchvision.io.read_image(img_path)
        label = self.y[idx]
        return img, label
    
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
                rgb_imgs = [f for f in os.listdir(os.path.join(self.path, c, sc)) if not any(d in f for d in disc)]
                
                nb_new = 0
                if self.train:
                    new = [img for img in rgb_imgs if int(img.split("_")[-1][:-4]) % self.train_test_ratio != 0]
                    nb_new = len(new)
                    self.x += new
                else:
                    new = [img for img in rgb_imgs if int(img.split("_")[-1][:-4]) % self.train_test_ratio == 0]
                    nb_new = len(new)
                    self.x += new

                self.y += [self.class_dict[c]] * nb_new