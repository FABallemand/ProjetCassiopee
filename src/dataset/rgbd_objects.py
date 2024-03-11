import os
import logging
import cv2
import torch
import torchvision
from torch.utils.data import Dataset

DEFAULT_TRANSOFRMATION = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

class RGBDObjectDataset(Dataset):
    """
    PyTorch dataset for the RGB-D Objects dataset.
    Link: https://rgbd-dataset.cs.washington.edu/dataset.html
    """

    def __init__(self, path, mode, modalities=["rgb"], transformation=DEFAULT_TRANSOFRMATION, train_test_ratio=8, validation_percentage=0.01, nb_samples=None):
        super().__init__()
        self.path = path

        self.mode = mode
        self.modalities = modalities
        self.transformation = transformation
        if self.transformation is None:
            self.transformation = DEFAULT_TRANSOFRMATION
        self.train_test_ratio = train_test_ratio
        self.validation_percentage = validation_percentage
        self.nb_samples = nb_samples

        self.class_dict = None

        self.x = []
        self.y = []
        self.removed = []
        
        self.create_labels_dict()
        self.load_data()

        if nb_samples is not None:
            self.x = self.x[:nb_samples]
            self.y = self.y[:nb_samples]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.path,
                                 "_".join(self.x[idx].split("_")[:-3]),
                                 "_".join(self.x[idx].split("_")[:-2]),
                                 self.x[idx])
        
        # RGB Data
        rgb = -1
        if "rgb" in self.modalities:
            rgb = cv2.imread(data_path + ".png")
            rgb = self.transformation(rgb)

        # Depth Data
        depth = -1
        if "depth" in self.modalities:
            depth = cv2.imread(data_path + "_depth.png")
            depth = self.transformation(depth)

        # Mask Data
        mask = -1
        if "mask" in self.modalities:
            mask = cv2.imread(data_path + "_mask.png")
            mask = self.transformation(mask)
        
        # Location Data
        loc_x = -1
        loc_y = -1
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
        disc = ["depth", "mask", "loc"]

        # Iterate over classes (eg: apple)
        for c in list(self.class_dict.keys()):

            # Iterate over sub-classes (eg: apple_1)
            for sc in os.listdir(os.path.join(self.path, c)):
                data = [f[:-4] for f in os.listdir(os.path.join(self.path, c, sc)) if not any(d in f for d in disc)]
                data = sorted(data)

                # Remove samples with missing data
                all_files = [f[:-4] for f in os.listdir(os.path.join(self.path, c, sc))]
                if "depth" in self.modalities:
                    for sample in data:
                        if (sample + "_depth") not in all_files:
                            logging.warning(f"Missing depth data for {sample}")
                            data.remove(sample)
                            self.removed.append(sample)
                if "mask" in self.modalities:
                    for sample in data:
                        if (sample + "_mask") not in all_files:
                            logging.warning(f"Missing mask data for {sample}")
                            data.remove(sample)
                            self.removed.append(sample)
                if "loc" in self.modalities:
                    for sample in data:
                        if (sample + "_loc") not in all_files:
                            logging.warning(f"Missing localisation data for {sample}")
                            data.remove(sample)
                            self.removed.append(sample)
                
                nb_new = 0
                if self.mode == "train":
                    new = [sample for sample in data if int(sample.split("_")[-1]) % self.train_test_ratio != 0]
                    new = sorted(new)
                    nb_new = int((1 - self.validation_percentage) * len(new))
                    self.x += new[:nb_new]
                elif self.mode == "validation":
                    new = [sample for sample in data if int(sample.split("_")[-1]) % self.train_test_ratio != 0]
                    new = sorted(new)
                    nb_new_train = int((1 - self.validation_percentage) * len(new))
                    nb_new = len(new) - nb_new_train
                    self.x += new[nb_new_train:]
                elif self.mode == "test":
                    new = [sample for sample in data if int(sample.split("_")[-1]) % self.train_test_ratio == 0]
                    nb_new = len(new)
                    self.x += new
                else:
                    logging.warning(f"WARNING: Invalid dataset mode {self.mode}, loading all images...")
                    nb_new = len(data)
                    self.x += data

                self.y += [self.class_dict[c]] * nb_new