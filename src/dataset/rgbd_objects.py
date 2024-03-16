import os
import logging
import random
import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

DEFAULT_TRANSOFRMATION = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


class RGBDObjectDataset(Dataset):
    """
    PyTorch dataset for the RGB-D Objects dataset.
    Link: https://rgbd-dataset.cs.washington.edu/dataset.html
    """

    def __init__(self, path, mode, class_names=None, modalities=["rgb"], transformation=DEFAULT_TRANSOFRMATION, train_test_ratio=8, validation_percentage=0.01, nb_samples=None):
        """
        Initialise RGBDObjectDataset instance.

        Parameters
        ----------
        path : str
            Path too the dataset
        mode : str
            Dataset use: "train", "validation" or "test"
        class_names : List[str], optional
            Name of the class to load data from, by default None
        modalities : list, optional
            Modalities to load: "rgb", "depth", "mask", "loc", by default ["rgb"]
        transformation : torchvision.transforms.Compose, optional
            Transformation to apply to image modalities, by default DEFAULT_TRANSOFRMATION
        train_test_ratio : int, optional
            Ratio of train images over test images, by default 8
        validation_percentage : float, optional
            Percentage of validation data among training data, by default 0.01
        nb_samples : int, optional
            Maximum number of samples in the dataset, by default None
        """
        super().__init__()
        self.path = path

        self.mode = mode
        self.modalities = modalities
        self.class_names = class_names
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
        
        self._create_labels_dict()
        self._load_data()

        if nb_samples is not None:
            # Shuffle data in order to have multiple classes
            x_and_y = list(zip(self.x, self.y))
            random.shuffle(x_and_y)
            x_and_y = x_and_y[:nb_samples]
            self.x = [x for x,y in x_and_y]
            self.y = [y for x,y in x_and_y]

    def __len__(self):
        return len(self.y)
    
    def _create_labels_dict(self):
        classes = os.listdir(self.path)
        classes = [c for c in classes if os.path.isdir(os.path.join(self.path, c))]
        classes = sorted(classes)
        self.class_dict = {c: i for i, c in enumerate(classes)}
        return self.class_dict
    
    def _load_data(self):
        disc = ["depth", "mask", "loc"]

        # Iterate over classes (eg: apple)
        for c in list(self.class_dict.keys()):

            # Skip this class if it is not the selected class
            if self.class_names is not None and c not in self.class_names:
                continue

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

    def _load_item_data(self, idx, data_path):

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
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.path,
                                 "_".join(self.x[idx].split("_")[:-3]),
                                 "_".join(self.x[idx].split("_")[:-2]),
                                 self.x[idx])
        
        rgb, depth, mask, loc_x, loc_y, label = self._load_item_data(idx, data_path)

        return rgb, depth, mask, loc_x, loc_y, label


class RGBDObjectDataset_Contrast(RGBDObjectDataset):
    """
    PyTorch dataset for the RGB-D Objects dataset and contrastive learning.
    Only override __getitem__ method.
    Link: https://rgbd-dataset.cs.washington.edu/dataset.html
    """

    def __init__(self, path, mode, class_names=None, modalities=["rgb"], transformation=DEFAULT_TRANSOFRMATION, train_test_ratio=8, validation_percentage=0.01, nb_samples=None):
        """
        Initialise RGBDObjectDataset_Contrast instance.

        Parameters
        ----------
        path : str
            Path too the dataset
        mode : str
            Dataset use: "train", "validation" or "test"
        class_names : List[str], optional
            Name of the class to load data from, by default None
        modalities : list, optional
            Modalities to load: "rgb", "depth", "mask", "loc", by default ["rgb"]
        transformation : torchvision.transforms.Compose, optional
            Transformation to apply to image modalities, by default DEFAULT_TRANSOFRMATION
        train_test_ratio : int, optional
            Ratio of train images over test images, by default 8
        validation_percentage : float, optional
            Percentage of validation data among training data, by default 0.01
        nb_samples : int, optional
            Maximum number of samples in the dataset, by default None
        """
        super().__init__(path, mode, class_names, modalities, transformation, train_test_ratio, validation_percentage, nb_samples)

    def __getitem__(self, p_idx_1):

        # Load positive data 1
        p_class = "_".join(self.x[p_idx_1].split("_")[:-3])
        p_subclass_1 = "_".join(self.x[p_idx_1].split("_")[:-2])
        p_data_path_1 = os.path.join(self.path,
                                     p_class,
                                     p_subclass_1,
                                     self.x[p_idx_1])
        p_data_1 = self._load_item_data(p_idx_1, p_data_path_1)

        # Load positive data 2
        p_idx_2 = random.randint(0, len(self) - 1)
        while not self.x[p_idx_2].startswith(p_class):
            p_idx_2 = random.randint(0, len(self) - 1)
        p_subclass_2 = "_".join(self.x[p_idx_2].split("_")[:-2])
        p_data_path_2 = os.path.join(self.path,
                                     p_class,
                                     p_subclass_2,
                                     self.x[p_idx_2])
        p_data_2 = self._load_item_data(p_idx_2, p_data_path_2)
        
        # Load negative data
        n_idx = random.randint(0, len(self) - 1)
        while self.x[n_idx].startswith(p_class):
            n_idx = random.randint(0, len(self) - 1)
        n_class = "_".join(self.x[n_idx].split("_")[:-3])
        n_subclass = "_".join(self.x[n_idx].split("_")[:-2])
        n_data_path = os.path.join(self.path,
                                   n_class,
                                   n_subclass,
                                   self.x[n_idx])        
        n_data = self._load_item_data(n_idx, n_data_path)

        return [p_data_1, p_data_2, n_data]