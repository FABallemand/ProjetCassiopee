import os
import random
import cv2
import torch
import torchvision
from torch.utils.data import Dataset

from ..transformation.custom_crop import RandomCrop, ObjectCrop
from ..transformation.random_transformation import RandomTransformation


DEFAULT_TRANSOFRMATION = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


class RGBDObjectDataset(Dataset):
    """
    PyTorch dataset for the RGB-D Objects dataset.
    Link: https://rgbd-dataset.cs.washington.edu/dataset.html
    """

    def __init__(self, path, mode, class_names=None, modalities=["rgb"],
                 transformation=DEFAULT_TRANSOFRMATION, crop_transformation=None,
                 train_percentage=0.6, validation_percentage=0.2, test_percentage=0.2, nb_max_samples=None):
        """
        Initialise RGBDObjectDataset instance.

        Parameters
        ----------
        path : str
            Path to the dataset
        mode : str
            Dataset use: "train", "validation" or "test"
        class_names : List[str], optional
            Name of the class to load data from, by default None
        modalities : list, optional
            Modalities to load: "rgb", "depth", "mask", "loc", by default ["rgb"]
        transformation : torchvision.transforms.Compose, optional
            Transformation to apply to image modalities, by default DEFAULT_TRANSOFRMATION
        crop_transformation : torchvision.transforms.Compose, optional
            Additional custom crop transformation to apply to image modalities, by default None
        train_percentage : float, optional
            Percentage of training images , by default 0.6
        validation_percentage : float, optional
            Percentage of validation images , by default 0.2
        test_percentage : float, optional
            Percentage of test images , by default 0.2
        nb_max_samples : int, optional
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
        self.crop_transformation = crop_transformation

        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage
        self.test_percentage = test_percentage
        self.nb_max_samples = nb_max_samples

        self.rgb_flag = "rgb" in self.modalities
        self.depth_flag = "depth" in self.modalities
        self.mask_flag = "mask" in self.modalities or isinstance(self.crop_transformation, ObjectCrop)
        self.loc_flag = "loc" in self.modalities or isinstance(self.crop_transformation, RandomCrop)

        self.class_dict = None

        self.x = []
        self.y = []
        self.removed = []
        
        self._create_labels_dict()
        self._load_data()

    def __str__(self):
        return (f"RGBDObjectDataset(path={self.path}, mode={self.mode}, class_names={self.class_names}, modalities={self.modalities}, "
                f"transformation={self.transformation}, crop_transformation={self.crop_transformation}, "
                f"train_percentage={self.train_percentage}, validation_percentage={self.validation_percentage}, test_percentage={self.test_percentage}, nb_max_samples={self.nb_max_samples})")
    
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
            for sc in [f for f in os.listdir(os.path.join(self.path, c)) if os.path.isdir(os.path.join(self.path, c, f))]:
                sc_path = os.path.join(self.path, c, sc)
                data = [f[:-4] for f in os.listdir(os.path.join(self.path, c, sc)) if os.path.isfile(os.path.join(sc_path, f)) and not any(d in f for d in disc)]

                # Remove samples with missing data
                all_files = [f[:-4] for f in os.listdir(os.path.join(self.path, c, sc))]
                if self.depth_flag:
                    for sample in data:
                        if (sample + "_depth") not in all_files:
                            print(f"Missing depth data for {sample}")
                            # data.remove(sample)
                            data = list(filter(lambda s: s != sample, data))
                            self.removed.append(sample)
                if self.mask_flag:
                    for sample in data:
                        if (sample + "_mask") not in all_files:
                            print(f"Missing mask data for {sample}")
                            # data.remove(sample)
                            data = list(filter(lambda s: s != sample, data))
                            self.removed.append(sample)
                if self.loc_flag:
                    for sample in data:
                        if (sample + "_loc") not in all_files:
                            print(f"Missing localisation data for {sample}")
                            # data.remove(sample)
                            data = list(filter(lambda s: s != sample, data))
                            self.removed.append(sample)

                # Sort data
                data = sorted(data)
                nb_samples = len(data)

                # Compute indices to extract correct data
                nb_train = int(self.train_percentage * nb_samples)
                nb_validation = int(self.validation_percentage * nb_samples)
                nb_test = int(self.test_percentage * nb_samples)

                # print(f"nb_train={nb_train}")
                # print(f"nb_validation={nb_validation}")
                # print(f"nb_test={nb_test}")
                
                if self.mode == "train":
                    self.x += data[:nb_train]
                    self.y += [self.class_dict[c]] * len(data[:nb_train])
                    # print(f"TEST -> {len(data[:nb_train])} | {nb_train} | {len(self.x)} | {len(self.y)}")
                elif self.mode == "validation":
                    self.x += data[nb_train:nb_train + nb_validation]
                    self.y += [self.class_dict[c]] * len(data[nb_train:nb_train + nb_validation])
                    # print(f"TEST -> {len(data[nb_train:nb_train + nb_validation])} | {nb_validation} | {len(self.x)} | {len(self.y)}")
                elif self.mode == "test":
                    self.x += data[nb_train + nb_validation:]
                    self.y += [self.class_dict[c]] * len(data[nb_train + nb_validation:])
                    # print(f"TEST -> {len(data[nb_train + nb_validation:])} | {nb_test} | {len(self.x)} | {len(self.y)}")
                else:
                    print(f"Invalid dataset mode {self.mode}, loading all images...")
                    self.x += data
                    self.y += [self.class_dict[c]] * nb_samples
        
        assert len(self.x) == len(self.y)
        # Reduce the number of samples to the specified number
        if self.nb_max_samples is not None and self.nb_max_samples < len(self.x):
            # Shuffle data in order to have multiple classes
            x_and_y = list(zip(self.x, self.y))
            random.shuffle(x_and_y)
            x_and_y = x_and_y[:self.nb_max_samples]
            self.x = [x for x,y in x_and_y]
            self.y = [y for x,y in x_and_y]
        assert len(self.x) == len(self.y)
        
        # print(f"Removed following samples from dataset because of missing data: {self.removed}")
        # print(f"Number of samples in the dataset: {len(self.x)} | {len(self.y)}")
        # print(f"Samples in the dataset: {self.x}")

    def _load_item_data(self, idx, data_path):

        # RGB Data
        rgb = -1
        if self.rgb_flag:
            rgb = cv2.imread(data_path + ".png")
            rgb = self.transformation(rgb)

        # Depth Data
        depth = -1
        if self.depth_flag:
            depth = cv2.imread(data_path + "_depth.png")
            depth = self.transformation(depth)

        # Mask Data
        mask = -1
        if self.mask_flag:
            mask = cv2.imread(data_path + "_mask.png")
            if mask is None:
                print(data_path)
            mask = self.transformation(mask)

        # Location Data
        loc_x = -1
        loc_y = -1
        if self.loc_flag:
            with open(data_path + "_loc.txt", "r") as loc_file:
                loc_x, loc_y = loc_file.readlines()[0].split(",")
                loc_x = int(loc_x)
                loc_y = int(loc_y)

        # Label
        label = self.y[idx]

        # Crop transformation
        if self.crop_transformation is not None:
            rgb, depth, mask, loc_x, loc_y = self.crop_transformation(rgb, depth, mask, loc_x, loc_y)

        return rgb, depth, mask, loc_x, loc_y, label
    
    def __getitem__(self, idx):
        # print(f"Get data for sample {idx}: {self.x[idx]} from dataset {self}")
        data_path = os.path.join(self.path,
                                 "_".join(self.x[idx].split("_")[:-3]),
                                 "_".join(self.x[idx].split("_")[:-2]),
                                 self.x[idx])
        
        rgb, depth, mask, loc_x, loc_y, label = self._load_item_data(idx, data_path)

        return rgb, depth, mask, loc_x, loc_y, label


class RGBDObjectDataset_Supervised_Contrast(RGBDObjectDataset):

    def __init__(self, path, mode, class_names=None, modalities=["rgb"],
                 transformation=DEFAULT_TRANSOFRMATION, crop_transformation=None,
                 train_percentage=0.6, validation_percentage=0.2, test_percentage=0.2, nb_max_samples=None):
        """
        PyTorch dataset for the RGB-D Objects dataset.
        Task: supervised learning with contrastive learning.
        Only override __getitem__ method.
        Link: https://rgbd-dataset.cs.washington.edu/dataset.html

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
        crop_transformation : torchvision.transforms.Compose, optional
            Additional custom crop transformation to apply to image modalities, by default None
        train_percentage : float, optional
            Percentage of training images , by default 0.6
        validation_percentage : float, optional
            Percentage of validation images , by default 0.2
        test_percentage : float, optional
            Percentage of test images , by default 0.2
        nb_max_samples : int, optional
            Maximum number of samples in the dataset, by default None
        """
        super().__init__(path, mode, class_names, modalities, transformation, crop_transformation, train_percentage, validation_percentage, test_percentage, nb_max_samples)

    def __str__(self):
        return (f"RGBDObjectDataset_Supervised_Contrast(path={self.path}, mode={self.mode}, class_names={self.class_names}, modalities={self.modalities}, "
                f"transformation={self.transformation}, crop_transformation={self.crop_transformation}, "
                f"train_percentage={self.train_percentage}, validation_percentage={self.validation_percentage}, test_percentage={self.test_percentage}, nb_max_samples={self.nb_max_samples})")
    
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
    

class RGBDObjectDataset_Unsupervised_Contrast(RGBDObjectDataset):
    """
    PyTorch dataset for the RGB-D Objects dataset.
    Task: unsupervised learning with contrastive learning.
    Only override __getitem__ method.
    Link: https://rgbd-dataset.cs.washington.edu/dataset.html
    """

    def __init__(self, path, mode, class_names=None, modalities=["rgb"],
                 transformation=DEFAULT_TRANSOFRMATION, crop_transformation=None,
                 train_percentage=0.6, validation_percentage=0.2, test_percentage=0.2, nb_max_samples=None):
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
        crop_transformation : torchvision.transforms.Compose, optional
            Additional custom crop transformation to apply to image modalities, by default None
        train_percentage : float, optional
            Percentage of training images , by default 0.6
        validation_percentage : float, optional
            Percentage of validation images , by default 0.2
        test_percentage : float, optional
            Percentage of test images , by default 0.2
        nb_max_samples : int, optional
            Maximum number of samples in the dataset, by default None
        """
        super().__init__(path, mode, class_names, modalities, transformation, crop_transformation, train_percentage, validation_percentage, test_percentage, nb_max_samples)
        self.generate_p_2 = RandomTransformation(output_size=(256, 256))

    def __str__(self):
        return (f"RGBDObjectDataset_Unsupervised_Contrast(path={self.path}, mode={self.mode}, class_names={self.class_names}, modalities={self.modalities}, "
                f"transformation={self.transformation}, crop_transformation={self.crop_transformation}, "
                f"train_percentage={self.train_percentage}, validation_percentage={self.validation_percentage}, test_percentage={self.test_percentage}, nb_max_samples={self.nb_max_samples})")
        
    def __getitem__(self, p_idx_1):

        # Load positive data 1
        p_class = "_".join(self.x[p_idx_1].split("_")[:-3])
        p_subclass_1 = "_".join(self.x[p_idx_1].split("_")[:-2])
        p_data_path_1 = os.path.join(self.path,
                                     p_class,
                                     p_subclass_1,
                                     self.x[p_idx_1])
        p_data_1 = self._load_item_data(p_idx_1, p_data_path_1)

        # Create positive data 2 from positive data 1
        p_data_2 = self.generate_p_2(*p_data_1)
        
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
    

class RGBDObjectDataset_Unsupervised_Contrast_bis(RGBDObjectDataset):
    """
    PyTorch dataset for the RGB-D Objects dataset.
    Task: unsupervised learning with contrastive learning.
    Only override __getitem__ method.
    Link: https://rgbd-dataset.cs.washington.edu/dataset.html
    """

    def __init__(self, path, mode, class_names=None, modalities=["rgb"],
                 transformation=DEFAULT_TRANSOFRMATION, crop_transformation=None,
                 train_percentage=0.6, validation_percentage=0.2, test_percentage=0.2, nb_max_samples=None):
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
        crop_transformation : torchvision.transforms.Compose, optional
            Additional custom crop transformation to apply to image modalities, by default None
        train_percentage : float, optional
            Percentage of training images , by default 0.6
        validation_percentage : float, optional
            Percentage of validation images , by default 0.2
        test_percentage : float, optional
            Percentage of test images , by default 0.2
        nb_max_samples : int, optional
            Maximum number of samples in the dataset, by default None
        """
        super().__init__(path, mode, class_names, modalities, transformation, crop_transformation, train_percentage, validation_percentage, test_percentage, nb_max_samples)

    def __str__(self):
        return (f"RGBDObjectDataset_Unsupervised_Contrast_bis(path={self.path}, mode={self.mode}, class_names={self.class_names}, modalities={self.modalities}, "
                f"transformation={self.transformation}, crop_transformation={self.crop_transformation}, "
                f"train_percentage={self.train_percentage}, validation_percentage={self.validation_percentage}, test_percentage={self.test_percentage}, nb_max_samples={self.nb_max_samples})")
        
    def __getitem__(self, p_idx_1):

        # Load positive data 1
        p_class = "_".join(self.x[p_idx_1].split("_")[:-3])
        p_subclass_1 = "_".join(self.x[p_idx_1].split("_")[:-2])
        p_data_path_1 = os.path.join(self.path,
                                     p_class,
                                     p_subclass_1,
                                     self.x[p_idx_1])
        p_data_1 = self._load_item_data(p_idx_1, p_data_path_1)
        
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

        return [p_data_1, n_data]