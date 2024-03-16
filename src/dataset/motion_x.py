import logging
import random
import cv2
import torch
import torchvision
from torch.utils.data import Dataset

class MotionXDataset(Dataset):
    """
    PyTorch dataset for the Motion-X dataset.
    Link: https://github.com/IDEA-Research/Motion-X
    """

    def __init__(self, path, padding=True, train_test_ratio=8, validation_percentage=0.01, nb_samples=None):
        super().__init__()
        self.path = path
        self.padding = padding
        self.train_test_ratio = train_test_ratio
        self.validation_percentage = validation_percentage

        self.class_dict = None
        self.max_length = 0

        self.x = []
        self.y = []
        self.labels = None
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
        self.class_dict = {"Mono": 0, "Bi": 1}
        return self.class_dict
    
    def _load_data(self):
        # Retrieve labels
        labels = pd.read_csv(os.path.join(os.path.dirname(self.path),
                                          "Annotation_gloses.csv"), sep="\t")
        labels.dropna(inplace=True)
        self.labels = {n: c for n, c in zip(labels["Nom.csv"], labels["Mono/Bi"])}
        
        # Retrieve files
        files = os.listdir(self.path)
        for name, label in self.labels.items():
            filename = name + ".csv"
            if filename not in files:
                self.removed.append(filename)
            else:
                self.x.append(filename)
                self.y.append(self.class_dict[label])

                # Retrieve max length
                data = pd.read_csv(os.path.join(self.path, filename), sep=";", header=[0,1])
                length = data.shape[0]
                if length > self.max_length:
                    self.max_length = length

    def __getitem__(self, idx):
        data_path = os.path.join(self.path, self.x[idx])

        data = pd.read_csv(data_path, sep=";", header=[0,1])
        label = self.y[idx]

        if self.padding:
            nb_padding_rows = self.max_length - data.shape[0]
            empty_rows = {c: [0 for _ in range(nb_padding_rows)] for c in data.columns}
            empty_data = pd.DataFrame(empty_rows)
            data = pd.concat([data, empty_data], ignore_index=True)
            data = data.to_numpy()
        return data, label