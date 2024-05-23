import os
import random
import pandas as pd
import csv
from torch.utils.data import Dataset
import numpy as np
import torch

class MocaplabDatasetFC(Dataset):
    """
    PyTorch dataset for the Mocaplab dataset.
    """

    def __init__(self, path, padding=True, train_test_ratio=8, validation_percentage=0.01, nb_samples=None):
        super().__init__()
        self.path = path
        print(f"TEST {self.path}")
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
    
    def _read_csv(self, csv_file) :
        data = []
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file, delimiter=';')
            n=0
            for line in csv_reader :
                if n>=2 :
                    values = line[2:]
                    for i in range(len(values)) :
                        values[i] = float(values[i])
                    data.append(values)
                n+=1
        data = np.stack(data)
        return data

    def __len__(self):
        return len(self.y)
    
    def _create_labels_dict(self):
        self.class_dict = {"Mono": 0, "Bi": 1}
        return self.class_dict
    
    def _load_data(self):
        # Retrieve labels
        labels = pd.read_csv(os.path.join(self.path, "Annotation_gloses.csv"),
                             sep="\t")
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
                data = self._read_csv(os.path.join(self.path, filename))
                length = len(data)
                if length > self.max_length:
                    self.max_length = length

    def __getitem__(self, idx):
        data_path = os.path.join(self.path, self.x[idx])

        data = self._read_csv(data_path)
        label = self.y[idx]

        if self.padding:
            data = data.tolist()
            for _ in range(self.max_length-len(data)) :
                data.append([0.0 for _ in range(237)])
            data = np.stack(data)
        
        return data, label, self.x[idx]