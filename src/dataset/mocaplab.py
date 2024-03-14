import os
import logging
import cv2
import pandas as pd
import csv
import torch
import torchvision
from torch.utils.data import Dataset

class MocaplabDataset(Dataset):
    """
    PyTorch dataset for the Mocaplab dataset.
    """

    def __init__(self, path, all_bones=False, train_test_ratio=8, validation_percentage=0.01):
        super().__init__()
        self.path = path
        self.all_bones = all_bones
        self.train_test_ratio = train_test_ratio
        self.validation_percentage = validation_percentage

        self.data = []
        self.labels = []

        self.load_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, nom):
        data_path = os.path.join(self.path, self.data[nom])
        with open(data_path+'.csv', "r") as file:
            reader = csv.reader(file)
            data = list(reader)

        # Extract frame, time and coordinates columns
        frames = [int(row[0]) for row in data[2:]]
        times = [float(row[1]) for row in data[2:]]
        coordinates = [[(row[i], row[i+1], row[i+2]) for i in range(2, len(row)-2, 3)] for row in data[2:]]

        label = self.labels[nom]

        return frames, times, coordinates, label

    def load_data(self):
        classes = pd.read_csv("Annotation_gloses.csv", sep="\t")
        classes.dropna(inplace=True)
        class_dict = {nom: classe for nom, classe in zip(classes["Nom.csv"], classes["Mono/Bi"])}
        self.labels = class_dict
        existing = os.listdir("Cassiopée") if not self.all_bones else os.listdir("Cassiopée_Allbones")
        for nom in class_dict:
            if nom+".csv" not in existing:
                continue
            else:
                if not self.all_bones :
                    self.ajouter_cellule_vide("Cassiopée/"+nom+".csv", 1, 1)
                obj = pd.read_csv("Cassiopée/"+nom+".csv", sep=";")
                self.data.append(obj)
        # get the maximum nuumber of frames
        max_length = max([len(data) for data in self.data])
        for data in self.data :
            # pad the data to the maximum length
            data = self.pad_max(data, max_length)
    
    def ajouter_cellule_vide(csv_file, row_index, col_index):
        with open(csv_file, "r") as file:
            reader = csv.reader(file, delimiter=";")
            data = list(reader)

        if row_index < len(data):
            if col_index < len(data[row_index]):
                data[row_index].insert(col_index, "")

                with open(csv_file, "w", newline="") as file:
                    writer = csv.writer(file, delimiter=";")
                    writer.writerows(data)
                # print("Cellule vide ajoutée avec succès.")
            else:
                pass
                # print("Index de colonne non valide.")
        # else:
            # print("Index de ligne non valide.")
    
    def pad_max(self, data, max_length):
        # Pad the data to the maximum length
        return data + [0] * (max_length - len(data))