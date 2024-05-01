import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import sys

sys.path.append("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/")
from src.dataset import MocaplabDatasetFC
from torch.utils.data import DataLoader

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.setup import setup_python, setup_pytorch
from src.dataset import MocaplabDatasetFC
from src.models.mocaplab import MocaplabFC
from fc.plot_results import plot_results
from fc.train import *

def plot_animation(i, data, label, prediction, nom):

    model = 'FC'

    data = data.numpy()
    
    x_data = data[:, 0::3]
    y_data = data[:, 1::3]
    z_data = data[:, 2::3]

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(min([min(x_data[i]) for i in range(len(x_data))]), max([max(x_data[i]) for i in range(len(x_data))]))
    ax.set_ylim(min([min(y_data[i]) for i in range(len(y_data))]), max([max(y_data[i]) for i in range(len(y_data))]))
    ax.set_zlim(min([min(z_data[i]) for i in range(len(z_data))]), max([max(z_data[i]) for i in range(len(z_data))]))

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    # Initialize an empty scatter plot (to be updated in the animation)
    scatter = ax.scatter([], [], [])

    # Initialize lines
    line_points_indices = [(0, 1), (0, 2), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 9), (8, 9), (1, 70),     # Chest and head
                           (3, 10), (2, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),                  # Right arm (without hand)
                           (2, 40), (3, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45),                  # Left arm (without hand)
                           (70, 71), (71, 72), (72, 73), (73, 74),                                              # Right leg
                           (70, 75), (75, 76), (76, 77), (77, 78),                                              # Left leg
                           (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),                                    # Right hand, pinky
                           (15, 21), (21, 22), (22, 23), (23, 24), (24, 25),                                    # Right hand, ring
                           (15, 26), (26, 27), (27, 28), (28, 29), (29, 30),                                    # Right hand, mid
                           (15, 31), (31, 32), (32, 33), (33, 34), (34, 35),                                    # Right hand, index
                           (15, 36), (36, 37), (37, 38), (38, 39),                                              # Right hand, thumb
                           (45, 46), (46, 47), (47, 48), (48, 49), (49, 50),                                    # Left hand, pinky
                           (45, 51), (51, 52), (52, 53), (53, 54), (54, 55),                                    # Left hand, ring
                           (45, 56), (56, 57), (57, 58), (58, 59), (59, 60),                                    # Left hand, mid
                           (45, 61), (61, 62), (62, 63), (63, 64), (64, 65),                                    # Left hand, index
                           (45, 66), (66, 67), (67, 68), (68, 69)]                                              # Left hand, thumb                
    
    # Plus qu'à faire les mains 

    num_lines = len(line_points_indices)
    lines = [ax.plot([], [], [], '-', color='red')[0] for _ in range(num_lines)]

    # Function to update the scatter plot
    def update(frame):    
        # Get the coordinates for the current frame
        frame_coordinates = (x_data[frame], z_data[frame], y_data[frame])

        # Update the scatter plot with new point positions
        scatter._offsets3d = frame_coordinates

        # Set the title to display the current data, label and frame
        ax.set_title(f"Data {i}, Label : {label}, Prediction : {prediction} with model {model} \nFrame: {frame}")

        # Adding lines between the joints
        for line, (start, end) in zip(lines, line_points_indices):
            line.set_data_3d([x_data[frame][start], x_data[frame][end]], [z_data[frame][start], z_data[frame][end]], [y_data[frame][start], y_data[frame][end]])

        return scatter, *lines

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(data), blit=True)
    
    # Save the animation as a GIF
    animation.save(f'self_supervised_learning/dev/ProjetCassiopee/src/visualisation/mocaplab/{nom}.gif', writer='pillow')
    plt.close()

def plot_animation_no_points(i, data, label, prediction, nom):

    model = 'FC'

    data = data.numpy()
    
    x_data = data[:, 0::3]
    y_data = data[:, 1::3]
    z_data = data[:, 2::3]

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(min([min(x_data[i]) for i in range(len(x_data))]), max([max(x_data[i]) for i in range(len(x_data))]))
    ax.set_ylim(min([min(y_data[i]) for i in range(len(y_data))]), max([max(y_data[i]) for i in range(len(y_data))]))
    ax.set_zlim(min([min(z_data[i]) for i in range(len(z_data))]), max([max(z_data[i]) for i in range(len(z_data))]))

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    # Initialize lines
    line_points_indices = [(0, 1), (0, 2), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 9), (8, 9), (1, 70),     # Chest and head
                           (3, 10), (2, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),                  # Right arm (without hand)
                           (2, 40), (3, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45),                  # Left arm (without hand)
                           (70, 71), (71, 72), (72, 73), (73, 74),                                              # Right leg
                           (70, 75), (75, 76), (76, 77), (77, 78),                                              # Left leg
                           (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),                                    # Right hand, pinky
                           (15, 21), (21, 22), (22, 23), (23, 24), (24, 25),                                    # Right hand, ring
                           (15, 26), (26, 27), (27, 28), (28, 29), (29, 30),                                    # Right hand, mid
                           (15, 31), (31, 32), (32, 33), (33, 34), (34, 35),                                    # Right hand, index
                           (15, 36), (36, 37), (37, 38), (38, 39),                                              # Right hand, thumb
                           (45, 46), (46, 47), (47, 48), (48, 49), (49, 50),                                    # Left hand, pinky
                           (45, 51), (51, 52), (52, 53), (53, 54), (54, 55),                                    # Left hand, ring
                           (45, 56), (56, 57), (57, 58), (58, 59), (59, 60),                                    # Left hand, mid
                           (45, 61), (61, 62), (62, 63), (63, 64), (64, 65),                                    # Left hand, index
                           (45, 66), (66, 67), (67, 68), (68, 69)]                                              # Left hand, thumb                
    
    # Plus qu'à faire les mains 

    num_lines = len(line_points_indices)
    lines = [ax.plot([], [], [], '-', color='red')[0] for _ in range(num_lines)]

    # Function to update the scatter plot
    def update(frame):

        # Set the title to display the current data, label and frame
        ax.set_title(f"Data {i}, Label : {label}, Prediction : {prediction} with model {model} \nFrame: {frame}")

        # Adding lines between the joints
        for line, (start, end) in zip(lines, line_points_indices):
            line.set_data_3d([x_data[frame][start], x_data[frame][end]], [z_data[frame][start], z_data[frame][end]], [y_data[frame][start], y_data[frame][end]])

        return *lines,

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(data), blit=True)
    
    # Save the animation as a GIF
    animation.save(f'self_supervised_learning/dev/ProjetCassiopee/src/visualisation/mocaplab/{nom}_noPoints.gif', writer='pillow')
    plt.close()

def plot_frame(i, data, label) :

    x_data = data[:, 0::3][0]
    y_data = data[:, 1::3][0]
    z_data = data[:, 2::3][0]

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlim(min(x_data), max(x_data))
    ax.set_ylim(min(z_data), max(z_data))
    ax.set_zlim(min(y_data), max(y_data))

    ax2 = fig.add_subplot(122)
    ax2.imshow(data)
    ax2.set_title("Full Array")

    ax.scatter(x_data, z_data, y_data)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title(f'Frame 0, data {i}, label = {label}')

    plt.savefig("self_supervised_learning/dev/ProjetCassiopee/src/visualisation/mocaplab/test.png")
    plt.close()

if __name__ == "__main__":
    
    print("#### Begin ####")

    DEVICE = torch.device("cpu")

    dataset = MocaplabDatasetFC(path="self_supervised_learning/dev/ProjetCassiopee/data/mocaplab/Cassiopée_Allbones",
                              padding = True, 
                              train_test_ratio = 8,
                              validation_percentage = 0.01)
    
    data_loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False)
    
    model = MocaplabFC(dataset.max_length*237).to(DEVICE)
    model.load_state_dict(torch.load("self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/fc/saved_models/model_20240325_141951.ckpt"))
    model = model.to(DEVICE)
    model = model.double()
    
    for i, batch in enumerate(data_loader) :

        data, label = batch
        data = data.to(DEVICE)
        label = label.to(DEVICE)
    
        data_flattened = data.view(data.size(0), -1)
        output = model(data_flattened.double())

        _, predicted = torch.max(output.data, dim=1)

        label = int(label[0])
        predicted = int(predicted[0])
        
        data = data.squeeze(0)
        nom = f'{i}_{label}_{predicted}'

        n0 = 0
        n1 = 0
        ndiff = 0

        if ndiff<2 and predicted != label :
            ndiff += 1
            print(f"Plotting {nom} ...")
            plot_animation_no_points(i, data, label, predicted, nom)
            plot_animation(i, data, label, predicted, nom)

        elif n0<2 and label==0 :
            assert(n0<=2)
            assert(label==0)
            n0 += 1
            print(f"Plotting {nom} ...")
            plot_animation_no_points(i, data, label, predicted, nom)
            plot_animation(i, data, label, predicted, nom)

        elif n1<2 and label==1 :
            assert(n1<=2)
            assert(label==1)
            n1 += 1
            print(f"Plotting {nom} ...")
            plot_animation_no_points(i, data, label, predicted, nom)
            plot_animation(i, data, label, predicted, nom)

    print('#### DONE ####')