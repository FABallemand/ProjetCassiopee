import torch
import sys
sys.path.append("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee")
from src.setup import setup_python, setup_pytorch
import matplotlib.pyplot as plt
from src.models.rgbd_object.autoencoder.autoencoder import TestAutoencoder

if __name__=='__main__':

    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch()
    #Define the path to the saved model
    model_path = 'path_to_the_saved_model'

    #Initialize an instance of the model
    model = TestAutoencoder().to(DEVICE)

    #Load the saved state dictionary
    model.load_state_dict(torch.load(model_path))

    # Get input 
    input_data = 

    # Reconstruction
    with torch.no_grad() :
        encoded, reconstructed = model(input_data)

    # Plot the original and reconstructed images
    n = min(5, input_data.size(0))  # Number of images to plot, limit to 5 for visualization
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Plot original images
        plt.subplot(2, n, i + 1)
        plt.imshow(input_data[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title('Original')
        plt.axis('off')

        # Plot reconstructed images
        plt.subplot(2, n, i + n + 1)
        plt.imshow(reconstructed[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')

    plt.show()