import os

import matplotlib.pyplot as plt

def reconstruction_plot(img, reconstruction, n, path):

    # Create figure
    fig, axs = plt.subplots(n, 2)

    for i in range(n):  
        axs[i,0].imshow(img[i].permute(1,2,0))
        axs[i,0].set_axis_off()
        axs[i,1].imshow(reconstruction[i].permute(1,2,0))
        axs[i,1].set_axis_off()
    
    plt.tight_layout()
    plt.savefig(path)