import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def plot_animation(frames, coordinates):
    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 6)  # Adjust the limits according to your data
    ax.set_ylim(0, 6)
    ax.set_zlim(0, 6)

    # Initialize an empty scatter plot (to be updated in the animation)
    scatter = ax.scatter([], [], [])

    # Function to update the scatter plot
    def update(frame):
        # Get the coordinates for the current frame
        frame_coordinates = coordinates[frame]

        # Update the scatter plot with new point positions
        scatter._offsets3d = frame_coordinates

        # Set the title to display the current frame
        ax.set_title(f"Frame: {frame}")

    # Create the animation
    animation = FuncAnimation(fig, update, frames=frames, blit=False)

    # Show the animation
    #plt.show()
    
    # Save the animation as a GIF
    animation.save('self_supervised_learning/dev/ProjetCassiopee/src/visualisation/animation.gif', writer='imagemagick')