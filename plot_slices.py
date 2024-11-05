import numpy as np
import matplotlib.pyplot as plt

def plot_slices(nd_array, save_as=None, cols=8, cmap='gray', vmin_val=None, vmax_val=None):
    """
    Plots slices from a 3D or 4D N-D array (e.g., [x, y, z] or [x, y, z, t]).

    Parameters:
        nd_array (numpy array): Input array (can be 3D or 4D). 
                                If 4D, it will average over the 4th dimension.
        save_as (str): File path to save the plot. If None, the plot will be displayed.
        cols (int): Number of columns to arrange the slices in.
        cmap (str): Colormap to use for plotting.
    """
    # If the array is 4D, average over the 4th dimension (time)
    if len(nd_array.shape) == 4:
        mean_data = np.mean(nd_array, axis=3)
    elif len(nd_array.shape) == 3:
        mean_data = nd_array
    else:
        raise ValueError("Input array must be 3D or 4D.")

    # Get the number of slices (3rd dimension of the data)
    num_slices = mean_data.shape[2]

    # Calculate number of rows needed based on the number of columns
    rows = num_slices // cols
    if num_slices % cols > 0:
        rows += 1

    # Create the subplots grid
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))

    # Plot each slice
    for i in range(num_slices):
        row = i // cols
        col = i % cols
        im = axes[row, col].imshow(mean_data[:, :, i], cmap=cmap, vmin=vmin_val, vmax=vmax_val)
        axes[row, col].set_title(f'Slice {i+1}', fontsize=8)
        axes[row, col].axis('off')  # Turn off axes for better visualization

    #cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    #cbar.set_label('Colorbar label')

    # Remove any empty subplots
    for j in range(i+1, rows * cols):
        fig.delaxes(axes[j // cols, j % cols])

    # Adjust the layout to avoid overlap
    plt.tight_layout()

    # Save to file if save_as is provided
    if save_as:
        plt.savefig(save_as)
        print(f"Plot saved as {save_as}")
    else:
        plt.show()
