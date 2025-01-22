import numpy as np
import os

def split_3d_to_2d(input_path, output_dir):
    # Load the 3D image (assuming it's a numpy array)
    img_3d = np.load(input_path)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the shape of the 3D image (assumed to be in [z, y, x] format)
    z_dim, y_dim, x_dim = img_3d.shape
    print(f"3D image shape: {img_3d.shape}")

    # Iterate over the z-axis (first dimension) and save each slice as a 2D image
    for z in range(z_dim):
        slice_2d = img_3d[z, :, :]  # Extract the 2D slice along the z-axis

        # Define the filename for saving the slice
        output_file = os.path.join(output_dir, f"slice_{z:03d}.npy")

        # Save the 2D slice
        np.save(output_file, slice_2d)
        print(f"Saved slice {z} to {output_file}")

def main():
    input_path = 'elastix_3d/7_to_12.npy'  # Path to the input 3D image
    output_dir = 'elastix_3d/slices_7_to_12'  # Folder to save the 2D slices

    # Split the 3D image into 2D slices and save them
    split_3d_to_2d(input_path, output_dir)

if __name__ == "__main__":
    main()