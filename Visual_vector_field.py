import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def visualize_transform_as_vector_field_2d_npy(image_path, transform_param_file):
    """
    Visualize the transformation as a vector field for a 2D image stored as a .npy file
    using the transform parameter file.
    
    Parameters:
        image_path (str): Path to the input image stored as a .npy file.
        transform_param_file (str): Path to the TransformParameters.0.txt file.
    """
    # Load the image from .npy file
    image_array = np.load(image_path)
    
    # Convert the NumPy array to a SimpleITK image
    image = sitk.GetImageFromArray(image_array)
    
    # Load the transform parameters from the TransformParameters.0.txt file
    transform = sitk.ReadTransform(transform_param_file)
    
    # Create a displacement field by evaluating the transform at each voxel
    displacement_field = sitk.TransformToDisplacementField(transform, image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection())
    
    # Convert displacement field to a numpy array
    displacement_array = sitk.GetArrayFromImage(displacement_field)
    
    # Visualize the displacement field as a vector field
    if len(displacement_array.shape) == 3:
        # For 2D images, we just take the middle slice
        displacement_slice = displacement_array[displacement_array.shape[0] // 2, :, :]
        
        # Create a grid of coordinates for the slice
        x, y = np.meshgrid(np.arange(displacement_slice.shape[1]), np.arange(displacement_slice.shape[0]))
        
        # Plot the vector field (displacement) using quiver
        plt.figure(figsize=(10, 8))
        plt.quiver(x, y, displacement_slice[:, :, 0], displacement_slice[:, :, 1], scale=10, color='blue')
        plt.title("2D Transformation Vector Field (Displacement) - Middle Slice")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    else:
        print("The displacement field is not in 2D format. Ensure the image is 2D.")

# Example usage:
image_path = "elastix_bspline/951.npy"  # Path to your .npy image
transform_param_file = 'log_file/TransformParameters.0.txt' # Path to your TransformParameters.0.txt

visualize_transform_as_vector_field_2d_npy(image_path, transform_param_file)