import SimpleITK as sitk
import numpy as np

def calculate_sse(image1, image2):
    """
    Calculate the Sum of Squared Errors (SSE) between two images using SimpleITK.
    
    Parameters:
        image1 (sitk.Image): The first image.
        image2 (sitk.Image): The second image.
    
    Returns:
        float: The Sum of Squared Errors between the two images.
    """
    
    # Convert images to numpy arrays
    image1_array = sitk.GetArrayFromImage(image1)
    image2_array = sitk.GetArrayFromImage(image2)
    
    # Check that the images are the same size
    if image1_array.shape != image2_array.shape:
        raise ValueError("Images must have the same dimensions and size")
    
    # Compute the sum of squared errors (SSE)
    sse = np.sum((image1_array - image2_array) ** 2)
    
    return sse

# Example usage
# Load two .npy files
image1_np = np.load("elastix_bspline/951.npy")  # Replace with your actual path
image2_np = np.load("elastix_bspline/952.npy")  # Replace with your actual path

# Convert the numpy arrays to SimpleITK images with the correct pixel type
image1 = sitk.GetImageFromArray(image1_np.astype(np.float32))
image2 = sitk.GetImageFromArray(image2_np.astype(np.float32))

# Calculate SSE
sse_value = calculate_sse(image1, image2)
print(f"Sum of Squared Errors (SSE) between the images: {sse_value}")
