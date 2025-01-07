import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to perform B-spline image registration
def Registration(fixed_image, moving_image):
    elastix_image_filter = sitk.ElastixImageFilter()
    
    # Define the B-spline registration parameters
    parameter_map = sitk.GetDefaultParameterMap('bspline')
    #parameter_map['Transform'] = ['BSplineTransform']
    parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    parameter_map['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    parameter_map['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']
    parameter_map['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
    parameter_map['NumberOfResolutions'] = ['5']
    parameter_map['MaximumNumberOfIterations'] = ['10000']  # Increase iterations
    parameter_map['GridSpacing'] = ['1', '1', '1']  # Finer grid spacing
    parameter_map['FinalGridSpacingInPhysicalUnits'] = ['1', '1', '1']
    parameter_map['SplineOrder'] = ['5']
    parameter_map['Interpolator'] = ['BSplineInterpolator']
    parameter_map['BSplineInterpolationOrder'] = ['3', '3', '3', '3']
    parameter_map['GridSpacingSchedule'] = ['5.0', '5.0', '5.0',  # for resolution level 1
                                            '4.0', '4.0', '4.0',  # for resolution level 2
                                            '3.0', '3.0', '3.0',  # for resolution level 3
                                            '2.0', '2.0', '2.0',  # for resolution level 4
                                            '1.0', '1.0', '1.0']  # for resolution level 5

    parameter_map['InitialTransformParametersFileName'] = ['']
    parameter_map['Transform'] = ['BSplineTransform']  # First affine, then BSpline

    
    elastix_image_filter.SetParameterMap([parameter_map])
    elastix_image_filter.SetFixedImage(fixed_image)
    elastix_image_filter.SetMovingImage(moving_image)
    elastix_image_filter.Execute()
    
    return elastix_image_filter.GetResultImage()

# Function to visualize the images
def Visualization(fixed_array, registered_array, moving):
    slice_idx = 8  # You can change this slice index
    
    # Create a figure with 3 subplots (one for each image)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the fixed, registered, and moving images (all slices in the z-axis)
    axes[0].imshow(fixed_array[slice_idx, :, :], cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Fixed Image Slice')
    
    axes[1].imshow(registered_array[slice_idx, :, :], cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Registered Moving Image Slice')
    
    axes[2].imshow(moving[slice_idx, :, :], cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Moving Image Slice')

    plt.tight_layout()
    plt.savefig('elastix_3d/3d-reg.jpg')

# Function to load 3D data from .npy files
def get3d_48_62(timeslice):
    if timeslice < 0 or timeslice > 24:
        raise IndexError('slices only between 0 and 25')
    
    vectorOfSubImages = sitk.VectorOfImage()
    path = 'results_slices/4dflow_KI_20241207-124205/'

    for k in range(48, 63):  # Slice numbers between 48 and 62
        slice_num = 109 - k
        slicenumber = slice_num * 25 + timeslice
        k_path = os.path.join(path, str(k), f'{slicenumber}.npy')
        
        # Load .npy file and convert to SimpleITK image
        try:
            arr_loaded = np.load(k_path)
            arr_img = sitk.GetImageFromArray(np.float32(arr_loaded))
            vectorOfSubImages.push_back(arr_img)
        except FileNotFoundError:
            print(f"Warning: {k_path} not found. Skipping slice.")
    
    return sitk.JoinSeries(vectorOfSubImages)

# Main function to perform the registration
def main():
    time_fixed = 12
    time_moving = 7
    
    # Load fixed and moving 3D images
    image_fixed = get3d_48_62(time_fixed)
    image_moving = get3d_48_62(time_moving)

    # preprocess
    """image_fixed = sitk.SmoothingRecursiveGaussian(image_fixed, sigma=2)
    image_moving = sitk.SmoothingRecursiveGaussian(image_moving, sigma=2)"""

    # masking
    """fixed_image_mask = sitk.OtsuThreshold(fixed_image, 0, 1)
    moving_image_mask = sitk.OtsuThreshold(moving_image, 0, 1)"""

    # Perform registration
    reg_image = Registration(image_fixed, image_moving)
    reg_image_array = sitk.GetArrayFromImage(reg_image)
    
    # Save the registered image as .npy
    np.save('elastix_3d/7_to_12.npy', reg_image_array)
    return reg_image_array

# Function to load and visualize 3D images
def load_3d_and_visualize():
    img_path = 'elastix_3d/7_to_12.npy'
    reg_image = np.load(img_path)
    
    # Load fixed image at a specific time
    time_fixed = 12
    time_moving = 7
    
    # Load fixed and moving 3D images
    image_fixed = get3d_48_62(time_fixed)
    image_moving = get3d_48_62(time_moving)
    image_fixed_array = sitk.GetArrayFromImage(image_fixed)
    image_moving = sitk.GetArrayFromImage(image_moving)
    # Visualize fixed and registered images
    Visualization(image_fixed_array, reg_image, image_moving)

if __name__ == "__main__":
    # Perform registration and visualize the result

    # using thinplate? 

    #return_image_list = main()  # Perform registration
    load_3d_and_visualize()  # Visualize fixed and registered images

    print("End of program")
