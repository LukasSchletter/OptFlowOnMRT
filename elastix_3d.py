import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from Visualization_3d import load_3d_and_visualize
from Visualization_3d import get3d_48_62
from Visualization_3d import Visualization

def generate_mask(image, mask_filename):
    # Apply Otsu thresholding to create a binary mask
    mask = sitk.OtsuThreshold(image, 0, 1)
    
    # Perform morphological operations to improve the mask
    mask_filled = sitk.BinaryFillhole(mask)  # Fill any holes in the mask
    mask_smooth = sitk.BinaryErode(mask_filled, [2, 2, 2])  # Optional: remove small artifacts with erosion
    
    # Save the mask image
    sitk.WriteImage(mask_smooth, mask_filename)
    
    return mask_smooth

def visualize_mask(image, mask, title='Mask Visualization'):
    # Convert SimpleITK images to numpy arrays
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    # Choose a slice to visualize (middle slice)
    mid_slice = image_array.shape[0] // 2

    # Plot the original image slice and the mask overlay
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display original image slice
    ax[0].imshow(image_array[mid_slice, :, :], cmap='gray')
    ax[0].set_title(f"Original Image (Slice {mid_slice})")
    ax[0].axis('off')

    # Display mask overlay
    ax[1].imshow(image_array[mid_slice, :, :], cmap='gray')
    ax[1].imshow(mask_array[mid_slice, :, :], cmap='jet', alpha=0.5)  # Mask overlay with transparency
    ax[1].set_title(f"Mask Overlay (Slice {mid_slice})")
    ax[1].axis('off')

    plt.suptitle(title)
    #plt.imshow()
    plt.savefig('elastix_3d/masks/test.jpg')
    print('masks saved')

def Registration(fixed_image, moving_image):
    elastix_image_filter = sitk.ElastixImageFilter()

    # Define file paths to save the mask images
    fixed_image_mask_path = 'fixed_image_mask.mha'
    moving_image_mask_path = 'moving_image_mask.mha'

    # Generate masks for both fixed and moving images using Otsu thresholding
    fixed_image_mask = generate_mask(fixed_image, fixed_image_mask_path)
    moving_image_mask = generate_mask(moving_image, moving_image_mask_path)

    # Visualize the generated masks for both fixed and moving images
    visualize_mask(fixed_image, fixed_image_mask, title="Fixed Image Mask")
    visualize_mask(moving_image, moving_image_mask, title="Moving Image Mask")

    # Set the masks for elastix image filter using the ParameterMap
    affine_parameter_map = sitk.GetDefaultParameterMap('affine')
    affine_parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    affine_parameter_map['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    affine_parameter_map['MaximumNumberOfIterations'] = ['2000']
    affine_parameter_map['RelaxationFactor'] = ['0.5']
    affine_parameter_map['LearningRate'] = ['1.0']
    #affine_parameter_map['FixedImagePyramid'] = ['FixedIdentityImagePyramid']  # No smoothing pyramid
    #affine_parameter_map['MovingImagePyramid'] = ['MovingIdentityImagePyramid']  # No smoothing pyramid
    affine_parameter_map['NumberOfResolutions'] = ['5']
    affine_parameter_map['InitialTransformParametersFileName'] = ['']
    affine_parameter_map['Transform'] = ['AffineTransform']

    # Add mask file paths to parameter map
    #affine_parameter_map['FixedImageMask'] = [fixed_image_mask_path]
    #affine_parameter_map['MovingImageMask'] = [moving_image_mask_path]

    #exit()
    # Perform affine registration first
    elastix_image_filter.SetParameterMap([affine_parameter_map])
    elastix_image_filter.SetFixedImage(fixed_image)
    elastix_image_filter.SetMovingImage(moving_image)
    elastix_image_filter.Execute()

    affine_result_image = elastix_image_filter.GetResultImage()
    print("Affine Registration complete.")

    #affine_result_image = moving_image #elastix_image_filter.GetResultImage()
    

    # Now apply B-spline registration for finer tuning
    bspline_parameter_map = sitk.GetDefaultParameterMap('bspline')
    bspline_parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    bspline_parameter_map['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    bspline_parameter_map['MaximumNumberOfIterations'] = ['50000']
    bspline_parameter_map['RelaxationFactor'] = ['0.1']  # Try smaller relaxation factor
    bspline_parameter_map['LearningRate'] = ['0.01']     # Try smaller learning rate
    #bspline_parameter_map['SmoothingSigmas'] = ['16', '8', '4', '2', '1', '0']
    #bspline_parameter_map['FixedImagePyramid'] = ['FixedIdentityImagePyramid']  # No smoothing pyramid
    #bspline_parameter_map['MovingImagePyramid'] = ['MovingIdentityImagePyramid']  
    bspline_parameter_map['NumberOfResolutions'] = ['1']
    #bspline_parameter_map['FinalGridSpacingInPhysicalUnits'] = ['1', '1', '1']

    """    # Set initial GridSpacingSchedule for the first resolution level
        bspline_parameter_map['GridSpacingSchedule'] = [
            '32.0', '32.0', '32.0',  # Resolution level 1 (coarse)
            '16.0', '16.0', '16.0',  # Resolution level 2
            '8.0', '8.0', '8.0',     # Resolution level 3
            '4.0', '4.0', '4.0',     # Resolution level 4
            '2.0', '2.0', '2.0',     # Resolution level 5
            '1.0', '1.0', '1.0'      # Resolution level 6 (fine)
        ]
    """

    # Set mask paths for B-spline registration as well
    #bspline_parameter_map['FixedImageMask'] = [fixed_image_mask_path]
    #bspline_parameter_map['MovingImageMask'] = [moving_image_mask_path]


    # Create a folder to store intermediate results
    intermediate_results_folder = 'elastix_3d/IntermediateResults/'
    if not os.path.exists(intermediate_results_folder):
        os.makedirs(intermediate_results_folder)

    # Set the fixed and moving images and start registration process
    elastix_image_filter.SetParameterMap([bspline_parameter_map])
    elastix_image_filter.SetFixedImage(fixed_image)
    

    current_result_image = affine_result_image
    # Define SmoothingSigmas for each resolution level
    #smoothing_sigmas = ['16', '8', '4', '2', '1', '0']  # Define it globally for each resolution

    # Run registration at each resolution level
    for resolution_level in range(6):
        print(f"Processing resolution level {resolution_level + 1}...")
        elastix_image_filter.SetMovingImage(current_result_image)  # Use the affine-registered image
          # Dynamically adjust the GridSpacingSchedule for each resolution level
        grid_spacing_schedule = ['16.0', '16.0', '16.0']  # Resolution level 1
        if resolution_level == 1:
            grid_spacing_schedule = ['8.0', '8.0', '8.0']  # Resolution level 2
        elif resolution_level == 2:
            grid_spacing_schedule = ['4.0', '4.0', '4.0']  # Resolution level 3
        elif resolution_level == 3:
            grid_spacing_schedule = ['2.0', '2.0', '2.0']  # Resolution level 4
        elif resolution_level == 4:
            grid_spacing_schedule = ['1.0', '1.0', '1.0']  # Resolution level 5
        elif resolution_level == 5:
            grid_spacing_schedule = ['0.5', '0.5', '0.5']  # Fine resolution level
        print('grid')
        print(int(bspline_parameter_map['NumberOfResolutions'][0]))
        print(grid_spacing_schedule)
        
        bspline_parameter_map['GridSpacingSchedule'] = grid_spacing_schedule
       # bspline_parameter_map['SmoothingSigmas'] = [smoothing_sigmas[resolution_level]]  # Update smoothing sigmas
        # Set the updated parameter map for the current resolution level
        elastix_image_filter.SetParameterMap([bspline_parameter_map])

        # Perform registration for this resolution level
        elastix_image_filter.Execute()

        # Get and save the intermediate result image at this resolution level
        current_result_image = elastix_image_filter.GetResultImage()
        print(f"Resolution Level {resolution_level + 1} Result")

        # Save intermediate result
        np.save(os.path.join(intermediate_results_folder, f"intermediate_result_level_{resolution_level + 1}.npy"), sitk.GetArrayFromImage(current_result_image))

        visualization_intermediate_results_folder = 'elastix_3d/IntermediateResults/'+str(resolution_level)
        # Optionally, you can visualize the intermediate results (e.g., using Visualization function)
        visualize_registration(fixed_image, moving_image, current_result_image, visualization_intermediate_results_folder)

    return elastix_image_filter.GetResultImage()

def visualize_registration(fixed_image, moving_image, current_result_image, folder_path):
    fixed = sitk.GetArrayFromImage(fixed_image)
    moving = sitk.GetArrayFromImage(moving_image)
    reg = sitk.GetArrayFromImage(current_result_image)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Visualization function to display the images
    Visualization(fixed, reg, moving, folder_path)



# Main function to perform the registration
def main():
    time_fixed = 7
    time_moving = 12
    
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
    #np.save('elastix_3d/Affine_7_to_12.npy', reg_image_array)
    return reg_image_array


if __name__ == "__main__":
    # Perform registration and visualize the result

    # using thinplate? 

    return_image_list = main()  # Perform registration
    #load_3d_and_visualize()  # Visualize fixed and registered images

    print("End of program")
