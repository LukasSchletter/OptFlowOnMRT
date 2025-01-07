import re
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from os import listdir
import sys
import subprocess
import collections
from pathlib import Path

def delete_line_from_file(file_path, line_number):
    """Delete a specific line (by line number) from a text file."""
    try:
        # Open the file and read all lines
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Check if the line number is valid
        if line_number <= 0 or line_number > len(lines):
            print(f"Error: Line number {line_number} is out of range.")
            return

        # Remove the specified line
        lines.pop(line_number - 1)  # line_number is 1-based, so adjust it for 0-based index
        
        # Open the file in write mode and overwrite with the remaining lines
        with open(file_path, 'w') as file:
            file.writelines(lines)
        
        print(f"Line {line_number} has been deleted from '{file_path}'.")

    except Exception as e:
        print(f"Error: {e}")

def Registration(fixed_image, moving_image, output_dir):
     # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a SimpleElastix object
    elastix_image_filter = sitk.ElastixImageFilter()


    # Set the path to the Elastix executable
    #elastix_image_filter.SetElastixExecutable(elastix_path)

    # Define B-spline registration parameters via the parameter file
    # In this case, we are using a B-spline transform with a grid resolution of 5
    # Define B-spline registration parameters via the parameter file
    parameter_map = sitk.GetDefaultParameterMap('bspline')
    parameter_map['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']
    parameter_map['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
    
    # Define the metric, optimizer, and interpolator
    parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    parameter_map['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    parameter_map['NumberOfResolutions'] = ['5']  # Number of multi-resolution levels
    parameter_map['MaximumNumberOfIterations'] = ['10000']
    parameter_map['Interpolator'] = ['BSplineInterpolator']
    parameter_map['BSplineInterpolationOrder'] = ['3', '3']  # 2D interpolation
    parameter_map['FinalGridSpacingInPhysicalUnits'] = ['1']
    parameter_map['InitialTransformParametersFileName'] = ['']
    
   
    # Set GridSpacingSchedule: The number of entries should be NumberOfResolutions * ImageDimension
    # For 2D images and 5 resolutions, we need 10 values (5 * 2)
    parameter_map['GridSpacingSchedule'] = ['5.0', '5.0',  # for resolution level 1
                                            '4.0', '4.0',  # for resolution level 2
                                            '3.0', '3.0',  # for resolution level 3
                                            '2.0', '2.0',  # for resolution level 4
                                            '1.0', '1.0']  # for resolution level 5
    # Add the parameter map to the filter
    elastix_image_filter.SetParameterMap([parameter_map])
    print('printing parameter map:')
    
    # Specify Affine followed by B-spline transformation
    parameter_map['Transform'] = ['BSplineTransform']  # Affine, then B-spline

    # Print statements will be saved to 'output.log' file

    elastix_image_filter.PrintParameterMap()

   
    #exit()
    # Perform the registration
    elastix_image_filter.SetFixedImage(fixed_image)
    elastix_image_filter.SetMovingImage(moving_image)

    # Set the output directory
    elastix_image_filter.SetOutputDirectory(output_dir)
  
   
    elastix_image_filter.Execute()


    # Get the result of the registration
    registered_image = elastix_image_filter.GetResultImage()

   # metric_value = elastix_image_filter.Get
    # Assuming you have a registered transform after using ImageRegistrationMethod
    transform = elastix_image_filter.GetTransformParameterMap()
    print(transform)

    # Save the transform
    #elastix_image_filter.WriteParameterFile(elastix_image_filter.GetParameterMap(), 'ParameterMap.txt')
    #sitk.WriteTransform(transform, "TransformParameters.0.txt")
    
    return registered_image
    # Save the registered image
    #sitk.WriteImage(registered_image, 'registered_image.nii')

# Visualize the results

def Visualization(fixed_image, registered_image):

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sitk.GetArrayViewFromImage(fixed_image)[fixed_image.GetSize()[2] // 2, :, :], cmap='gray')
    plt.title('Fixed Image')

    plt.subplot(1, 2, 2)
    plt.imshow(sitk.GetArrayViewFromImage(registered_image)[registered_image.GetSize()[2] // 2, :, :], cmap='gray')
    plt.title('Registered Moving Image')

    plt.show()

def Normalization(img):
    norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    return norm

def extract_metric_value_from_log(output_dir):
    # Try to read the console output file if the log is missing
    console_log_path = os.path.join(output_dir, "elastix_console_output.txt")
    
    # Initialize the metric value
    metric_value = None
    
    # Read the console output file and look for the final metric value
    if os.path.exists(console_log_path):
        with open(console_log_path, "r") as log_file:
            log_content = log_file.read()
            
            # Print the entire log to help with debugging (optional)
            print("Elastix console log content:\n")
            print(log_content)
            
            # Try matching the specific phrase for the metric value
            match = re.search(r"Final metric value\s*=\s*(-?[0-9\.]+)", log_content)
            
            if match:
                metric_value = float(match.group(1))  # Extract the numeric value
                print(f"Extracted Metric Value: {metric_value}")
            else:
                print("Final metric value not found in console log.")
    
    # Return the extracted metric value or None if not found
    return metric_value

def main():
  
    directory_path = 'results_slices/4dflow_KI_20241207-124205/71' # 951 - 975
   
    # get fixed - it's better to take an images in the middle, i.e. 951+12 = 963
    arr_loaded = np.load(directory_path + '/963.npy') 
    #arr_loaded = Normalization(arr_loaded)
    fixed = sitk.GetImageFromArray(arr_loaded)
    
    name_list = []

    #name = input("Please enter the name under which we store the output_file: ")

    for dir_content in listdir(directory_path):

        string = dir_content
        string_2 = string.removesuffix('.npy')
        name_list.append(string_2)
    
    print(name_list)
    
    check_its_working = False

    return_image_list = []

     # List to store the metric values
    
    output_dir = "log_file"
    for s in tqdm(range(0, len(name_list), 1)):
        my_file = Path("elastix_bspline_gridspacing/" + name_list[s] + ".npy")
        if my_file.is_file():
            print(name_list[s] + " exists")
            continue
        
            
        string = directory_path + '/' + name_list[s] + '.npy'
        print('tqdm')
        
        print(string)
        
        movarr_loaded = np.load(string)
    
        movarr_loaded = np.float32(movarr_loaded)
        #movarr_loaded = Normalization(movarr_loaded)
        moving = sitk.GetImageFromArray(movarr_loaded)
        
        reg_img = Registration(fixed, moving, output_dir)

        path_log = 'log_file/TransformParameters.0.txt'
        #delete_line_from_file(path_log, 20)

        reg_img = sitk.GetArrayFromImage(reg_img)
        #return registration(fixed, moving)
        return_image_list.append(reg_img)
        

        plt.imshow(reg_img)
        plt.savefig('elastix_bspline_gridspacing/' + name_list[s] + '.jpg')
                   
        np.save('elastix_bspline_gridspacing/' + name_list[s] + '.npy', reg_img)
        if check_its_working:
             pass
             #break # it might makes sense to stop after one registration and to check your results
        
        check_its_working = True
        #exit()
        
    return return_image_list



if __name__ == "__main__":

  
    print('main:')
    return_image_list = main()
    #return_image = sitk.GetArrayFromImage(return_image)
    
    print('end of program')