import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from os import listdir



def Registration(fixed_image, moving_image):
    # Initialize the SimpleElastix filter
    elastix_image_filter = sitk.ElastixImageFilter()

    # Set the path to the Elastix executable
    #elastix_image_filter.SetElastixExecutable(elastix_path)

    # Define B-spline registration parameters for 3D images with grid resolution 5
    parameter_map = sitk.GetDefaultParameterMap('bspline')

    # Set parameter map for 3D B-spline registration
    parameter_map['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']
    parameter_map['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
    parameter_map['Transform'] = ['BSplineTransform']
    parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    parameter_map['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    parameter_map['NumberOfResolutions'] = ['4']  # Use multi-resolution strategy
    parameter_map['MaximumNumberOfIterations'] = ['2000']  # Max iterations per resolution

    # For B-spline grid resolution (order 5), set the grid spacing
    parameter_map['GridSpacing'] = ['5', '5', '5']  # 5x5x5 grid spacing for 3D
    parameter_map['FinalGridSpacingInPhysicalUnits'] = ['5', '5', '5']

    # Set the order of the B-spline transform
    parameter_map['SplineOrder'] = ['5']  # B-spline order 5

    # Add the parameter map to the Elastix filter
    elastix_image_filter.SetParameterMap([parameter_map])

    # Perform the registration
    elastix_image_filter.SetFixedImage(fixed_image)
    elastix_image_filter.SetMovingImage(moving_image)
    elastix_image_filter.Execute()

    # Get the result of the registration
    registered_image = elastix_image_filter.GetResultImage()

    # Save the registered image
    #sitk.WriteImage(registered_image, 'registered_image_3d.nii')
    return registered_image
    # Visualize the result (select a slice from the 3D images to show)
    # Display slices in the z-axis direction for visualization
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    registered_array = sitk.GetArrayFromImage(registered_image)

# Visualize a middle slice from the z-axis
def Visualization(fixed_array, registered_array):
    slice_idx = 8 #fixed_array.GetSize()[2] // 2

    """plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_array[slice_idx, :, :], cmap='gray')
    plt.title('Fixed Image Slice')

    plt.subplot(1, 2, 2)
    plt.imshow(registered_array[slice_idx, :, :], cmap='gray')
    plt.title('Registered Moving Image Slice')

    plt.subplot(1, 2, 3)
    plt.imshow(registered_array[slice_idx, :, :], cmap='gray')
    plt.title('Registered Moving Image Slice')"""


    # Create a figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the images
    axes[0].imshow(fixed_array[slice_idx, :, :], cmap='gray')
    axes[0].axis('off')  # Turn off axis
    axes[0].set_title('Fixed Image Slice')

    axes[1].imshow(registered_array[slice_idx, :, :], cmap='gray')
    axes[1].axis('off')  # Turn off axis
    axes[1].set_title('Registered Moving Image Slice')

    axes[2].imshow(registered_array[slice_idx, :, :], cmap='gray')
    axes[2].axis('off')  # Turn off axis
    axes[2].set_title('Moving Image Slice')

    plt.savefig('elastix_3d/3d-reg.jpg')


def get3d_48_62(timeslice):
    # get slices between 48 and 62
    if(timeslice < 0 or timeslice > 24):
        raise IndexError('slices only between 0 and 25')
    vectorOfSubImages = sitk.VectorOfImage()

    path = 'results_slices/4dflow_KI_20241207-124205/'
    for k in range(48,63,1):
        
        #print('slice: '  + str(k))
        slice = 109 - k
        #print('transformed slice: ' + str(slice))
        slicenumber = slice*25 + timeslice
        k_path = path + str(k) + '/' + str(slicenumber) +'.npy'
        print(k_path)
        arr_loaded = np.load(k_path) 
        arr_loaded = np.float32(arr_loaded)
        arr_img = sitk.GetImageFromArray(arr_loaded)
        vectorOfSubImages.push_back(arr_img)
        #print(list[slicenumber + timeslice][0])
        
    image = sitk.JoinSeries(vectorOfSubImages)
   
    return(image)  

def main():
  
    time_fixed = 12
    time_moving = 7
    image_fixed = get3d_48_62(time_fixed)
    #image_fixed_array = sitk.GetArrayFromImage(image_fixed)
    #print(image_fixed_array.shape)
    #print(image_fixed.GetDimension())
    image_moving = get3d_48_62(time_moving)
    reg_image = Registration(image_fixed, image_moving)
    reg_image = sitk.GetArrayFromImage(reg_image)
    np.save('elastix_3d/' + '7_to_12' + '.npy', reg_image)

    return 1

    # get fixed - it's better to take an images in the middle, i.e. 951+12 = 963
    arr_loaded = np.load(directory_path + '/963.npy') 
    fixed = sitk.GetImageFromArray(arr_loaded)
    
    name_list = []
    for dir_content in listdir(directory_path):

        string = dir_content
        string_2 = string.removesuffix('.npy')
        name_list.append(string_2)
    
    print(name_list)
    
    check_its_working = False

    return_image_list = []
    for s in tqdm(range(0, len(name_list), 1)):
        string = directory_path + '/' + name_list[s] + '.npy'
        print('tqdm')
        
        print(string)
        
        movarr_loaded = np.load(string)
    
        movarr_loaded = np.float32(movarr_loaded)
        
        moving = sitk.GetImageFromArray(movarr_loaded)
        
        reg_img = Registration(fixed, moving)
        reg_img = sitk.GetArrayFromImage(reg_img)
        #return registration(fixed, moving)
        return_image_list.append(reg_img)

        plt.imshow(reg_img)
        plt.savefig('elastix_bspline/' + name_list[s] + '.jpg')
        np.save('elastix_bspline/' + name_list[s] + '.npy', reg_img)
        if check_its_working:
             pass
             #break # it might makes sense to stop after one registration and to check your results
        
        check_its_working = True

        
    return return_image_list


def load_3d_and_visualize():
    img_path = 'elastix_3d/7_to_12.npy'
    img = np.load(img_path)

    time_fixed = 12
    #time_moving = 7
    image_fixed = get3d_48_62(time_fixed)
    image_fixed = sitk.GetArrayFromImage(image_fixed)
    print(image_fixed.shape)
    Visualization(image_fixed, img)

if __name__ == "__main__":

    plt.axis('off')
    #load_3d_and_visualize()
    print('main:')
    return_image_list = main()
    print(return_image_list)
    #return_image = sitk.GetArrayFromImage(return_image)
    
    

    
    
    
    print('end of program')

    exit()

    # Paths to your 3D images
    fixed_image_path = 'fixed_image.nii'  # Path to fixed 3D image
    moving_image_path = 'moving_image.nii'  # Path to moving 3D image

    # Path to Elastix executable (adjust according to your installation)
    #elastix_path = 'path/to/elastix'  # Replace with the path to elastix executable

    # Create SimpleITK image objects for fixed and moving images
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
  
    print('main:')
    return_image_list = main()
    #return_image = sitk.GetArrayFromImage(return_image)""