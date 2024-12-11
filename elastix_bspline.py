import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from os import listdir

def Registration(fixed_image, moving_image):

    # Create a SimpleElastix object
    elastix_image_filter = sitk.ElastixImageFilter()

    # Set the path to the Elastix executable
    #elastix_image_filter.SetElastixExecutable(elastix_path)

    # Define B-spline registration parameters via the parameter file
    # In this case, we are using a B-spline transform with a grid resolution of 5
    parameter_map = sitk.GetDefaultParameterMap('bspline')
    parameter_map['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']
    parameter_map['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
    
    parameter_map['Transform'] = ['BSplineTransform']
    parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    parameter_map['Optimizer'] = ['AdaptiveStochasticGradientDescent']
    parameter_map['NumberOfResolutions'] = ['4']
    parameter_map['MaximumNumberOfIterations'] = ['2000']
    parameter_map['Interpolator'] = ['BSplineInterpolator']
    parameter_map['BSplineInterpolationOrder'] = ['3', '3', '3', '3']
    #parameter_map['GridSpacing'] = ['5']  # B-spline grid spacing (resolution 5)
    #parameter_map['GridSpacingSchedule'] = [' 5.0, 5.0, 4.0, 4.0, 2.0 , 2.0 , 1.0, 1.0']
    parameter_map['FinalGridSpacingInPhysicalUnits'] = ['5']
    parameter_map['FinalBSplineInterpolationOrder'] = ['5']
    #parameter_map['PassiveEdgeWidth'] = ['4, 4'] 

    # Add the parameter map to the filter
    elastix_image_filter.SetParameterMap([parameter_map])
    print('printing parameter map:')
    elastix_image_filter.PrintParameterMap()

    #exit()
    # Perform the registration
    elastix_image_filter.SetFixedImage(fixed_image)
    elastix_image_filter.SetMovingImage(moving_image)
    elastix_image_filter.Execute()

    # Get the result of the registration
    registered_image = elastix_image_filter.GetResultImage()

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


def main():
  
    directory_path = 'results_slices/4dflow_KI_20241207-124205/71' # 951 - 975
   
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
        plt.savefig('elastix_bspline_gridspacing/' + name_list[s] + '.jpg')
        np.save('elastix_bspline_gridspacing/' + name_list[s] + '.npy', reg_img)
        if check_its_working:
             pass
             #break # it might makes sense to stop after one registration and to check your results
        
        check_its_working = True

        
    return return_image_list



if __name__ == "__main__":

  
    print('main:')
    return_image_list = main()
    #return_image = sitk.GetArrayFromImage(return_image)
    
    print('end of program')