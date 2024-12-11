import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from os import listdir

#Doesn't work yet



# Visualize the fixed and moving images
def Visualization_input(fixed_image, moving_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sitk.GetArrayViewFromImage(fixed_image)[fixed_image.GetSize()[2]//2, :, :], cmap='gray')
    plt.title('Fixed Image')
    plt.subplot(1, 2, 2)
    plt.imshow(sitk.GetArrayViewFromImage(moving_image)[moving_image.GetSize()[2]//2, :, :], cmap='gray')
    plt.title('Moving Image')
    plt.show()

def Registration(fixed_image, moving_image):
    # Set up the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Use Mutual Information for the metric
    registration_method.SetMetricAsMattesMutualInformation()
    registration_method.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.SampleBased)
    registration_method.SetMetricSamplingPercentage(100)
    registration_method.SetNumberOfIterations(200)

    # Set the optimizer (GradientDescent)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200)

    # Set the transform (BSpline Transform)
    bspline_transform = sitk.BSplineTransformInitializer(fixed_image, [5, 5])  # Grid resolution of 5

    registration_method.SetInitialTransform(bspline_transform)

    # Set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Perform the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Apply the final transformation to the moving image
    resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0)

    return resampled_image

# Visualize the result
def Visualization_resampled(fixed_image, resampled_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sitk.GetArrayViewFromImage(fixed_image)[fixed_image.GetSize()[2]//2, :, :], cmap='gray')
    plt.title('Fixed Image')

    plt.subplot(1, 2, 2)
    plt.imshow(sitk.GetArrayViewFromImage(resampled_image)[resampled_image.GetSize()[2]//2, :, :], cmap='gray')
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
        plt.savefig('Registration_bspline_5/' + name_list[s] + '.jpg')

        if check_its_working:
             pass
             #break # it might makes sense to stop after one registration and to check your results
        
        check_its_working = True

        
    return return_image_list


if __name__ == "__main__":

    #Doesn't work yet
    

    #Visualization_input(fixed_image, moving_image)
    res_img_list = main()
    #Visualization_resampled(fixed_image, res_img)

    print('end of programm')