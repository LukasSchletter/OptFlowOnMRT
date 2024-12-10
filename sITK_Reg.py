import SimpleITK as sitk
import sys
#import os
import matplotlib.pyplot as plt
#import itk
from configparser import ConfigParser
#import itkJPEGImageIO
#include <itkJPEGImageIO.h>
from os import listdir
from packaging.version import parse
from importlib.metadata import version
import numpy as np
#from itkwidgets import view
from tqdm import tqdm
#matplotlib inline
from ipywidgets import interact

if parse(version('itk')) < parse('5.3'):
    raise ValueError("ITK greater than version 5.3.0 is required for this notebook")

def registration(fixed, moving):
    #### Simple Elastix
        
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed)
        elastixImageFilter.SetMovingImage(moving)
        parameterMapVector = sitk.VectorOfParameterMap()
        parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
        parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
        elastixImageFilter.SetParameterMap(parameterMapVector)

        print('printing parameter map:')
        elastixImageFilter.PrintParameterMap()

        #exit()

        elastixImageFilter.Execute()
        resultImage = elastixImageFilter.GetResultImage()

    
        print('only one : this will be returned')
        return resultImage


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
    
    return_image_list = []
    for s in tqdm(range(0, len(name_list), 1)):
        string = directory_path + '/' + name_list[s] + '.npy'
        print('tqdm')
        
        print(string)
        
        movarr_loaded = np.load(string)
    
        movarr_loaded = np.float32(movarr_loaded)
        
        moving = sitk.GetImageFromArray(movarr_loaded)
        
        return registration(fixed, moving)
        # return_image_list.append(registration(fixed, moving))
        break # it might makes sense to stop after one registration and to check your results
        
    return return_image_list


if __name__ == "__main__":
    print('main:')
    return_image = main()
    return_image = sitk.GetArrayFromImage(return_image)
    plt.imshow(return_image)
    plt.savefig('sITK_Reg.jpg')
    print('end of program')





"""  for s in range(0, len(arr_list), 1):
if(name_list[s] == '951'):
print('fixed')



arr_loaded = np.float32(arr_loaded)
fixed = sitk.GetImageFromArray(arr_loaded)"""