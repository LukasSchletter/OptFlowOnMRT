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

"""def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f" = {method.GetMetricValue():7.5f} "
        + f" : {method.GetOptimizerPosition()}"
    )

def writing(data):
    with open("output.txt", "w") as txt_file:
        for number in data:
           txt_file.write(" ".join(str(number)) ) # + "\n")"""

def main(args):
    """ if len(args) < 3:
        print(
            "Usage:",
            "ImageRegistrationMethod2",
            "<fixedImageFilter> <movingImageFile>  <outputTransformFile>",
        )
        print(args[0])
        print(args[1])
        print(args[2])
        sys.exit(1)
   """


    arr_list = []
    name_list = []
    image_npy_list = []
    # middle slices = 48 - 62
    directory_path = 'results_slices/4dflow_KI_20241207-124205/71'
    return_list = []
    #name_list.append('reg')
    for dir_content in listdir(directory_path):
        
        movarr_loaded = np.load(directory_path +'/' + dir_content)
        movarr_loaded = np.float32(movarr_loaded)
        moving = sitk.GetImageFromArray(movarr_loaded)
        arr_list.append(movarr_loaded)
        string = dir_content
        string_2 = string.removesuffix('.npy')
        name_list.append(string_2)
        image_npy_list.append(moving)
        print(dir_content)
    
    print(name_list)
    
    """ print(name_list)
    print(image_npy_list)
    print(arr_list)"""
    #moving = sitk.ReadImage(args[2], sitk.sitkFloat32)

    
    # get fixed
    for s in range(0, len(arr_list), 1):
        if(name_list[s] == '951'):
            print('fixed')
            #fixed = sitk.ReadImage(args[1], sitk.sitkFloat32)
            arr_loaded = np.load(directory_path + '/951.npy')
            """ arr_loaded = np.delete(arr_loaded,np.s_[0:25],1)
            arr_loaded = np.delete(arr_loaded,np.s_[105:],1)
            arr_loaded = np.delete(arr_loaded,np.s_[0:45],0)
            arr_loaded = np.delete(arr_loaded,np.s_[105:],0)"""
            fixed = sitk.GetImageFromArray(arr_loaded)
            #print(fixed.GetPixelIDTypeAsString())
            arr_loaded = np.float32(arr_loaded)
            fixed = sitk.GetImageFromArray(arr_loaded)
            #print(fixed.GetPixelIDTypeAsString())

    
    regcounter = 0

    for s in tqdm(range(0, len(arr_list), 1)):
        string = directory_path + '/' + name_list[s] + '.npy'
        print(string)
        #print(test)
        
        movarr_loaded = np.load(string)
        """movarr_loaded = np.delete(movarr_loaded,np.s_[0:25],1)
        movarr_loaded = np.delete(movarr_loaded,np.s_[105:],1)
        movarr_loaded = np.delete(movarr_loaded,np.s_[0:45],0)
        movarr_loaded = np.delete(movarr_loaded,np.s_[105:],0)"""
        """movarr_loaded[0:40,:] = 255
        movarr_loaded[145:,:] = 255
        movarr_loaded[:,:25] = 255
        movarr_loaded[:,125:] = 255"""
        #print(movarr_loaded[0:35,:])
        #print(movarr_loaded.shape)
        movarr_loaded = np.float32(movarr_loaded)
        print(movarr_loaded.size)
        moving = sitk.GetImageFromArray(movarr_loaded)
        movarr = sitk.GetArrayFromImage(moving)
        print(movarr.shape)
        #return_list.append(moving)
        #print(moving.GetHeight())
        #print(moving.GetDepth())
        #moving = sitk.Normalize(moving)
        #moving = sitk.DiscreteGaussian(moving, 2.0)
        """ plt.imshow(movarr_loaded)
        plt.savefig('output/moving.png')
        # fixed/moving data
        print('fixed/moving data:')"""
        
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

        exit()

        elastixImageFilter.Execute()
        resultImage = elastixImageFilter.GetResultImage()

    
        print('only one : this will be returned')
        return_list.append(resultImage)
           
        """regcounter += 1
        if regcounter == 5:
            break"""
            
        #return {"name": name_list, "composition": return_list}
        #m = sitk.SquaredDifference(fixed,moving)
        #m_array = sitk.GetArrayFromImage(m)
       
        

        
        
    #return return_list 
    print('this will be returned')
    return {"name": name_list, "composition": return_list}
   


if __name__ == "__main__":
    print('main:')
    return_dict = main(sys.argv)
    comp_list = return_dict["composition"]
    name_list = return_dict["name"]
    counter = 0
    print('Länge return list')
    #print(len(comp_list))
    for compo in range(0, len(comp_list),1):
        #print(counter)
        counter += 1
        nda = sitk.GetArrayFromImage(comp_list[compo])
        np.save('reg_normal/'+name_list[compo], nda)
        plt.imshow(nda)
        plt.savefig('reg_normal/'+name_list[compo])
    '''return_dict = main(sys.argv)
    if "SITK_NOSHOW" not in os.environ:
        print('main returned')
        print('matplot')
        plt.imshow(return_dict["composition"])
        print('direkt composition')
        #print(return_dict["composition"])
        plt.savefig('output_itk')
        #itk.imwrite(return_dict["composition"], "composition.png")
        #sitk.imwrite(return_dict["composition"], "scomposition.png")
        img1 = sitk.ReadImage('car2.jpg')
        print('img1')
        #print(img1)
        nda = sitk.GetArrayFromImage(return_dict["composition"])
        print('nda')
        print (nda)
        #itk.imwrite(nda, "composition.png")
        writer = sitk.ImageFileWriter()
        writer.SetFileName("scomposition.png")
        writer.Execute(return_dict["composition"])
        #sitk.imwrite(nda, "scomposition.png")
        plt.imshow(nda)
        plt.savefig('output/nda')
        #sitk.Show(img1, title="cthead1")
        #sitk.Show(return_dict["composition"], "ImageRegistration2 Composition")'''