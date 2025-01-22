#groupwise registration

#4d = 110 slices á 25 pictures

# groupwise registration mit 25 slices
# 3d registration mit fixed and moving image
# 4d mit 110 slices als 3d statt 25! --> einfach reshape

#hm klappt nicht:
    # möglichkeit finden die 4d bilder wieder aufzuteilen und in Einzelbilder zu packen
    # Funktion schreiben, die die Einzelbilder wieder abspeichert
    # Kontrolle des Vorgangs --> Für Ende der Registrierung ist Visualisierung ohnehin nötig (siehe unten)
    # registrierung durchführen
    # registrierung auf calci durchführen
    # ergebnis der registrierung visualisieren

# Differenz bilden
    # Was soll das Ergebnis der Differenz sein?
    # Verschiedene Ansätze probieren
# Nutze RAFT auf den registrierten Bildern!


import SimpleITK as sitk
import sys
import os
import matplotlib.pyplot as plt
import itk
from configparser import ConfigParser
import numpy as np
#xximport itkJPEGImageIO
#include <itkJPEGImageIO.h>
import itk
from os import listdir
from packaging.version import parse
from importlib.metadata import version

if parse(version('itk')) < parse('5.3'):
    raise ValueError("ITK greater than version 5.3.0 is required for this notebook")

import numpy as np
#from itkwidgets import view
from matplotlib import pyplot as plt

#matplotlib inline
from ipywidgets import interact

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f" = {method.GetMetricValue():7.5f} "
        + f" : {method.GetOptimizerPosition()}"
    )

def writing(data):
    with open("output.txt", "w") as txt_file:
        for number in data:
           txt_file.write(" ".join(str(number)) ) # + "\n")

def split_and_write(image):
    path = 'reg_15res_15_sl_mov_14.npy'
    image_2 = np.load(path)
    print(image_2.shape)
    for k in range(0,15,1):
        print(k)
        print(image_2[0].shape)
        subimage = image_2[0]
        plt.imshow(subimage)
        plt.savefig('reg_15/images/' +str(k) +'.jpg')

    pass

def main_2d(args):

    
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
    directory_path = 'slice_18_npy'
    return_list = []
    #name_list.append('reg')
    for dir_content in listdir(directory_path):
        test = 'test vor variables'
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
            #print('fixed')
            #fixed = sitk.ReadImage(args[1], sitk.sitkFloat32)
            arr_loaded = np.load('slice_18_npy/951.npy')
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

    for s in range(0, len(arr_list), 1):
        string = 'slice_18_npy/' + name_list[s] + '.npy'
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

        elastixImageFilter.Execute()
        resultImage = elastixImageFilter.GetResultImage()

    
        print('only one : this will be returned')
        return_list.append(resultImage)
           
        """regcounter += 1
        if regcounter == 5:
            break"""
            
       # return {"name": name_list, "composition": return_list}
        #m = sitk.SquaredDifference(fixed,moving)
        #m_array = sitk.GetArrayFromImage(m)
       
        

        
        
    #return return_list 
    print('this will be returned')
    return {"name": name_list, "composition": return_list}
   
def load_images():

    pass

def get3dImage(list,slice):
    ThreeDimage = []
    vectorOfSubImages = sitk.VectorOfImage()
    if(slice < 0 or slice > 24):
        raise IndexError('slices only between 0 and 109')
    #print('You get slice ' + str(slice))
    #slice = 109 - slice
    slicenumber = slice*25
    
    
    for timeslice in range(0,110,1):
        #print(timeslice*25+slice)
        #print(list[timeslice*25+slice][0])
        ThreeDimage.append(list[timeslice*25 + slice][1])
        vectorOfSubImages.push_back(list[timeslice*25 + slice][1])
    counter = 0
    """for k in range(0,25,1):
        #print(list[slicenumber + k][0])
        ThreeDimage.append(list[slicenumber + k][1])
        vectorOfSubImages.push_back(list[slicenumber + k][1])"""

    image = sitk.JoinSeries(vectorOfSubImages)
   
    return(image)

def get3dImage_110(list,slice):
    ThreeDimage = []
    vectorOfSubImages = sitk.VectorOfImage()
    if(slice < 0 or slice > 24):
        raise IndexError('slices only between 0 and 25')
    #print('You get slice ' + str(slice))
    #slice = 109 - slice
    slicenumber = slice*25

    print('slice = ' + str(slice))
    for timeslice in range(0,110,1):
        print('timeslice = ' + str(timeslice))
        print(timeslice*25+slice)
        print(list[timeslice*25+slice][0])
        ThreeDimage.append(list[timeslice*25 + slice][1])
        vectorOfSubImages.push_back(list[timeslice*25 + slice][1])
    counter = 0
    """for k in range(0,25,1):
        #print(list[slicenumber + k][0])
        ThreeDimage.append(list[slicenumber + k][1])
        vectorOfSubImages.push_back(list[slicenumber + k][1])"""

    image = sitk.JoinSeries(vectorOfSubImages)

    return(image)

def get3dImage_25(list,slice):
    ThreeDimage = []
    vectorOfSubImages = sitk.VectorOfImage()
    if(slice < 0 or slice > 109):
        raise IndexError('slices only between 0 and 109')
    print('You get slice ' + str(slice))
    slice = 109 - slice
    slicenumber = slice*25
    
   
    counter = 0
    for k in range(0,25,1):
        #print(list[slicenumber + k][0])
        ThreeDimage.append(list[slicenumber + k][1])
        vectorOfSubImages.push_back(list[slicenumber + k][1])

    image = sitk.JoinSeries(vectorOfSubImages)
   
    return(image)

def get3d_48_62(list, timeslice):
    # get slices between 48 and 62
    if(timeslice < 0 or timeslice > 24):
        raise IndexError('slices only between 0 and 25')
    vectorOfSubImages = sitk.VectorOfImage()
    for k in range(48,63,1):
        #print('slice: '  + str(k))
        slice = 109 - k
        #print('transformed slice: ' + str(slice))
        slicenumber = slice*25
        vectorOfSubImages.push_back(list[slicenumber + timeslice][1])
        #print(list[slicenumber + timeslice][0])
        
    image = sitk.JoinSeries(vectorOfSubImages)
   
    return(image)  

def main_group():

    split_and_write(2)
    return 1

    arr_list = []
    name_list = []
    name_list_image = []
    np.set_printoptions(threshold=sys.maxsize)
    all_image_list = []
    #print(all_image_list)
    all_image_name_list = []
    image_npy_list = []
    directory_path = 'resNpSlices/4dflow_KI_20241104-233450'
    return_list = []
    MetaEntityList = []
    entity_list = []
    #name_list.append('reg')
    for dir_content in listdir(directory_path):
        #print('new folder')
        string = dir_content
        intstring = int(dir_content)
        #print(string)
        #print(intstring)
        path_2 = directory_path + '/' + dir_content
        #print(path_2)

        for image in listdir(path_2):
            #print(image)
            imagestring = image.removesuffix('.npy')
            intimagestring = int(imagestring)
            #print(intimagestring)
            movarr_loaded = np.load(directory_path + '/' + dir_content +'/' + image)
            movarr_loaded = np.float32(movarr_loaded)
            moving = sitk.GetImageFromArray(movarr_loaded)
            #print(type(moving))
            movarr = sitk.GetArrayFromImage(moving)
            #all_image_list[intimagestring] = moving
            entity = [int(imagestring), moving]
            entity_list.append(entity)

        """string = dir_content
        string_2 = string.removesuffix('_npy')
        string_3 = string_2.removeprefix('slice_')
        name_list.append(string_3)
        path_2 = 'npy_images_3d/' + dir_content
        content_image_list = []
        arr_list_image = []
        entity_list = []
        for image in listdir(path_2):
            #print(image)
            imagestring = image.removesuffix('.npy')
            #print(directory_path + '/' + dir_content +'/' + image)
            movarr_loaded = np.load(directory_path + '/' + dir_content +'/' + image)
            movarr_loaded = np.float32(movarr_loaded)
            moving = sitk.GetImageFromArray(movarr_loaded)
            movarr = sitk.GetArrayFromImage(moving)
            all_image_list.append(moving)
            arr_list_image.append(movarr_loaded)
            content_image_list.append(imagestring)
            all_image_name_list.append(imagestring)
            entity = [int(imagestring), moving]
            entity_list.append(entity)

        

        name_list_image.append(content_image_list)
        arr_list.append(arr_list_image)
        entity_list.sort()
        plot_list(entity_list)
        MetaEntity = [int(string_3), entity_list]
        print(dir_content)
        MetaEntityList.append(MetaEntity)
        """
    entity_list.sort() # von 1 bis 2750
    """first = True
    last = True
    counter = 0
    print(len(entity_list))
    for ent in entity_list:
        if(first):
            print(ent[0])
            first = False
        counter += 1
        #print(counter)
        if(counter == len(entity_list)):
            print(ent[0])
            last = False    """

    """ print('going 3d')
    # test 3d/4d pop method
    image_3_d_25 = get3dImage_25(entity_list,56)
    image_3_d_120 = get3dImage_110(entity_list,7)
    #image_3_d_test = get3dImage(entity_list,24)
    print('printing 3d')
    #print(image_3_d_test)"""
    # print('printed 3d')
    """dim = image_3_d_test.GetDimension()
    print(dim)"""
    """dim = image_3_d_120.GetDimension()
    print(dim)
    dim = image_3_d_25.GetDimension()
    print(dim)"""
    """image_2_d = sitk.Extract(image_3_d_test)
    dim = image_2_d.GetDimension()
    print(dim)"""
    
    
   

    # make the 25 3d images with JoinSeries
    # pushback in vector of images for 4d image
    print('going 4d')

    #image_3_d = get3dImage(entity_list,18)
    
    """for slices in range(0,25,1):
        print(slices)
        image_3_d = get3dImage(entity_list,slices)
        vectorOfImages.push_back(image_3_d)"""

    image_3_d_48 = get3d_48_62(entity_list,13)

    print(type(image_3_d_48))
    print(image_3_d_48.GetSize())

    fixed = get3d_48_62(entity_list,13)
    moving_number = 14
    moving = get3d_48_62(entity_list,moving_number)
    
    savepath = 'reg_15'
    
    #return 1
    
    # Register
    #### Simple Elastix
        
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.Execute()
    resultImage = elastixImageFilter.GetResultImage()

    resultImage = sitk.GetArrayFromImage(resultImage)

    np.save(savepath + '/res_15_sl_mov_'+ str(moving_number) + '.npy',resultImage)
    split_and_write(resultImage)
    return 1
    
  

if __name__ == "__main__":
    print('main_2:')
    test = main_group()
    print('end of program')
    print(test)
    """return_dict = main(sys.argv)
    comp_list = return_dict["composition"]
    name_list = return_dict["name"]
    counter = 0
    print('Länge return list')
    #print(len(comp_list))
    for compo in range(0, len(comp_list),1):
        #print(counter)
        counter += 1
        nda = sitk.GetArrayFromImage(comp_list[compo])
        plt.imshow(nda)
        plt.savefig('output/'+name_list[compo])"""
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