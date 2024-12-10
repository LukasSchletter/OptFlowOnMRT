import SimpleITK as sitk
import os

# Paths to your images
fixed_image_path = 'fixed_image.nii'
moving_image_path = 'moving_image.nii'

# Path to Elastix executable (adjust this according to your installation)
elastix_path = 'path/to/elastix'  # Replace with actual path

# Create a SimpleITK image object for the fixed and moving images
fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

# Create a SimpleElastix object
elastix_image_filter = sitk.ElastixImageFilter()

# Set the path to the Elastix executable
elastix_image_filter.SetElastixExecutable(elastix_path)

# Define B-spline registration parameters via the parameter file
# In this case, we are using a B-spline transform with a grid resolution of 5
parameter_map = sitk.GetDefaultParameterMap('bspline')
parameter_map['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']
parameter_map['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
parameter_map['Transform'] = ['BSplineTransform']
parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
parameter_map['Optimizer'] = ['AdaptiveStochasticGradientDescent']
parameter_map['NumberOfResolutions'] = ['4']
parameter_map['MaximumNumberOfIterations'] = ['500']
parameter_map['GridSpacing'] = ['5']  # B-spline grid spacing (resolution 5)
parameter_map['FinalGridSpacingInPhysicalUnits'] = ['5']

# Add the parameter map to the filter
elastix_image_filter.SetParameterMap([parameter_map])

# Perform the registration
elastix_image_filter.SetFixedImage(fixed_image)
elastix_image_filter.SetMovingImage(moving_image)
elastix_image_filter.Execute()

# Get the result of the registration
registered_image = elastix_image_filter.GetResultImage()

# Save the registered image
sitk.WriteImage(registered_image, 'registered_image.nii')

# Visualize the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sitk.GetArrayViewFromImage(fixed_image)[fixed_image.GetSize()[2] // 2, :, :], cmap='gray')
plt.title('Fixed Image')

plt.subplot(1, 2, 2)
plt.imshow(sitk.GetArrayViewFromImage(registered_image)[registered_image.GetSize()[2] // 2, :, :], cmap='gray')
plt.title('Registered Moving Image')

plt.show()

