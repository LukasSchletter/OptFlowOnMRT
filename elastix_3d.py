import SimpleITK as sitk
import os
import matplotlib.pyplot as plt

# Paths to your 3D images
fixed_image_path = 'fixed_image.nii'  # Path to fixed 3D image
moving_image_path = 'moving_image.nii'  # Path to moving 3D image

# Path to Elastix executable (adjust according to your installation)
elastix_path = 'path/to/elastix'  # Replace with the path to elastix executable

# Create SimpleITK image objects for fixed and moving images
fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

# Initialize the SimpleElastix filter
elastix_image_filter = sitk.ElastixImageFilter()

# Set the path to the Elastix executable
elastix_image_filter.SetElastixExecutable(elastix_path)

# Define B-spline registration parameters for 3D images with grid resolution 5
parameter_map = sitk.GetDefaultParameterMap('bspline')

# Set parameter map for 3D B-spline registration
parameter_map['FixedImagePyramid'] = ['FixedSmoothingImagePyramid']
parameter_map['MovingImagePyramid'] = ['MovingSmoothingImagePyramid']
parameter_map['Transform'] = ['BSplineTransform']
parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
parameter_map['Optimizer'] = ['AdaptiveStochasticGradientDescent']
parameter_map['NumberOfResolutions'] = ['4']  # Use multi-resolution strategy
parameter_map['MaximumNumberOfIterations'] = ['500']  # Max iterations per resolution

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
sitk.WriteImage(registered_image, 'registered_image_3d.nii')

# Visualize the result (select a slice from the 3D images to show)
# Display slices in the z-axis direction for visualization
fixed_array = sitk.GetArrayFromImage(fixed_image)
registered_array = sitk.GetArrayFromImage(registered_image)

# Visualize a middle slice from the z-axis
slice_idx = fixed_image.GetSize()[2] // 2

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(fixed_array[slice_idx, :, :], cmap='gray')
plt.title('Fixed Image Slice')

plt.subplot(1, 2, 2)
plt.imshow(registered_array[slice_idx, :, :], cmap='gray')
plt.title('Registered Moving Image Slice')

plt.show()