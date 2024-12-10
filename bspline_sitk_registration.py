import SimpleITK as sitk
import matplotlib.pyplot as plt

# Read the fixed and moving images
fixed_image = sitk.ReadImage('fixed_image.nii', sitk.sitkFloat32)
moving_image = sitk.ReadImage('moving_image.nii', sitk.sitkFloat32)

# Visualize the fixed and moving images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sitk.GetArrayViewFromImage(fixed_image)[fixed_image.GetSize()[2]//2, :, :], cmap='gray')
plt.title('Fixed Image')
plt.subplot(1, 2, 2)
plt.imshow(sitk.GetArrayViewFromImage(moving_image)[moving_image.GetSize()[2]//2, :, :], cmap='gray')
plt.title('Moving Image')
plt.show()

# Set up the registration method
registration_method = sitk.ImageRegistrationMethod()

# Use Mutual Information for the metric
registration_method.SetMetricAsMattes()
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

# Visualize the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sitk.GetArrayViewFromImage(fixed_image)[fixed_image.GetSize()[2]//2, :, :], cmap='gray')
plt.title('Fixed Image')

plt.subplot(1, 2, 2)
plt.imshow(sitk.GetArrayViewFromImage(resampled_image)[resampled_image.GetSize()[2]//2, :, :], cmap='gray')
plt.title('Registered Moving Image')

plt.show()