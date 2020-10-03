import SimpleITK as sitk


def registration(fixedImage, movingImage, labels):
    """ Registers moving image to the fixed.

    Args:
        fixedImage: Fixed image of the registration, SimpleItk image.
        movingImage: Fixed image of the registration, SimpleItk image.
        labels: Labels of the moving image, used to create a mask for the
            registration. SimpleItk image.
    Returns:
        The result of the registration, SimpleItk image.

    """
    initial_transform = sitk.CenteredTransformInitializer(fixedImage,
                                                          movingImage,
                                                       sitk.AffineTransform(3),
                                                       sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Create the mask
    l = sitk.GetArrayFromImage(labels)
    l[l>1] = 1
    mask = sitk.GetImageFromArray(l)
    mask.CopyInformation(movingImage)

    registration_method.SetMetricMovingMask(mask)

    registration_method.SetMetricAsMeanSquares()

    registration_method.SetMetricSamplingStrategy(registration_method.NONE) # REGULAR RANDOM NONE
    registration_method.SetMetricSamplingPercentage(0.75) # 0.5

    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescent(learningRate=0.75, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    final_transform = sitk.AffineTransform(initial_transform)

    registration_method.SetInitialTransform(final_transform)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [2,1,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return registration_method.Execute(sitk.Cast(fixedImage, sitk.sitkFloat32), sitk.Cast(movingImage, sitk.sitkFloat32))

