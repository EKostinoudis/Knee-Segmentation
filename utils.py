import SimpleITK as sitk
import numpy as np
from scipy.spatial import distance


def resample(movingImage, fixedImage, transform, interpolator=sitk.sitkNearestNeighbor):
    """
    Transoforms the moving image and resamples it
    based on the fixed.

    Args:
        movingImage: Fixed image of the tranformation, SimpleItk image.
        fixedImage: Fixed image of the tranformation, SimpleItk image.
        transform: The transform, SimpleItk transform.
        interpolator: The interpolator for the resampling.
    Returns:
        The tranformed image, SimpleItk image.

    """
    resampleImage = sitk.ResampleImageFilter()
    resampleImage.SetReferenceImage(fixedImage)

    resampleImage.SetInterpolator(interpolator)
    resampleImage.SetTransform(transform)

    return resampleImage.Execute(movingImage)

def resampleLabels(labels, fixedImage, t):
    """
    Transoforms the labels and resamples them
    based on the fixed image.
    """
    resampleLabels = sitk.ResampleImageFilter()
    resampleLabels.SetReferenceImage(fixedImage)

    resampleLabels.SetInterpolator(sitk.sitkNearestNeighbor)
    resampleLabels.SetTransform(t)

    return resampleLabels.Execute(labels)

def resampleImage(movingImage, fixedImage, t):
    """
    Transoforms the moving image and resamples it
    based on the fixed.
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixedImage)

    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(t)

    return resample.Execute(movingImage)

def shrinkImage(image, factor):
    """ Shrink image

    Args:
        image: Image to be shrinked, SimpleItk image.
        factor: Factor of the expansion.
        interpolator: Interpolator.
    Returns:
        Shrinked image.
    """
    if factor > 1:
        shrinkFilter = sitk.ShrinkImageFilter()
        shrinkFilter.SetShrinkFactor(factor)
        return shrinkFilter.Execute(image)
    else:
        return image

def readShrinkImage(path, factor):
    """ Read and shrink image

    Args:
        path: Path to image.
        factor: Factor of the expansion.
    Returns:
        Shrinked image.
    """
    return shrinkImage(sitk.ReadImage(path), factor)

def expandImage(image, factor, interpolator=sitk.sitkLinear):
    """ Expand image

    Args:
        image: Image to be expanded, SimpleItk image.
        factor: Factor of the expansion.
        interpolator: Interpolator.
    Returns:
        Expanded image.
    """
    if factor > 1:
        expandFilter = sitk.ExpandImageFilter()
        expandFilter.SetExpandFactor(factor)
        expandFilter.SetInterpolator(interpolator)
        return expandFilter.Execute(image)
    else:
        return image

def readExpandImage(path, factor, interpolator=sitk.sitkLinear):
    """ Read and shrink image

    Args:
        path: Path to image.
        factor: Factor of the expansion.
        interpolator: Interpolator.
    Returns:
        Expanded image.
    """
    return expandImage(sitk.ReadImage(path), factor, interpolator)


def registration(fixedImage, movingImage, labels):
    """ Registers moving image to the fixed.

    Args:
        fixedImage: Fixed image of the registration, SimpleItk image.
        movingImage: Fixed image of the registration, SimpleItk image.
        labels: Labels of the moving image, used to create a mask for the
            registration. SimpleItk image.
    Returns:
        The transform of the registration.

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


def registration2(fixedImage, movingImage, labels):
    """ Registers moving image to the fixed.

    Args:
        fixedImage: Fixed image of the registration, SimpleItk image.
        movingImage: Fixed image of the registration, SimpleItk image.
        labels: Labels of the moving image, used to create a mask for the
            registration. SimpleItk image.
    Returns:
        The transform of the registration.

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

    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR) # REGULAR RANDOM NONE
    registration_method.SetMetricSamplingPercentage(0.5)

    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescent(learningRate=0.75, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    final_transform = sitk.AffineTransform(initial_transform)

    registration_method.SetInitialTransform(final_transform)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [8,4,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [4,2,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return registration_method.Execute(sitk.Cast(fixedImage, sitk.sitkFloat32), sitk.Cast(movingImage, sitk.sitkFloat32))

def Dice(setA, setB, labelsNum):
    """Calculates the Dice coefficient for the two sets

    Args:
        setA: Set for the coefficient calculation, numpy array.
        setB: Set for the coefficient calculation, numpy array.
        labelsNum: Number of labels of the sets.
    Returns:
        The Dice coefficient.

    """
    dice = []
    for k in range(0,labelsNum+1):
        dice.append(np.sum(setB[setA==k]==k)*2.0 / (np.sum(setB[setB==k]==k) + np.sum(setA[setA==k]==k)))
    return dice


def createA(images, voxel, P, N):
    # Allocate memory
    A = np.zeros(shape=(P[0]*P[1]*P[2], len(images)*N[0]*N[1]*N[2]), dtype='uint16')

    index = 0
    for image in images:
        box = image[voxel[0] - P[0]//2 - N[0]//2: voxel[0] + P[0]//2 + N[0]//2 + 1,
                    voxel[1] - P[1]//2 - N[1]//2: voxel[1] + P[1]//2 + N[1]//2 + 1,
                    voxel[2] - P[2]//2 - N[2]//2: voxel[2] + P[2]//2 + N[2]//2 + 1,
                   ]

        for n0 in range(P[0]//2, P[0]//2 + N[0]):
            for n1 in range(P[1]//2, P[1]//2 + N[1]):
                for n2 in range(P[2]//2, P[2]//2 + N[2]):
                    A[:, index] = box[n0-(P[0]//2):n0+P[0]//2+1,
                                      n1-(P[1]//2):n1+P[1]//2+1,
                                      n2-(P[2]//2):n2+P[2]//2+1,
                                     ].reshape((-1,))
                    index += 1

    return A

def createB(image, voxel, P):
    return image[voxel[0]-(P[0]-1)//2:voxel[0]+(P[0]-1)//2+1,
                 voxel[1]-(P[1]-1)//2:voxel[1]+(P[1]-1)//2+1,
                 voxel[2]-(P[2]-1)//2:voxel[2]+(P[2]-1)//2+1,
                ].reshape((-1,))

def createL(labels, voxel, N):
    # Allocate memory
    L = np.zeros(shape=(len(labels)*N[0]*N[1]*N[2]), dtype='uint8')

    mul = N[0]*N[1]*N[2]
    for i, label in enumerate(labels):
        L[i*mul:(i+1)*mul] = label[-(N[0]//2)+voxel[0]:N[0]//2+voxel[0]+1,
                                   -(N[1]//2)+voxel[1]:N[1]//2+voxel[1]+1,
                                   -(N[2]//2)+voxel[2]:N[2]//2+voxel[2]+1,
                                  ].reshape((-1,))

    return L

def createLDict(labels, voxel, N, numOfLabels):
    # Allocate memory
    L = np.zeros(shape=(numOfLabels+1, len(labels)*N[0]*N[1]*N[2]), dtype='uint8')

    mul = N[0]*N[1]*N[2]
    for i, label in enumerate(labels):

        L[label[-(N[0]//2)+voxel[0]:N[0]//2+voxel[0]+1,
                -(N[1]//2)+voxel[1]:N[1]//2+voxel[1]+1,
                -(N[2]//2)+voxel[2]:N[2]//2+voxel[2]+1,
               ].reshape((-1,)),
            range(i*mul,(i+1)*mul)] = 1

    return L

def mse3D(image1, image2):
    return ((sitk.GetArrayFromImage(image1) - sitk.GetArrayFromImage(image2))**2).mean()

def mse3DLabels(image1, image2, labels):
    return (((sitk.GetArrayFromImage(image1) -
        sitk.GetArrayFromImage(image2))[sitk.GetArrayFromImage(labels) > 0])**2).mean()

def calculateCropShape(images, P, N, minx, maxx, miny, maxy, minz, maxz):
    shape = [[minx - (P[0]//2 + N[0]//2), maxx + (P[0]//2 + N[0]//2)],
             [miny - (P[1]//2 + N[1]//2), maxy + (P[1]//2 + N[1]//2)],
             [minz - (P[2]//2 + N[2]//2), maxz + (P[2]//2 + N[2]//2)],
            ]

    copyShape = [[],[],[]]

    # x range
    if shape[0][0] < 0:
        copyShape[0].append(0)
    else:
        copyShape[0].append(shape[0][0])

    if shape[0][1] > images[0].shape[0] - 1:
        copyShape[0].append(images[0].shape[0] - 1)
    else:
        copyShape[0].append(shape[0][1])

    # y range
    if shape[1][0] < 0:
        copyShape[1].append(0)
    else:
        copyShape[1].append(shape[1][0])

    if shape[1][1] > images[0].shape[1] - 1:
        copyShape[1].append(images[0].shape[1] - 1)
    else:
        copyShape[1].append(shape[1][1])

    #z range
    if shape[2][0] < 0:
        copyShape[2].append(0)
    else:
        copyShape[2].append(shape[2][0])

    if shape[2][1] > images[0].shape[2] - 1:
        copyShape[2].append(images[0].shape[2] - 1)
    else:
        copyShape[2].append(shape[2][1])

    offset = []
    offset.append(copyShape[0][0] - shape[0][0])
    offset.append(copyShape[1][0] - shape[1][0])
    offset.append(copyShape[2][0] - shape[2][0])

    length = []
    length.append(copyShape[0][1] - copyShape[0][0] + 1)
    length.append(copyShape[1][1] - copyShape[1][0] + 1)
    length.append(copyShape[2][1] - copyShape[2][0] + 1)

    return (shape, copyShape, offset, length)

def saveSegmentation(segmentation, originalImagePath, savePathName, copyShape, offset, length, verbose=True):
    originalImage = sitk.ReadImage(originalImagePath)
    originalImageShape = originalImage.GetSize()[::-1]
    finalSegmentation = np.zeros(originalImageShape, dtype='uint8')
    finalSegmentation[copyShape[0][0]:copyShape[0][1]+1,
                      copyShape[1][0]:copyShape[1][1]+1,
                      copyShape[2][0]:copyShape[2][1]+1,] = \
                            segmentation[offset[0]:offset[0]+length[0],
                                         offset[1]:offset[1]+length[1],
                                         offset[2]:offset[2]+length[2]]
    finalSegmentation = sitk.GetImageFromArray(finalSegmentation)
    finalSegmentation.CopyInformation(originalImage)
    sitk.WriteImage(finalSegmentation, savePathName, True)
    if verbose:
        print(f"Saved segmentation for image: {originalImagePath} " \
                + f"with the name: {savePathName}")


def translateToOriginal(segmentation, originalImagePath, copyShape, offset, length):
    originalImage = sitk.ReadImage(originalImagePath)
    originalImageShape = originalImage.GetSize()[::-1]
    finalSegmentation = np.zeros(originalImageShape, dtype='uint8')
    finalSegmentation[copyShape[0][0]:copyShape[0][1]+1,
                      copyShape[1][0]:copyShape[1][1]+1,
                      copyShape[2][0]:copyShape[2][1]+1,] = \
                            segmentation[offset[0]:offset[0]+length[0],
                                         offset[1]:offset[1]+length[1],
                                         offset[2]:offset[2]+length[2]]
    return finalSegmentation

def cropImage(image, shape, offset, length, copyShape, dtype):
    newImage = np.zeros((shape[0][1] - shape[0][0] + 1,
                         shape[1][1] - shape[1][0] + 1,
                         shape[2][1] - shape[2][0] + 1,
                        ), dtype=dtype, order="C")
    newImage[offset[0]:offset[0]+length[0],
             offset[1]:offset[1]+length[1],
             offset[2]:offset[2]+length[2],] = \
                     image[copyShape[0][0]:copyShape[0][1]+1,
                           copyShape[1][0]:copyShape[1][1]+1,
                           copyShape[2][0]:copyShape[2][1]+1,]
    return newImage


def registrationElastix(fixedImage, movingImage, outDir="."):
    """ Registers moving image to the fixed using Simple Elastix.

    Args:
        fixedImage: Fixed image of the registration, SimpleItk image.
        movingImage: Fixed image of the registration, SimpleItk image.
    Returns:
        The transform of the registration.

    """
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.SetOutputDirectory(outDir)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetParameter("WriteResultImage", "false")
    elastixImageFilter.Execute()

    transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0]

    transform = sitk.AffineTransform(3)
    params = [float(i) for i in transformParameterMap.asdict()['TransformParameters']]
    transform.SetTranslation(params[9:])
    transform.SetMatrix(params[0:9])
    center = [float(i) for i in transformParameterMap.asdict()['CenterOfRotationPoint']]
    transform.SetCenter(center)
    return transform


def registrationElastix2(fixedImage, movingImage, outDir="."):
    """ Registers moving image to the fixed using Simple Elastix.

    Args:
        fixedImage: Fixed image of the registration, SimpleItk image.
        movingImage: Fixed image of the registration, SimpleItk image.
    Returns:
        The transform of the registration.

    """
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.SetOutputDirectory(outDir)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetParameter("Metric", "AdvancedMeanSquares")
    elastixImageFilter.SetParameter("MaximumNumberOfIterations", "8192")
    elastixImageFilter.SetParameter("WriteResultImage", "false")
    elastixImageFilter.Execute()

    transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0]

    transform = sitk.AffineTransform(3)
    params = [float(i) for i in transformParameterMap.asdict()['TransformParameters']]
    transform.SetTranslation(params[9:])
    transform.SetMatrix(params[0:9])
    center = [float(i) for i in transformParameterMap.asdict()['CenterOfRotationPoint']]
    transform.SetCenter(center)
    return transform


def registrationElastixMask(fixedImage, movingImage, labels, outDir="."):
    """ Registers moving image to the fixed.

    Args:
        fixedImage: Fixed image of the registration, SimpleItk image.
        movingImage: Fixed image of the registration, SimpleItk image.
        labels: Labels of the moving image, used to create a mask for the
            registration. SimpleItk image.
    Returns:
        The transform of the registration.

    """
    # Create the mask
    l = sitk.GetArrayFromImage(labels)
    l[l>1] = 1
    mask = sitk.GetImageFromArray(l)
    mask.CopyInformation(movingImage)


    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.AddMovingMask(mask)
    elastixImageFilter.SetOutputDirectory(outDir)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetParameter("Metric", "AdvancedMeanSquares")
    elastixImageFilter.SetParameter("MaximumNumberOfIterations", "2048")
    elastixImageFilter.SetParameter("NumberOfSamplesForExactGradient", "8192")
    elastixImageFilter.SetParameter("NumberOfSpatialSamples", "8192")
    elastixImageFilter.SetParameter("SamplingPercentage", "0.8")
    elastixImageFilter.SetParameter("AutomaticTransformInitialization", "true")
    elastixImageFilter.SetParameter("ErodeMovingMask", "false")
    elastixImageFilter.SetParameter("CheckNumberOfSamples", "false")
    elastixImageFilter.SetParameter("WriteResultImage", "false")
    elastixImageFilter.Execute()

    transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0]

    transform = sitk.AffineTransform(3)
    params = [float(i) for i in transformParameterMap.asdict()['TransformParameters']]
    transform.SetTranslation(params[9:])
    transform.SetMatrix(params[0:9])
    center = [float(i) for i in transformParameterMap.asdict()['CenterOfRotationPoint']]
    transform.SetCenter(center)
    return transform
