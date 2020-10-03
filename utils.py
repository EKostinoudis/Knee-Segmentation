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

def resampleSegmentation(labels, fixedImage, t):
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
    for k in range(1,labelsNum+1):
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

