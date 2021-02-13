#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from time import time
from sklearn.linear_model import Lasso
# from libc.stdlib cimport malloc, free
from libc.stdint cimport uint16_t, uint8_t
from libc.math cimport sqrt, fabs, exp
from libc.float cimport FLT_MIN
from libc.stdio cimport printf, fflush, stdout
from libc.time cimport clock, CLOCKS_PER_SEC, clock_t
from scipy.linalg.cython_blas cimport sdot, scopy, saxpy, sasum, snrm2
import spams


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline float _max(float a, float b) nogil:
    return a if a >= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline float _abs(float a) nogil:
    return a if a >= 0.0 else -a

'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _gapSafeCD(int yLen, int wLen, float[::1] R, float[::1] w, float[::1] XColsSqSum, float[::1] y, float[::1,:] X, double lamda, uint8_t[::1] activeSet, long long fce=10, double tolerance=1e-4, long long maxIterations=10000) nogil:
    """
    Calculates 1/2 ||X*w - y||^2_2 + lamda* ||w||_1
    using iterative coordinate descent with gap safe rules.
    """

    # Increment, for blas use
    cdef int inc = 1

    cdef int iteration, elem, j, i
    cdef float temp, tempMax, dualScale
    cdef float prevW
    cdef float dualityGap
    cdef float primal, dual
    cdef float normR, normW, radius

    # Calculate the residual
    # R = y - Xw
    for i in range(yLen):
        R[i] = y[i] - sdot(&wLen, &X[0,i], &inc, &w[0], &inc) 
    # w = 0 so R = y
    # scopy(&yLen, &y[0], &inc, &R[0], &inc)

    # Initialize activeSet
    for elem in range(wLen):
        activeSet[elem] = 1

    for iteration in range(maxIterations):
        if iteration % fce == 0 and iteration != 0:
            # max(X.T * R)
            dualScale = FLT_MIN
            for elem in range(wLen):
                if activeSet[elem] != 1:
                    continue

                # X[:, elem].T * R
                temp = sdot(&yLen, &R[0], &inc, &X[0, elem], &inc)

                if temp > dualScale:
                    dualScale = temp

            # max(X.T * R, lamda)
            if lamda > dualScale:
                dualScale = lamda

            # ||R||_2, second norm of R
            normR = snrm2(&yLen, &R[0], &inc)

            # ||w||_1, w[elem] >=0 for every elem
            normW = 0
            for elem in range(wLen):
                normW += w[elem]

            # primal = 1/2 ||X*w - y||_2^2 + lamda* ||w||_1
            # 1/2    ||R||_2^2    + lamda* ||w||_1
            primal = 0.5 * (normR ** 2) + lamda * normW 

            # dual = lamda * theta*y - 1/2 * lamda^2 * ||theta||_2^2
            # dual = (lamda / dualScale) * R^T*y - 1/2 * lamda^2 / dualScale^2 * ||-R||_2^2
            # theta = -R / max(lamda, ||X^T*R||_inf)
            temp = lamda / dualScale
            dual = temp * sdot(&yLen, &R[0], &inc, &y[0], &inc) - 0.5 * ((temp * normR) ** 2)

            dualityGap = primal - dual

            if dualityGap <= tolerance:
                break

            # Gap safe radius
            # radius = sqrt(2*dualityGap) / lamda
            radius = sqrt(dualityGap) / lamda
            
            # Update active set
            for elem in range(wLen):
                if activeSet[elem] != 1:
                    continue
                """
                # |X[:, elem].T * R| / dualScale
                temp = fabs(sdot(&yLen, &R[0], &inc, &X[0, elem], &inc)) / dualScale

                if temp >= 1:
                    continue

                # max(X[:,elem]
                tempMax = fabs(X[0, elem])
                for j in range(1, yLen):
                    if fabs(X[j, elem]) > tempMax:
                        tempMax = fabs(X[j, elem])

                temp += radius * tempMax
                """

                #############
                temp = radius * sqrt(XColsSqSum[elem])
                if temp >= 1:
                    continue
                temp += fabs(sdot(&yLen, &R[0], &inc, &X[0, elem], &inc)) / dualScale
                #############

                if temp < 1:
                    activeSet[elem] = 0
                    prevW = w[elem]
                    w[elem] = 0
            
                    # Calculate the new residual
                    # R -= (w[elem] - prevW) * X[:, elem]
                    temp = -(w[elem] - prevW)
                    saxpy(&yLen, &temp, &X[0, elem], &inc, &R[0], &inc)

        # Coordiante descent
        for elem in range(wLen):
            # X[:, elem] == 0
            if activeSet[elem] != 1 or XColsSqSum[elem] == 0.0:
                continue

            # X[:, elem].T * R
            # for use in soft-thresholding operator
            temp = sdot(&yLen, &R[0], &inc, &X[0, elem], &inc)

            # w[elem] + (temp / |X_j|_2^2)
            temp /= XColsSqSum[elem]
            temp += w[elem]

            # Hold the previous w[elem] value
            prevW = w[elem]

            # Calculate soft-thresholding
            # we want w >= 0
            if temp >= 0:
                w[elem] = _max(temp - lamda / XColsSqSum[elem], 0.0)
            else:
                w[elem] = 0.0

            # Calculate the new residual
            # R -= (w[elem] - prevW) * X[:, elem]
            if w[elem] != 0.0:
                temp = -(w[elem] - prevW)
                saxpy(&yLen, &temp, &X[0, elem], &inc, &R[0], &inc)
'''


'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void _elasticNet(int yLen, int wLen, float[::1] R, float[::1] w, float[::1] XColsSqSum, float[::1] y, float[::1,:] X, double alpha, double beta, double tolerance=1e-4, long long maxIterations=10000) nogil:
    # Increment, for blas use
    cdef int inc = 1

    # Calculate the residual
    # R = y - Xw
    # for i in range(yLen):
    #     R[i] = y[i] - sdot(&wLen, &X[0,i], &inc, &w[0], &inc) 
    # w = 0 so R = y
    scopy(&yLen, &y[0], &inc, &R[0], &inc)

    cdef int iteration, elem
    cdef float temp
    cdef float prevW
    cdef float wMax, dwMax, tempMax
    cdef float dualityGap
    cdef float initialTolerance = tolerance

    tolerance *= sdot(&yLen, &y[0], &inc, &y[0], &inc)

    for iteration in range(maxIterations):
        wMax = 0.0
        dwMax = 0.0
        for elem in range(wLen):
            # if all X[:, elem] == 0
            if XColsSqSum[elem] == 0.0:
                continue

            # Remove the contribution of w[elem] to the residual
            # R = y - X[:, not elem] * w[not elem]
            # R += w[elem] * X[:, elem]
            if w[elem] != 0.0:
                saxpy(&yLen, &w[elem], &X[0, elem], &inc, &R[0], &inc)

            # X[:, elem].T * R ; R (without the X[:, elem] * w[elem])
            # for use in soft-thresholding operator
            temp = sdot(&yLen, &R[0], &inc, &X[0, elem], &inc)

            # Hold the previous w[elem] value
            prevW = w[elem]

            # Calculate soft-thresholding
            # we want w >= 0
            if temp >= 0:
                w[elem] = _max(temp - alpha, 0.0) / (XColsSqSum[elem] + beta)
            else:
                w[elem] = 0.0

            # Calculate the new residual
            # R -= w[elem] * X[:, elem]
            if w[elem] != 0.0:
                temp = -w[elem]
                saxpy(&yLen, &temp, &X[0, elem], &inc, &R[0], &inc)

            # Maximum w
            if w[elem] > wMax:
                wMax = w[elem]

            # Maximum absolute w change
            temp = _abs(w[elem] - prevW)
            if temp > dwMax:
                dwMax = temp

        if wMax == 0.0 or (dwMax / wMax) < initialTolerance or iteration == maxIterations:
            # max(X.T * R - beta * w)
            for i in range(wLen):
                temp = sdot(&yLen, &R[0], &inc, &X[0, i], &inc) - beta * w[i]
                if i != 0:
                    if temp > tempMax:
                        tempMax = temp
                else:
                    tempMax = temp

            if(tempMax > alpha):
                if beta != 0.0:
                    # 1/2 * (R^T*R) * (1 + (alpha/tempMax)^2) + 
                    # a * sum(w) - (alpha/tempMax) R^T*y +
                    # 1/2 * beta * w^T^w * (1 + (alpha/tempMax)^2)
                    temp = (alpha / tempMax) 
                    temp *= temp
                    temp += 1
                    dualityGap = 0.5 * sdot(&yLen, &R[0], &inc, &R[0], &inc) * temp \
                            + alpha * sasum(&wLen, &w[0], &inc) \
                            - (alpha/tempMax) * sdot(&yLen, &R[0], &inc, &y[0], &inc) \
                            + 0.5 * beta * temp * sdot(&wLen, &w[0], &inc, &w[0], &inc)
                else:
                    # 1/2 * (R^T*R) * (1 + (alpha/tempMax)^2) + 
                    # a * sum(w) - (alpha/tempMax) R^T*y
                    temp = (alpha / tempMax) 
                    temp *= temp
                    temp += 1
                    dualityGap = 0.5 * sdot(&yLen, &R[0], &inc, &R[0], &inc) * temp \
                            + alpha * sasum(&wLen, &w[0], &inc) \
                            - (alpha/tempMax) * sdot(&yLen, &R[0], &inc, &y[0], &inc)
            else:
                if beta != 0.0:
                    # (R^T*R) + a * sum(w) - R^T*y + beta * w^T^w
                    dualityGap = sdot(&yLen, &R[0], &inc, &R[0], &inc) \
                            + alpha * sasum(&wLen, &w[0], &inc) \
                            - sdot(&yLen, &R[0], &inc, &y[0], &inc) \
                            + beta * sdot(&wLen, &w[0], &inc, &w[0], &inc)
                else:
                    # (R^T*R) + a * sum(w) - R^T*y
                    dualityGap = sdot(&yLen, &R[0], &inc, &R[0], &inc) \
                            + alpha * sasum(&wLen, &w[0], &inc) \
                            - sdot(&yLen, &R[0], &inc, &y[0], &inc)

            if dualityGap < tolerance:
                break
'''


            

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def createA(np.ndarray[np.uint16_t, ndim=4] images,int[::1] voxel,int[::1] P,int[::1] N):
    cdef int imagesLen = images.shape[0]

    # Allocate memory
    cdef np.ndarray[np.uint16_t, ndim=2] A = np.zeros(shape=(P[0]*P[1]*P[2], imagesLen*N[0]*N[1]*N[2]), dtype='uint16')
    
    cdef int index = 0
    cdef int n0, n1, n2, index2
    cdef int i, j, k
    cdef int[3] v
    # cdef np.ndarray[np.uint16_t, ndim=3] image
    for image in range(imagesLen):
        for n0 in range(-(N[0]-1)//2, (N[0]-1)//2+1):
            for n1 in range(-(N[1]-1)//2, (N[1]-1)//2+1):
                for n2 in range(-(N[2]-1)//2, (N[2]-1)//2+1):
                    v = [voxel[0] + n0, voxel[1] + n1, voxel[2] + n2]
                    index2 = 0
                    for i in range(v[0]-(P[0]-1)//2,v[0]+(P[0]-1)//2+1):
                        for j in range(v[1]-(P[1]-1)//2,v[1]+(P[1]-1)//2+1):
                            for k in range(v[2]-(P[2]-1)//2,v[2]+(P[2]-1)//2+1):
                                A[index2, index] = images[image,i,j,k]
                                index2 +=1
                    index += 1                    
    return A


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void _createA(uint16_t[::1,:] A, uint16_t[:,:,:,::1] images, int* voxel,int[::1] P,int[::1] N) nogil:
    cdef int imagesLen = images.shape[0]

    cdef int index = 0
    cdef int n0, n1, n2, index2
    cdef int i, j, k
    cdef int[3] v
    # cdef np.ndarray[np.uint16_t, ndim=3] image
    for image in range(imagesLen):
        for n0 in range(-(N[0]-1)//2, (N[0]-1)//2+1):
            v[0] = voxel[0] + n0
            for n1 in range(-(N[1]-1)//2, (N[1]-1)//2+1):
                v[1] = voxel[1] + n1
                for n2 in range(-(N[2]-1)//2, (N[2]-1)//2+1):
                    # v = [voxel[0] + n0, voxel[1] + n1, voxel[2] + n2]
                    v[2] = voxel[2] + n2

                    index2 = 0
                    for i in range(v[0]-(P[0]-1)//2,v[0]+(P[0]-1)//2+1):
                        for j in range(v[1]-(P[1]-1)//2,v[1]+(P[1]-1)//2+1):
                            for k in range(v[2]-(P[2]-1)//2,v[2]+(P[2]-1)//2+1):
                                A[index2, index] = images[image,i,j,k]
                                index2 +=1
                    index += 1                    


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void _createAFloat(float[::1,:] A, uint16_t[:,:,:,::1] images, int* voxel,int[::1] P,int[::1] N) nogil:
    cdef int imagesLen = images.shape[0]

    cdef int index = 0
    cdef int n0, n1, n2, index2
    cdef int i, j, k
    cdef int[3] v
    # cdef np.ndarray[np.uint16_t, ndim=3] image
    for image in range(imagesLen):
        for n0 in range(-(N[0]-1)//2, (N[0]-1)//2+1):
            v[0] = voxel[0] + n0
            for n1 in range(-(N[1]-1)//2, (N[1]-1)//2+1):
                v[1] = voxel[1] + n1
                for n2 in range(-(N[2]-1)//2, (N[2]-1)//2+1):
                    # v = [voxel[0] + n0, voxel[1] + n1, voxel[2] + n2]
                    v[2] = voxel[2] + n2

                    index2 = 0
                    for i in range(v[0]-(P[0]-1)//2,v[0]+(P[0]-1)//2+1):
                        for j in range(v[1]-(P[1]-1)//2,v[1]+(P[1]-1)//2+1):
                            for k in range(v[2]-(P[2]-1)//2,v[2]+(P[2]-1)//2+1):
                                A[index2, index] = <float> images[image,i,j,k]
                                index2 +=1
                    index += 1                    


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def createB(np.ndarray[np.uint16_t, ndim=3] image, int[::1] voxel, int[::1] P):
    cdef np.ndarray[np.uint16_t, ndim=1] B = np.zeros(shape=(P[0]*P[1]*P[2]), dtype='uint16')
    cdef int i, j, k
    cdef int index = 0
    for i in range(voxel[0]-(P[0]-1)//2,voxel[0]+(P[0]-1)//2+1):
        for j in range(voxel[1]-(P[1]-1)//2,voxel[1]+(P[1]-1)//2+1):
            for k in range(voxel[2]-(P[2]-1)//2,voxel[2]+(P[2]-1)//2+1):
                B[index] = image[i,j,k]
                index += 1
    return B
        

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void _createB(uint16_t[::1] B, uint16_t[:,:,::1] image,int* voxel, int[::1] P) nogil:
    cdef int i, j, k
    cdef int index = 0
    for i in range(voxel[0]-(P[0]-1)//2,voxel[0]+(P[0]-1)//2+1):
        for j in range(voxel[1]-(P[1]-1)//2,voxel[1]+(P[1]-1)//2+1):
            for k in range(voxel[2]-(P[2]-1)//2,voxel[2]+(P[2]-1)//2+1):
                B[index] = image[i,j,k]
                index += 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void _createBFloat(float[::1] B, uint16_t[:,:,::1] image,int* voxel, int[::1] P) nogil:
    cdef int i, j, k
    cdef int index = 0
    for i in range(voxel[0]-(P[0]-1)//2,voxel[0]+(P[0]-1)//2+1):
        for j in range(voxel[1]-(P[1]-1)//2,voxel[1]+(P[1]-1)//2+1):
            for k in range(voxel[2]-(P[2]-1)//2,voxel[2]+(P[2]-1)//2+1):
                B[index] = <float> image[i,j,k]
                index += 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void _createBFloat2(float[::1,:] B, uint16_t[:,:,::1] image,int* voxel, int[::1] P) nogil:
    cdef int i, j, k
    cdef int index = 0
    for i in range(voxel[0]-(P[0]-1)//2,voxel[0]+(P[0]-1)//2+1):
        for j in range(voxel[1]-(P[1]-1)//2,voxel[1]+(P[1]-1)//2+1):
            for k in range(voxel[2]-(P[2]-1)//2,voxel[2]+(P[2]-1)//2+1):
                B[index,0] = <float> image[i,j,k]
                index += 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def createL(np.ndarray[np.uint8_t, ndim=4] labels,int[::1] voxel,int[::1] N):
    cdef int labelsLen = labels.shape[0]
    # Allocate memory
    cdef np.ndarray[np.uint8_t, ndim=1] L = np.zeros(shape=(labelsLen*N[0]*N[1]*N[2]), dtype='uint8')

    cdef int index = 0
    cdef int i, j, k, label
    for label in range(labelsLen):
        for i in range(-(N[0]//2)+voxel[0],N[0]//2+voxel[0]+1):
            for j in range(-(N[1]//2)+voxel[1],N[1]//2+voxel[1]+1):
                for k in range(-(N[2]//2)+voxel[2],N[2]//2+voxel[2]+1):
                    L[index] = labels[label, i, j, k]
                    index += 1

    return L


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void _createL(uint8_t[::1] L, uint8_t[:,:,:,::1] labels, int* voxel, int[::1] N) nogil:
    cdef int labelsLen = labels.shape[0]
    # Allocate memory
    # cdef np.ndarray[np.uint8_t, ndim=1] L = np.zeros(shape=(labelsLen*N[0]*N[1]*N[2]), dtype='uint8')

    cdef int index = 0
    cdef int i, j, k, label
    for label in range(labelsLen):
        for i in range(-(N[0]//2)+voxel[0],N[0]//2+voxel[0]+1):
            for j in range(-(N[1]//2)+voxel[1],N[1]//2+voxel[1]+1):
                for k in range(-(N[2]//2)+voxel[2],N[2]//2+voxel[2]+1):
                    L[index] = labels[label, i, j, k]
                    index += 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def labelFusion(np.ndarray[np.float64_t, ndim=1] w, np.ndarray[np.uint8_t, ndim=1] L):
    cdef int wLen = w.shape[0]
    cdef int i
    cdef double s = 0.
    cdef double sumW = 0
    for i in range(wLen):
        s += w[i]
        if L[i]:
            sumW += w[i]

    cdef double ret = 0.
    
    if (s != 0.):
        ret = sumW / s

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def segmentation(double L):
    return 1 if L >= 0.5 else 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline uint8_t _segmentation(double[::1] w, uint8_t[::1] L,long long numOfLabels) nogil:
    cdef int wLen = w.shape[0]
    cdef int i
    cdef double s = 0.
    # cdef double *sumLabels = <double *> malloc(numOfLabels * sizeof(double))
    cdef double[10] sumLabels

    for i in range(numOfLabels):
        sumLabels[i] = 0.

    for i in range(wLen):
        s += w[i]
        # if L[i] and w[i] != 0.:
        if L[i] > 0:
            sumLabels[L[i]-1] += w[i]

    if (s != 0.):
        for i in range(numOfLabels):
            if sumLabels[i] / s >= 0.5:
                return i + 1 

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline uint8_t _segmentationFloat(float[::1] w, uint8_t[::1] L,long long numOfLabels, int wLen) nogil:
    cdef int i
    cdef float s = 0.
    # cdef double *sumAll = <double *> malloc(numOfLabels * sizeof(double))
    cdef float[10] sumAll

    for i in range(numOfLabels):
        sumAll[i] = 0.

    for i in range(wLen):
        s += w[i]
        if L[i] and w[i] != 0.:
            sumAll[L[i]-1] += w[i]

    if (s != 0.):
        for i in range(numOfLabels):
            if sumAll[i] / s >= 0.5:
                return i + 1 

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline uint8_t _segmentationFloatMax(float[::1] w, uint8_t[::1] L,long long numOfLabels, int wLen) nogil:
    cdef int i
    cdef float s = 0.
    # cdef double *sumAll = <double *> malloc(numOfLabels * sizeof(double))
    cdef float[10] sumAll
    cdef float maxValue
    cdef int maxIndex

    for i in range(numOfLabels):
        sumAll[i] = 0.

    for i in range(wLen):
        s += w[i]
        sumAll[L[i]] += w[i]

    if (s != 0.):
        maxValue = sumAll[0] / s
        maxIndex = 0
        for i in range(1, numOfLabels + 1):
            if sumAll[i] / s > maxValue:
                maxValue = sumAll[i] / s
                maxIndex = i

        return maxIndex

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.uint8_t, ndim=3] _applySPBM(np.ndarray[np.uint16_t, ndim=3] segImage, 
                       np.ndarray[np.uint16_t, ndim=4] images, 
                       np.ndarray[np.uint8_t, ndim=4] labels,
                       long long numOfLabels,
                       int[::1] P,
                       int[::1] N,
                       double lassoTol=0.01,
                       double lassoMaxIter=1e4, 
                       bint verboseX=True, 
                       bint verboseY=True,
                       long long xmin=-1,
                       long long xmax=-1,
                       long long ymin=-1,
                       long long ymax=-1,
                       long long zmin=-1,
                       long long zmax=-1): 
    size = segImage.shape
    sizeImages = images[0].shape
    sizeLabels = labels[0].shape

    assert (size[0] == sizeImages[0] and size[1] == sizeImages[1] and size[2] == sizeImages[2]), "Segmentation images and images has different size."
    assert (size[0] == sizeLabels[0] and size[1] == sizeLabels[1] and size[2] == sizeLabels[2]), "Segmentation images and labels has different size."
    assert (images.shape[0] == labels.shape[0]), "Images and labels number mismatch."
    assert (numOfLabels >= 1), "Number of labels lower than one."
    

    # Allocate memory for the segmentation
    cdef np.ndarray[np.uint8_t, ndim=3] newSegmentation = np.zeros(shape=(size[0], size[1], size[2]), dtype='uint8')
    
    # x range
    if not (xmin >= N[0]//2 + P[0]//2 and xmin < size[0] - N[0]//2 - P[0]//2 - 1):
        xmin = N[0]//2 + P[0]//2
    if not (xmax >= N[0]//2 + P[0]//2 and xmax < size[0] - N[0]//2 - P[0]//2 - 1 and xmax > xmin):
        xmax = size[0] - N[0]//2 - P[0]//2 - 1
        
    # y range
    if not (ymin >= N[1]//2 + P[1]//2 and ymin < size[1] - N[1]//2 - P[1]//2 - 1):
        ymin = N[1]//2 + P[1]//2
    if not (ymax >= N[1]//2 + P[1]//2 and ymax < size[1] - N[1]//2 - P[1]//2 - 1 and ymax > ymin):
        ymax = size[1] - N[1]//2 - P[1]//2 - 1
        
    # z range
    if not (zmin >= N[2]//2 + P[2]//2 and zmin < size[2] - N[2]//2 - P[2]//2 - 1):
        zmin = N[2]//2 + P[2]//2
    if not (zmax >= N[2]//2 + P[2]//2 and zmax < size[2] - N[2]//2 - P[2]//2 - 1 and zmax > zmin):
        zmax = size[2] - N[2]//2 - P[2]//2 - 1
    
    cdef int[3] v 
    cdef int x, y, z
    cdef int labelsLen = labels.shape[0]
    cdef int LLen = labelsLen*N[0]*N[1]*N[2]
    cdef int imagesLen = images.shape[0]
    cdef int BLen = P[0]*P[1]*P[2]

    cdef np.ndarray[np.uint8_t, ndim=1] L = np.zeros(shape=(LLen), dtype='uint8')
    cdef uint8_t[::1] _L = L

    cdef np.ndarray[np.uint16_t, ndim=2] A = np.zeros(shape=(BLen, LLen), dtype='uint16', order='F')
    cdef uint16_t[::1,:] _A = A


    cdef np.ndarray[np.uint16_t, ndim=1] B = np.zeros(shape=(BLen), dtype='uint16')
    cdef uint16_t[::1] _B = B

    cdef uint8_t[:, :, :, ::1] _labels = labels
    cdef uint16_t[:, :, :, ::1] _images = images
    cdef uint16_t[:, :, ::1] _segImage = segImage


    cdef uint8_t allSame
    
    cdef int alphaLen = A.shape[1]

    cdef np.ndarray[np.float64_t, ndim=1] alpha = np.zeros(shape=A.shape[1], dtype='float64')
    cdef double[::1] _alpha = alpha

    cdef long long maxArg, sumArg


    for x in range(xmin, xmax):
        v[0] = x

        if verboseX:
            tstartX = time()
        for y in range(ymin, ymax):
            v[1] = y

            if verboseY:
                print('\tX:', x, '\tY:', y , end=" ")
                tstartY = time()

            for z in range(zmin, zmax):
                v[2] = z

                _createL(_L, _labels, v, N)

                allSame = 1
                for ii in range(1,LLen):
                    if _L[ii] != _L[0]:
                        allSame = 0
                        break

                if allSame == 1:
                    newSegmentation[x, y, z] = L[0]
                    continue

                _createA(_A, _images, v, P, N)
                _createB(_B, _segImage, v, P)

                allSame = 1
                for ii in range(BLen):
                    if _B[ii] != 0:
                        allSame = 0
                        break

                # max(A.T * B)
                if allSame == 0:
                    maxArg = 0
                    for ii in range(LLen):
                        sumArg = 0
                        for jj in range(BLen):
                            sumArg += <long long> A[jj,ii] * B[jj] 

                        if sumArg > maxArg:
                            maxArg = sumArg

                    _alpha = Lasso(alpha=maxArg * lassoTol, positive=True, max_iter=lassoMaxIter).fit(A,B).coef_
                    # lasso = Lasso(alpha=np.amax(A.T @ B) * lassoTol, positive=True, max_iter=lassoMaxIter)

                    # lasso.fit(A,B)
                    # _alpha = lasso.coef_
                else:
                    # alpha = np.zeros(shape=A.shape[1], dtype='float64')
                    for ii in range(alphaLen):
                        _alpha[ii] = 0.

                newSegmentation[x, y, z] = _segmentation(_alpha, _L, numOfLabels)

                # for ll in range(1,numOfLabels + 1):
                    # label = labelFusion(alpha, L == ll)
                #     label = _labelFusion(alpha, L == ll)
                #     seg = segmentation(label)
                #     if seg == 1:
                #         newSegmentation[x, y, z] = ll
                #         break

            if verboseY:
                print("\tTime:", time() - tstartY)
        if verboseX:
                print("X:", x, "\tTime:", time() - tstartX)
    return newSegmentation


def applySPBM(segImage, 
                      images, 
                      labels,
                      numOfLabels,
                      P,
                      N,
                      lassoTol=0.01,
                      lassoMaxIter=1e4, 
                      verboseX=True, 
                      verboseY=True,
                      xmin=-1,
                      xmax=-1,
                      ymin=-1,
                      ymax=-1,
                      zmin=-1,
                      zmax=-1): 
#    segImage = np.array(segImage, order='C')
#    images = np.array(images)
#    labels = np.array(labels)
#    P = np.array(P, dtype=np.int32)
#    N = np.array(N, dtype=np.int32)
    return _applySPBM(segImage, 
                              images, 
                              labels,
                              numOfLabels,
                              P,
                              N,
                              lassoTol,
                              lassoMaxIter, 
                              verboseX, 
                              verboseY,
                              xmin,
                              xmax,
                              ymin,
                              ymax,
                              zmin,
                              zmax) 
    


'''
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def applySPBM2(np.ndarray[np.uint16_t, ndim=3] segImage, 
                       np.ndarray[np.uint16_t, ndim=4] images, 
                       np.ndarray[np.uint8_t, ndim=4] labels,
                       long long numOfLabels,
                       int[::1] P,
                       int[::1] N,
                       double lassoTol=0.01,
                       double lassoMaxIter=1e4, 
                       bint verboseX=True, 
                       bint verboseY=True,
                       long long xmin=-1,
                       long long xmax=-1,
                       long long ymin=-1,
                       long long ymax=-1,
                       long long zmin=-1,
                       long long zmax=-1): 
    size = segImage.shape
    sizeImages = images[0].shape
    sizeLabels = labels[0].shape

    assert (size[0] == sizeImages[0] and size[1] == sizeImages[1] and size[2] == sizeImages[2]), "Segmentation images and images has different size."
    assert (size[0] == sizeLabels[0] and size[1] == sizeLabels[1] and size[2] == sizeLabels[2]), "Segmentation images and labels has different size."
    assert (images.shape[0] == labels.shape[0]), "Images and labels number mismatch."
    assert (numOfLabels >= 1), "Number of labels lower than one."
    assert (numOfLabels <= 10), "Number of labels must be less or equal to 10."
    

    # Allocate memory for the segmentation
    cdef np.ndarray[np.uint8_t, ndim=3] newSegmentation = np.zeros(shape=(size[0], size[1], size[2]), dtype='uint8')
    
    # x range
    if not (xmin >= N[0]//2 + P[0]//2 and xmin < size[0] - N[0]//2 - P[0]//2 - 1):
        xmin = N[0]//2 + P[0]//2
    if not (xmax >= N[0]//2 + P[0]//2 and xmax < size[0] - N[0]//2 - P[0]//2 - 1 and xmax > xmin):
        xmax = size[0] - N[0]//2 - P[0]//2 - 1
        
    # y range
    if not (ymin >= N[1]//2 + P[1]//2 and ymin < size[1] - N[1]//2 - P[1]//2 - 1):
        ymin = N[1]//2 + P[1]//2
    if not (ymax >= N[1]//2 + P[1]//2 and ymax < size[1] - N[1]//2 - P[1]//2 - 1 and ymax > ymin):
        ymax = size[1] - N[1]//2 - P[1]//2 - 1
        
    # z range
    if not (zmin >= N[2]//2 + P[2]//2 and zmin < size[2] - N[2]//2 - P[2]//2 - 1):
        zmin = N[2]//2 + P[2]//2
    if not (zmax >= N[2]//2 + P[2]//2 and zmax < size[2] - N[2]//2 - P[2]//2 - 1 and zmax > zmin):
        zmax = size[2] - N[2]//2 - P[2]//2 - 1
    
    cdef int[3] v 
    cdef int x, y, z
    cdef int labelsLen = labels.shape[0]
    cdef int LLen = labelsLen*N[0]*N[1]*N[2]
    cdef int imagesLen = images.shape[0]
    cdef int BLen = P[0]*P[1]*P[2]

    cdef np.ndarray[np.uint8_t, ndim=1] L = np.zeros(shape=(LLen), dtype='uint8')
    cdef uint8_t[::1] _L = L

    cdef np.ndarray[np.float32_t, ndim=2] A = np.zeros(shape=(BLen, LLen), dtype=np.single, order='F')
    cdef float[::1,:] _A = A

    cdef np.ndarray[np.float32_t, ndim=1] B = np.zeros(shape=(BLen), dtype=np.single)
    cdef float[::1] _B = B

    cdef np.ndarray[np.float32_t, ndim=1] AColsSquared = np.zeros(shape=(LLen), dtype=np.single, order='F')
    cdef float[::1] _AColsSquared = AColsSquared

    cdef uint8_t[:, :, :, ::1] _labels = labels
    cdef uint16_t[:, :, :, ::1] _images = images
    cdef uint16_t[:, :, ::1] _segImage = segImage

    cdef uint8_t allSame
    
    cdef int alphaLen = A.shape[1]

    cdef np.ndarray[np.float32_t, ndim=1] alpha = np.zeros(shape=A.shape[1], dtype=np.single)
    cdef float[::1] _alpha = alpha

    # For use in _gapSafeCD
    cdef np.ndarray[np.uint8_t, ndim=1] activeSet = np.zeros(shape=A.shape[1], dtype='uint8')
    cdef uint8_t[::1] _activeSet = activeSet

    # For use in _gapSafeCD
    cdef np.ndarray[np.float32_t, ndim=1] R = np.zeros(shape=A.shape[1], dtype=np.single)
    cdef float[::1] _R = R

    
    cdef float maxArg, sumArg
    cdef float colSum
    cdef long long _lassoMaxIter = <long long> lassoMaxIter

    cdef clock_t _tstartX, _tstartY


    with nogil:
        for x in range(xmin, xmax):
            v[0] = x

            if verboseX:
                # tstartX = time()
                _tstartX = clock()
            for y in range(ymin, ymax):
                v[1] = y

                if verboseY:
                    # print('\tX:', x, '\tY:', y , end=" ")
                    printf("\tX: %d \tY: %d ", x, y)
                    # tstartY = time()
                    _tstartY = clock()

                for z in range(zmin, zmax):
                    v[2] = z

                    _createL(_L, _labels, v, N)

                    allSame = 1
                    for ii in range(1,LLen):
                        if L[ii] != L[0]:
                            allSame = 0
                            break

                    if allSame == 1:
                        newSegmentation[x, y, z] = L[0]
                        continue

                    _createAFloat(_A, _images, v, P, N)
                    _createBFloat(_B, _segImage, v, P)

                    allSame = 1
                    for ii in range(BLen):
                        if B[ii] != 0:
                            allSame = 0
                            break

                    # Init weights for Lasso
                    for ii in range(alphaLen):
                        alpha[ii] = 0.

                    # If all B values are NOT 0
                    if allSame == 0:
                        maxArg = 0
                        for ii in range(LLen):
                            sumArg = 0
                            for jj in range(BLen):
                                sumArg += B[jj] * A[jj,ii]

                            if sumArg > maxArg:
                                maxArg = sumArg

                        # Center A
                        for ii in range(LLen):
                            AColsSquared[ii] = 0.0
                            colSum = 0.0
                            for jj in range(BLen):
                                colSum += A[jj,ii]

                            colSum /= BLen
                            for jj in range(BLen):
                                A[jj,ii] -= colSum
                                AColsSquared[ii] += A[jj,ii] * A[jj,ii]
#                                 if _AColsSquared[ii] != 0:
#                                     A[jj,ii] /= _AColsSquared[ii]
#                                 else:
#                                     _AColsSquared[ii] = 1

                        # Center B
                        colSum = 0.0
                        for ii in range(BLen):
                            colSum += B[ii]
                        colSum /= BLen
                        for ii in range(BLen):
                            B[ii] -= colSum

                        # _tstartY = clock()
                        _elasticNet(BLen, LLen, _R, _alpha, _AColsSquared, _B, _A, maxArg * lassoTol, 0.0, tolerance=0.0001, maxIterations=_lassoMaxIter)
                        # _elasticNet(BLen, LLen, _R, _alpha, _AColsSquared, _B, _A, maxArg * lassoTol, 0.0, tolerance=lassoTol, maxIterations=_lassoMaxIter)

                        # _gapSafeCD(BLen, LLen, _R, _alpha, _AColsSquared, _B, _A, maxArg * lassoTol, _activeSet, fce=10, tolerance=lassoTol, maxIterations=_lassoMaxIter)

                        # printf("Time 1: %f\n", (<double>clock() - _tstartY) / CLOCKS_PER_SEC)

                        # _tstartY = clock()
                        # _alpha2 = Lasso(alpha=maxArg * lassoTol, positive=True, max_iter=lassoMaxIter).fit(A,B).coef_
                        # printf("Time 2: %f\n\n", (<double>clock() - _tstartY) / CLOCKS_PER_SEC)
                        # for ii in range(LLen):
                        #     if _alpha[ii] != 0:
                        #         print(1, ii, _alpha[ii])
                        # for ii in range(LLen):
                        #     if _alpha2[ii] != 0:
                        #         print(2, ii, _alpha2[ii])
                        # print()


                    # newSegmentation[x, y, z] = _segmentationFloat(_alpha, _L, numOfLabels)
                    newSegmentation[x, y, z] = _segmentationFloatMax(_alpha, _L, numOfLabels)

                if verboseY:
                    # print("\tTime:", time() - tstartY)
                    printf("\tTime: %f\n", (<double>clock() - _tstartY) / CLOCKS_PER_SEC)
            if verboseX:
                # print("X:", x, "\tTime:", time() - tstartX)
                printf("X: %d \tTime: %f\n", x, (<double>clock() - _tstartX) / CLOCKS_PER_SEC)
    return newSegmentation
'''

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int _minResidual(uint16_t[::1,:] A, uint16_t[::1]B, uint8_t[::1] L, double[::1] alpha, int alphaLen, int BLen, long long numOfLabels):
    # Increment, for blas use
    cdef int inc = 1

    cdef double R[10]
    cdef double temp
    cdef int i, j

    for label in range(numOfLabels+1):
        R[label] = 0

        for i in range(BLen):
            temp = B[i]
            for j in range(alphaLen):
                if L[j] == label:    
                    temp -= A[i,j] * alpha[j]

            R[label] += sqrt(temp * temp)

    cdef double minRes = R[0]
    cdef int minPos = 0

    for i in range(1, numOfLabels+1):
        if R[i] < minRes:
            minRes = R[i]
            minPos = i

    return minPos

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int _minResidualFloat(float[::1,:] A, float[::1,:] B, uint8_t[::1] L, float[::1] alpha, int alphaLen, int BLen, long long numOfLabels):
    cdef float R[10]
    cdef float tempR[10]
    # cdef float temp
    cdef int i, j
    cdef uint8_t label

    for i in range(BLen):
        for label in range(numOfLabels+1):
            tempR[label] = B[i, 0]

        for j in range(alphaLen):
            tempR[L[j]] -= A[i,j] * alpha[j]

        for label in range(numOfLabels+1):
            R[label] += sqrt(tempR[label] * tempR[label])

    '''
    for label in range(numOfLabels+1):
        R[label] = 0

        for i in range(BLen):
            temp = B[i, 0]
            for j in range(alphaLen):
                if L[j] == label:    
                    temp -= A[i,j] * alpha[j]

            R[label] += sqrt(temp * temp)
    '''

    cdef float minRes = R[0]
    cdef int minPos = 0

    for i in range(1, numOfLabels+1):
        if R[i] < minRes:
            minRes = R[i]
            minPos = i

    return minPos

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[np.uint8_t, ndim=3] _applySRC(np.ndarray[np.uint16_t, ndim=3] segImage, 
                       np.ndarray[np.uint16_t, ndim=4] images, 
                       np.ndarray[np.uint8_t, ndim=4] labels,
                       long long numOfLabels,
                       int[::1] P,
                       int[::1] N,
                       double lassoTol=0.01,
                       double lassoMaxIter=1e4, 
                       bint verboseX=True, 
                       bint verboseY=True,
                       long long xmin=-1,
                       long long xmax=-1,
                       long long ymin=-1,
                       long long ymax=-1,
                       long long zmin=-1,
                       long long zmax=-1): 
    size = segImage.shape
    sizeImages = images[0].shape
    sizeLabels = labels[0].shape

    assert (size[0] == sizeImages[0] and size[1] == sizeImages[1] and size[2] == sizeImages[2]), "Segmentation images and images has different size."
    assert (size[0] == sizeLabels[0] and size[1] == sizeLabels[1] and size[2] == sizeLabels[2]), "Segmentation images and labels has different size."
    assert (images.shape[0] == labels.shape[0]), "Images and labels number mismatch."
    assert (numOfLabels >= 1), "Number of labels lower than one."
    

    # Allocate memory for the segmentation
    cdef np.ndarray[np.uint8_t, ndim=3] newSegmentation = np.zeros(shape=(size[0], size[1], size[2]), dtype='uint8')
    
    # x range
    if not (xmin >= N[0]//2 + P[0]//2 and xmin < size[0] - N[0]//2 - P[0]//2 - 1):
        xmin = N[0]//2 + P[0]//2
    if not (xmax >= N[0]//2 + P[0]//2 and xmax < size[0] - N[0]//2 - P[0]//2 - 1 and xmax > xmin):
        xmax = size[0] - N[0]//2 - P[0]//2 - 1
        
    # y range
    if not (ymin >= N[1]//2 + P[1]//2 and ymin < size[1] - N[1]//2 - P[1]//2 - 1):
        ymin = N[1]//2 + P[1]//2
    if not (ymax >= N[1]//2 + P[1]//2 and ymax < size[1] - N[1]//2 - P[1]//2 - 1 and ymax > ymin):
        ymax = size[1] - N[1]//2 - P[1]//2 - 1
        
    # z range
    if not (zmin >= N[2]//2 + P[2]//2 and zmin < size[2] - N[2]//2 - P[2]//2 - 1):
        zmin = N[2]//2 + P[2]//2
    if not (zmax >= N[2]//2 + P[2]//2 and zmax < size[2] - N[2]//2 - P[2]//2 - 1 and zmax > zmin):
        zmax = size[2] - N[2]//2 - P[2]//2 - 1
    
    cdef int[3] v 
    cdef int x, y, z
    cdef int labelsLen = labels.shape[0]
    cdef int LLen = labelsLen*N[0]*N[1]*N[2]
    cdef int imagesLen = images.shape[0]
    cdef int BLen = P[0]*P[1]*P[2]

    cdef np.ndarray[np.uint8_t, ndim=1] L = np.zeros(shape=(LLen), dtype='uint8')
    cdef uint8_t[::1] _L = L

    cdef np.ndarray[np.uint16_t, ndim=2] A = np.zeros(shape=(BLen, LLen), dtype='uint16', order='F')
    cdef uint16_t[::1,:] _A = A


    cdef np.ndarray[np.uint16_t, ndim=1] B = np.zeros(shape=(BLen), dtype='uint16')
    cdef uint16_t[::1] _B = B

    cdef uint8_t[:, :, :, ::1] _labels = labels
    cdef uint16_t[:, :, :, ::1] _images = images
    cdef uint16_t[:, :, ::1] _segImage = segImage


    cdef uint8_t allSame
    
    cdef int alphaLen = A.shape[1]

    cdef np.ndarray[np.float64_t, ndim=1] alpha = np.zeros(shape=A.shape[1], dtype='float64')
    cdef double[::1] _alpha = alpha

    cdef long long maxArg, sumArg


    for x in range(xmin, xmax):
        v[0] = x

        if verboseX:
            tstartX = time()
        for y in range(ymin, ymax):
            v[1] = y

            if verboseY:
                print('\tX:', x, '\tY:', y , end=" ")
                tstartY = time()

            for z in range(zmin, zmax):
                v[2] = z

                _createL(_L, _labels, v, N)

                allSame = 1
                for ii in range(1,LLen):
                    if _L[ii] != _L[0]:
                        allSame = 0
                        break

                if allSame == 1:
                    newSegmentation[x, y, z] = L[0]
                    continue

                _createA(_A, _images, v, P, N)
                _createB(_B, _segImage, v, P)

                allSame = 1
                for ii in range(BLen):
                    if _B[ii] != 0:
                        allSame = 0
                        break

                # max(A.T * B)
                if allSame == 0:
                    maxArg = 0
                    for ii in range(LLen):
                        sumArg = 0
                        for jj in range(BLen):
                            sumArg += <long long> A[jj,ii] * B[jj] 

                        if sumArg > maxArg:
                            maxArg = sumArg

                    _alpha = Lasso(alpha=maxArg * lassoTol, positive=True, max_iter=lassoMaxIter).fit(A,B).coef_

                    newSegmentation[x, y, z] = _minResidual(_A, _B, _L, _alpha, alphaLen, BLen, numOfLabels)
                else:
                    newSegmentation[x, y, z] = 0

            if verboseY:
                print("\tTime:", time() - tstartY)
        if verboseX:
                print("X:", x, "\tTime:", time() - tstartX)
    return newSegmentation


def applySRC(segImage, 
             images, 
             labels,
             numOfLabels,
             P,
             N,
             lassoTol=0.01,
             lassoMaxIter=1e4, 
             verboseX=True, 
             verboseY=True,
             xmin=-1,
             xmax=-1,
             ymin=-1,
             ymax=-1,
             zmin=-1,
             zmax=-1): 
    return _applySRC(segImage, 
                     images, 
                     labels,
                     numOfLabels,
                     P,
                     N,
                     lassoTol,
                     lassoMaxIter, 
                     verboseX, 
                     verboseY,
                     xmin,
                     xmax,
                     ymin,
                     ymax,
                     zmin,
                     zmax) 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def applySPBMandSRC(np.ndarray[np.uint16_t, ndim=3] segImage, 
                       np.ndarray[np.uint16_t, ndim=4] images, 
                       np.ndarray[np.uint8_t, ndim=4] labels,
                       long long numOfLabels,
                       int[::1] P,
                       int[::1] N,
                       double lassoTol=0.01,
                       double lassoMaxIter=1e4, 
                       bint verboseX=True, 
                       bint verboseY=True,
                       long long xmin=-1,
                       long long xmax=-1,
                       long long ymin=-1,
                       long long ymax=-1,
                       long long zmin=-1,
                       long long zmax=-1): 
    size = segImage.shape
    sizeImages = images[0].shape
    sizeLabels = labels[0].shape

    assert (size[0] == sizeImages[0] and size[1] == sizeImages[1] and size[2] == sizeImages[2]), "Segmentation images and images has different size."
    assert (size[0] == sizeLabels[0] and size[1] == sizeLabels[1] and size[2] == sizeLabels[2]), "Segmentation images and labels has different size."
    assert (images.shape[0] == labels.shape[0]), "Images and labels number mismatch."
    assert (numOfLabels >= 1), "Number of labels lower than one."
    

    # Allocate memory for the segmentation
    cdef np.ndarray[np.uint8_t, ndim=3] newSegmentationSPBM = np.zeros(shape=(size[0], size[1], size[2]), dtype='uint8')
    cdef np.ndarray[np.uint8_t, ndim=3] newSegmentationSRC = np.zeros(shape=(size[0], size[1], size[2]), dtype='uint8')
    
    # x range
    if not (xmin >= N[0]//2 + P[0]//2 and xmin < size[0] - N[0]//2 - P[0]//2 - 1):
        xmin = N[0]//2 + P[0]//2
    if not (xmax >= N[0]//2 + P[0]//2 and xmax < size[0] - N[0]//2 - P[0]//2 - 1 and xmax > xmin):
        xmax = size[0] - N[0]//2 - P[0]//2 - 1
        
    # y range
    if not (ymin >= N[1]//2 + P[1]//2 and ymin < size[1] - N[1]//2 - P[1]//2 - 1):
        ymin = N[1]//2 + P[1]//2
    if not (ymax >= N[1]//2 + P[1]//2 and ymax < size[1] - N[1]//2 - P[1]//2 - 1 and ymax > ymin):
        ymax = size[1] - N[1]//2 - P[1]//2 - 1
        
    # z range
    if not (zmin >= N[2]//2 + P[2]//2 and zmin < size[2] - N[2]//2 - P[2]//2 - 1):
        zmin = N[2]//2 + P[2]//2
    if not (zmax >= N[2]//2 + P[2]//2 and zmax < size[2] - N[2]//2 - P[2]//2 - 1 and zmax > zmin):
        zmax = size[2] - N[2]//2 - P[2]//2 - 1
    
    cdef int[3] v 
    cdef int x, y, z
    cdef int labelsLen = labels.shape[0]
    cdef int LLen = labelsLen*N[0]*N[1]*N[2]
    cdef int imagesLen = images.shape[0]
    cdef int BLen = P[0]*P[1]*P[2]

    cdef np.ndarray[np.uint8_t, ndim=1] L = np.zeros(shape=(LLen), dtype='uint8')
    cdef uint8_t[::1] _L = L

    cdef np.ndarray[np.uint16_t, ndim=2] A = np.zeros(shape=(BLen, LLen), dtype='uint16', order='F')
    cdef uint16_t[::1,:] _A = A


    cdef np.ndarray[np.uint16_t, ndim=1] B = np.zeros(shape=(BLen), dtype='uint16')
    cdef uint16_t[::1] _B = B

    cdef uint8_t[:, :, :, ::1] _labels = labels
    cdef uint16_t[:, :, :, ::1] _images = images
    cdef uint16_t[:, :, ::1] _segImage = segImage


    cdef uint8_t allSame
    
    cdef int alphaLen = A.shape[1]

    cdef np.ndarray[np.float64_t, ndim=1] alpha = np.zeros(shape=A.shape[1], dtype='float64')
    cdef double[::1] _alpha = alpha

    cdef long long maxArg, sumArg


    for x in range(xmin, xmax):
        v[0] = x

        if verboseX:
            tstartX = time()
        for y in range(ymin, ymax):
            v[1] = y

            if verboseY:
                print('\tX:', x, '\tY:', y , end=" ")
                tstartY = time()

            for z in range(zmin, zmax):
                v[2] = z

                _createL(_L, _labels, v, N)

                allSame = 1
                for ii in range(1,LLen):
                    if _L[ii] != _L[0]:
                        allSame = 0
                        break

                if allSame == 1:
                    newSegmentationSPBM[x, y, z] = L[0]
                    newSegmentationSRC[x, y, z] = L[0]
                    continue

                _createA(_A, _images, v, P, N)
                _createB(_B, _segImage, v, P)

                allSame = 1
                for ii in range(BLen):
                    if _B[ii] != 0:
                        allSame = 0
                        break

                # max(A.T * B)
                if allSame == 0:
                    maxArg = 0
                    for ii in range(LLen):
                        sumArg = 0
                        for jj in range(BLen):
                            sumArg += <long long> A[jj,ii] * B[jj] 

                        if sumArg > maxArg:
                            maxArg = sumArg

                    _alpha = Lasso(alpha=maxArg * lassoTol, positive=True, max_iter=lassoMaxIter).fit(A,B).coef_

                    # SRC segmentation
                    newSegmentationSRC[x, y, z] = _minResidual(_A, _B, _L, _alpha, alphaLen, BLen, numOfLabels)
                else:
                    # SRC segmentation
                    newSegmentationSRC[x, y, z] = 0

                    for ii in range(alphaLen):
                        _alpha[ii] = 0.

                # SPBM segmentation
                newSegmentationSPBM[x, y, z] = _segmentation(_alpha, _L, numOfLabels)

            if verboseY:
                print("\tTime:", time() - tstartY)
        if verboseX:
                print("X:", x, "\tTime:", time() - tstartX)
    return (newSegmentationSPBM, newSegmentationSRC)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def applySPBMandSRCSpams(np.ndarray[np.uint16_t, ndim=3] segImage, 
                         np.ndarray[np.uint16_t, ndim=4] images, 
                         np.ndarray[np.uint8_t, ndim=4] labels,
                         long long numOfLabels,
                         int[::1] P,
                         int[::1] N,
                         double lassoTol=0.01,
                         double lassoMaxIter=1e4, # Not used atm
                         bint verboseX=True, 
                         bint verboseY=True,
                         long long xmin=-1,
                         long long xmax=-1,
                         long long ymin=-1,
                         long long ymax=-1,
                         long long zmin=-1,
                         long long zmax=-1,
                         long long numThreads=-1,
                         long long lassoL=-1):
    size = segImage.shape
    sizeImages = images[0].shape
    sizeLabels = labels[0].shape

    assert (size[0] == sizeImages[0] and size[1] == sizeImages[1] and size[2] == sizeImages[2]), "Segmentation images and images has different size."
    assert (size[0] == sizeLabels[0] and size[1] == sizeLabels[1] and size[2] == sizeLabels[2]), "Segmentation images and labels has different size."
    assert (images.shape[0] == labels.shape[0]), "Images and labels number mismatch."
    assert (numOfLabels >= 1), "Number of labels lower than one."
    

    # Allocate memory for the segmentation
    cdef np.ndarray[np.uint8_t, ndim=3] newSegmentationSPBM = np.zeros(shape=(size[0], size[1], size[2]), dtype='uint8')
    cdef np.ndarray[np.uint8_t, ndim=3] newSegmentationSRC = np.zeros(shape=(size[0], size[1], size[2]), dtype='uint8')
    
    # x range
    if not (xmin >= N[0]//2 + P[0]//2 and xmin < size[0] - N[0]//2 - P[0]//2 - 1):
        xmin = N[0]//2 + P[0]//2
    if not (xmax >= N[0]//2 + P[0]//2 and xmax < size[0] - N[0]//2 - P[0]//2 - 1 and xmax > xmin):
        xmax = size[0] - N[0]//2 - P[0]//2 - 1
        
    # y range
    if not (ymin >= N[1]//2 + P[1]//2 and ymin < size[1] - N[1]//2 - P[1]//2 - 1):
        ymin = N[1]//2 + P[1]//2
    if not (ymax >= N[1]//2 + P[1]//2 and ymax < size[1] - N[1]//2 - P[1]//2 - 1 and ymax > ymin):
        ymax = size[1] - N[1]//2 - P[1]//2 - 1
        
    # z range
    if not (zmin >= N[2]//2 + P[2]//2 and zmin < size[2] - N[2]//2 - P[2]//2 - 1):
        zmin = N[2]//2 + P[2]//2
    if not (zmax >= N[2]//2 + P[2]//2 and zmax < size[2] - N[2]//2 - P[2]//2 - 1 and zmax > zmin):
        zmax = size[2] - N[2]//2 - P[2]//2 - 1
    
    cdef int[3] v 
    cdef int x, y, z
    cdef int labelsLen = labels.shape[0]
    cdef int LLen = labelsLen*N[0]*N[1]*N[2]
    cdef int imagesLen = images.shape[0]
    cdef int BLen = P[0]*P[1]*P[2]

    cdef np.ndarray[np.uint8_t, ndim=1] L = np.zeros(shape=(LLen), dtype='uint8')
    cdef uint8_t[::1] _L = L

    cdef np.ndarray[np.float32_t, ndim=2] A = np.zeros(shape=(BLen, LLen), dtype=np.single, order='F')
    cdef float[::1,:] _A = A

    cdef np.ndarray[np.float32_t, ndim=2] B = np.zeros(shape=(BLen, 1), dtype=np.single, order='F')
    cdef float[::1,:] _B = B


    cdef uint8_t[:, :, :, ::1] _labels = labels
    cdef uint16_t[:, :, :, ::1] _images = images
    cdef uint16_t[:, :, ::1] _segImage = segImage


    cdef uint8_t allSame
    
    cdef int alphaLen = A.shape[1]

    # cdef np.ndarray[np.float32_t, ndim=1] alpha = np.zeros(shape=A.shape[1], dtype=np.single)
    # cdef float[::1] _alpha = alpha
    cdef float[::1] _alpha

    cdef float maxArg, sumArg, colSum, sumSqred

    cdef int ii, jj


    for x in range(xmin, xmax):
        v[0] = x

        if verboseX:
            tstartX = time()
        for y in range(ymin, ymax):
            v[1] = y

            if verboseY:
                print('\tX:', x, '\tY:', y , end=" ")
                tstartY = time()

            for z in range(zmin, zmax):
                v[2] = z

                _createL(_L, _labels, v, N)

                allSame = 1
                for ii in range(1,LLen):
                    if _L[ii] != _L[0]:
                        allSame = 0
                        break

                if allSame == 1:
                    newSegmentationSPBM[x, y, z] = L[0]
                    newSegmentationSRC[x, y, z] = L[0]
                    continue

                _createAFloat(_A, _images, v, P, N)
                _createBFloat2(_B, _segImage, v, P)

                allSame = 1
                for ii in range(BLen):
                    if _B[ii, 0] != 0.:
                        allSame = 0
                        break

                if allSame == 0:
                    # Center, normalize A
                    # for every column
                    for ii in range(LLen):
                        colSum = 0.
                        for jj in range(BLen):
                            colSum += A[jj,ii]

                        colSum /= BLen
                        sumSqred = 0.
                        for jj in range(BLen):
                            A[jj,ii] -= colSum
                            sumSqred += A[jj,ii] * A[jj,ii]

                        if sumSqred != 0.:
                            for jj in range(BLen):
                                A[jj,ii] /= sumSqred 

                    # Center, normalize B
                    colSum = 0.0
                    for ii in range(BLen):
                        colSum += B[ii, 0]
                    colSum /= BLen
                    sumSqred = 0.
                    for ii in range(BLen):
                        B[ii, 0] -= colSum
                        sumSqred += B[ii, 0] * B[ii, 0]
                    if sumSqred != 0.:
                        for ii in range(BLen):
                            B[ii, 0] /= sumSqred

                    # max(A.T * B)
                    maxArg = 0
                    for ii in range(LLen):
                        sumArg = 0
                        for jj in range(BLen):
                            sumArg += A[jj,ii] * B[jj, 0] 

                        if sumArg > maxArg:
                            maxArg = sumArg

                    _alpha = spams.lasso(B, A, 
                                   return_reg_path = False, 
                                   lambda1 = maxArg * lassoTol, 
                                   lambda2 = 0.,
                                   pos = True,
                                   mode = 2,
                                   numThreads = numThreads,
                                   L = lassoL,
                                   max_length_path = <long long>lassoMaxIter
                                  ).A[:,0]

                    # SPBM segmentation
                    # newSegmentationSPBM[x, y, z] = _segmentationFloat(_alpha, _L, numOfLabels, alphaLen)
                    newSegmentationSPBM[x, y, z] = _segmentationFloatMax(_alpha, _L, numOfLabels, alphaLen)

                    # SRC segmentation
                    newSegmentationSRC[x, y, z] = _minResidualFloat(_A, _B, _L, _alpha, alphaLen, BLen, numOfLabels)
                else:
                    # SPBM segmentation
                    newSegmentationSPBM[x, y, z] = 0

                    # SRC segmentation
                    newSegmentationSRC[x, y, z] = 0

            if verboseY:
                print("\tTime:", time() - tstartY)
        if verboseX:
                print("X:", x, "\tTime:", time() - tstartX)

    return (newSegmentationSPBM, newSegmentationSRC)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def applySPEP(np.ndarray[np.uint16_t, ndim=3] segImage, 
              np.ndarray[np.uint16_t, ndim=4] images, 
              np.ndarray[np.uint8_t, ndim=4] labels,
              long long numOfLabels,
              int[::1] P,
              int[::1] N,
              bint verboseX=True, 
              bint verboseY=True,
              long long xmin=-1,
              long long xmax=-1,
              long long ymin=-1,
              long long ymax=-1,
              long long zmin=-1,
              long long zmax=-1,
              double th = 0.95,
             ):
    size = segImage.shape
    sizeImages = images[0].shape
    sizeLabels = labels[0].shape

    assert (size[0] == sizeImages[0] and size[1] == sizeImages[1] and size[2] == sizeImages[2]), "Segmentation images and images has different size."
    assert (size[0] == sizeLabels[0] and size[1] == sizeLabels[1] and size[2] == sizeLabels[2]), "Segmentation images and labels has different size."
    assert (images.shape[0] == labels.shape[0]), "Images and labels number mismatch."
    assert (numOfLabels >= 1), "Number of labels lower than one."
    

    # Allocate memory for the segmentation
    cdef np.ndarray[np.uint8_t, ndim=3] newSegmentation = np.zeros(shape=(size[0], size[1], size[2]), dtype='uint8')
    
    # x range
    if not (xmin >= N[0]//2 + P[0]//2 and xmin < size[0] - N[0]//2 - P[0]//2 - 1):
        xmin = N[0]//2 + P[0]//2
    if not (xmax >= N[0]//2 + P[0]//2 and xmax < size[0] - N[0]//2 - P[0]//2 - 1 and xmax > xmin):
        xmax = size[0] - N[0]//2 - P[0]//2 - 1
        
    # y range
    if not (ymin >= N[1]//2 + P[1]//2 and ymin < size[1] - N[1]//2 - P[1]//2 - 1):
        ymin = N[1]//2 + P[1]//2
    if not (ymax >= N[1]//2 + P[1]//2 and ymax < size[1] - N[1]//2 - P[1]//2 - 1 and ymax > ymin):
        ymax = size[1] - N[1]//2 - P[1]//2 - 1
        
    # z range
    if not (zmin >= N[2]//2 + P[2]//2 and zmin < size[2] - N[2]//2 - P[2]//2 - 1):
        zmin = N[2]//2 + P[2]//2
    if not (zmax >= N[2]//2 + P[2]//2 and zmax < size[2] - N[2]//2 - P[2]//2 - 1 and zmax > zmin):
        zmax = size[2] - N[2]//2 - P[2]//2 - 1
    
    cdef int[3] v 
    cdef int x, y, z
    cdef int labelsLen = labels.shape[0]
    cdef int LLen = labelsLen*N[0]*N[1]*N[2]
    cdef int imagesLen = images.shape[0]
    cdef int BLen = P[0]*P[1]*P[2]

    cdef np.ndarray[np.uint8_t, ndim=1] L = np.zeros(shape=(LLen), dtype='uint8')
    cdef uint8_t[::1] _L = L

    cdef np.ndarray[np.float32_t, ndim=2] A = np.zeros(shape=(BLen, LLen), dtype=np.single, order='F')
    cdef float[::1,:] _A = A

    cdef np.ndarray[np.float32_t, ndim=1] B = np.zeros(shape=(BLen), dtype=np.single, order='F')
    cdef float[::1] _B = B


    cdef uint8_t[:, :, :, ::1] _labels = labels
    cdef uint16_t[:, :, :, ::1] _images = images
    cdef uint16_t[:, :, ::1] _segImage = segImage


    cdef uint8_t allSame
    
    cdef int wLen = A.shape[1]

    cdef np.ndarray[np.float32_t, ndim=1] w = np.zeros(shape=A.shape[1], dtype=np.single)
    cdef float[::1] _w = w

    cdef int ii, jj

    cdef int minarg = -1 
    cdef float minn, temp, h, ss
    cdef float meani, meanj, sigmai, sigmaj

    cdef int possition = P[2] * P[1] * (P[0]//2) + P[1] * (P[0]//2) + P[2]//2
    cdef float eps = np.finfo(np.float32).eps

    for x in range(xmin, xmax):
        v[0] = x

        if verboseX:
            tstartX = time()
        for y in range(ymin, ymax):
            v[1] = y

            if verboseY:
                print('\tX:', x, '\tY:', y , end=" ")
                tstartY = time()

            for z in range(zmin, zmax):
                v[2] = z

                _createL(_L, _labels, v, N)

                allSame = 1
                for ii in range(1,LLen):
                    if _L[ii] != _L[0]:
                        allSame = 0
                        break

                if allSame == 1:
                    newSegmentation[x, y, z] = L[0]
                    continue

                _createAFloat(_A, _images, v, P, N)
                _createBFloat(_B, _segImage, v, P)

                allSame = 1
                for ii in range(BLen):
                    if _B[ii] != 0.:
                        allSame = 0
                        break

                allSame == 0
                if allSame == 0:
                    # argmin(P(B) - P(A[:,i]) + eps
                    for ii in range(LLen):
                        temp = 0.
                        for jj in range(BLen):
                            temp += (B[jj] - A[jj, ii])**2

                        # cache temp to w
                        w[ii] = temp

                        if ii == 0:
                            minn = temp
                            minarg = 0
                        else:
                            if temp < minn:
                                minn = temp
                                minarg = ii
                    # end of argmin

                    h = A[possition, minarg] + eps

                    # mean and standard deviation for B
                    meani = 0.
                    for ii in range(BLen):
                        meani += B[ii]
                    meani /= BLen

                    sigmai = 0.
                    for ii in range(BLen):
                        sigmai += (B[ii] - meani)**2
                    sigmai = sqrt(sigmai / BLen)
                    

                    for ii in range(LLen):
                        # mean and standard deviation for A[:, ii]
                        meanj = 0.
                        for jj in range(BLen):
                            meanj += A[jj, ii]
                        meanj /= BLen

                        sigmaj = 0.
                        for jj in range(BLen):
                            sigmaj += (A[jj, ii] - meanj)**2
                        sigmaj = sqrt(sigmaj / BLen)

                        ss = ((2 * meani * meanj) / (meani**2 + meanj**2)) * \
                             ((2 * sigmai * sigmaj) / (sigmai**2 + sigmaj**2))

                        if ss > th:
                            w[ii] = exp(-w[ii] / (h * BLen))
                        else:
                            w[ii] = 0

                    # segmentation
                    newSegmentation[x, y, z] = _segmentationFloat(_w, 
                                                        _L, 
                                                        numOfLabels, 
                                                        wLen)
                else:
                    # SPBM segmentation
                    newSegmentation[x, y, z] = 0

            if verboseY:
                print("\tTime:", time() - tstartY)
        if verboseX:
                print("X:", x, "\tTime:", time() - tstartX)

    return newSegmentation


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def applyMV(np.ndarray[np.uint16_t, ndim=3] segImage, 
            np.ndarray[np.uint16_t, ndim=4] images, 
            np.ndarray[np.uint8_t, ndim=4] labels,
            long long numOfLabels,
            bint verboseX=True, 
            bint verboseY=True,
            long long xmin=-1,
            long long xmax=-1,
            long long ymin=-1,
            long long ymax=-1,
            long long zmin=-1,
            long long zmax=-1,
           ):
    size = segImage.shape
    sizeImages = images[0].shape
    sizeLabels = labels[0].shape

    assert (size[0] == sizeImages[0] and size[1] == sizeImages[1] and size[2] == sizeImages[2]), "Segmentation images and images has different size."
    assert (size[0] == sizeLabels[0] and size[1] == sizeLabels[1] and size[2] == sizeLabels[2]), "Segmentation images and labels has different size."
    assert (images.shape[0] == labels.shape[0]), "Images and labels number mismatch."
    assert (numOfLabels >= 1), "Number of labels lower than one."
    
    xmin = 0
    xmax = size[0]
    ymin = 0
    ymax = size[1]
    zmin = 0
    zmax = size[2]
        
    # Allocate memory for the segmentation
    cdef np.ndarray[np.uint8_t, ndim=3] newSegmentation= np.zeros(shape=(size[0], size[1], size[2]), dtype='uint8')
    
    cdef int[3] v 
    cdef int x, y, z
    cdef int labelsLen = labels.shape[0]
    cdef int LLen = labelsLen
    cdef int imagesLen = images.shape[0]

    cdef np.ndarray[np.uint8_t, ndim=1] L = np.zeros(shape=(LLen), dtype='uint8')
    cdef uint8_t[::1] _L = L

    cdef uint8_t[:, :, :, ::1] _labels = labels

    cdef uint8_t allSame
    
    cdef int alphaLen = LLen

    cdef np.ndarray[np.float32_t, ndim=1] alpha = np.ones(shape=LLen, dtype=np.single)
    cdef float[::1] _alpha = alpha

    cdef int ii, jj

    NN = np.array([1,1,1], dtype=np.int32)
    cdef int[::1] N = NN


    for x in range(xmin, xmax):
        v[0] = x

        if verboseX:
            tstartX = time()
        for y in range(ymin, ymax):
            v[1] = y

            if verboseY:
                print('\tX:', x, '\tY:', y , end=" ")
                tstartY = time()

            for z in range(zmin, zmax):
                v[2] = z

                _createL(_L, _labels, v, N)

                allSame = 1
                for ii in range(1,LLen):
                    if _L[ii] != _L[0]:
                        allSame = 0
                        break

                if allSame == 1:
                    newSegmentation[x, y, z] = L[0]
                    continue

                newSegmentation[x, y, z] = _segmentationFloatMax(_alpha, _L, numOfLabels, alphaLen)


            if verboseY:
                print("\tTime:", time() - tstartY)
        if verboseX:
                print("X:", x, "\tTime:", time() - tstartX)

    return newSegmentation

