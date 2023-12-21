import SimpleITK as sitk
import os
import numpy as np
import cv2
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import cv2
from skimage import feature
import argparse
import glob
import cv2
import imutils


data_path = '/Users/bran/Downloads/SD_spine/CSM-AI' ## replace this with your folder containing data with the same structure of CSM-AI
pat_list = os.listdir(os.path.join(data_path, 'images'))

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist

for pp in pat_list:
    if pp != '.DS_Store':
        print(pp)
        pat_shape = []
        pat_texture = []

        #### extract shape features based on Hu moments. 
        sag_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path,  'derivatives', pp, 'sag-seg.nii.gz')))
        non_empty_slices = [slice_ for slice_ in sag_mask if np.any(slice_)]
        # Create a new 3D array with non-empty slices
        new_3d_mask_array = np.array(non_empty_slices)
        
        for ii in range(np.shape(new_3d_mask_array)[0]):
            slice = new_3d_mask_array[ii]
            slice[slice==2] = 1
            cnts = cv2.findContours(np.uint8(slice), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

        ## extract the ROI from the image, resize it to a canonical size,
        ## compute the Hu Moments feature vector for the ROI, and update
        ## the data matrix
            (x, y, w, h) = cv2.boundingRect(c)
            roi = cv2.resize(new_3d_mask_array[ii, y:y + h, x:x + w], (100, 100))
            moments = cv2.HuMoments(cv2.moments(roi)).flatten()
            pat_shape.append(moments)
        pat_shape = np.asarray(pat_shape)
        mean_pat_shape = np.mean(pat_shape, axis = 0)
        # print(mean_pat_shape)
        np.save(os.path.join(os.path.join(data_path,  'derivatives', pp, 'shape_features.nii.gz')), mean_pat_shape)

    ### extract shape features based on LBP. 


        axial_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path,  'derivatives', pp, 'cor-seg.nii.gz')))
        axial_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path,  'images', pp, 'cor.nii.gz')))

        non_empty_slices = [slice_ for slice_ in axial_mask if np.any(slice_)]
        # Create a new 3D array with non-empty slices
        new_3d_mask_array = np.array(non_empty_slices)
        # print(np.sum(new_3d_mask_array))
        # print(np.shape(new_3d_mask_array))
        empty_slice_indices = [i for i, slice_ in enumerate(axial_mask) if np.any(slice_)]

        new_3d_image_array = []
        for ii in empty_slice_indices:
            new_3d_image_array.append(axial_image[ii])
        new_3d_image_array = np.array(new_3d_image_array)
        # print(np.shape(new_3d_image_array))
        # print(np.sum(new_3d_image_array))
        ## parameters for LBP
        desc = LocalBinaryPatterns(24, 8)
        size = 48

        for ii in range(np.shape(new_3d_mask_array)[0]):
            mask_array = new_3d_mask_array[ii]
            image_array = new_3d_image_array[ii]
            M = cv2.moments(mask_array)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
           # print(cX, cY)
            patch = image_array[int(cX-size/2):int(cX+size/2), int(cY-size/2):int(cY+size/2)]
            patch = np.uint8(255*patch/np.max(patch))
            patch_mask = mask_array[int(cX-size/2):int(cX+size/2), int(cY-size/2):int(cY+size/2)]
            hist = desc.describe(patch*patch_mask)
            pat_texture.append(hist)
        mean_pat_texture = np.mean(pat_texture, axis = 0)
        # print(mean_pat_texture)
        np.save(os.path.join(os.path.join(data_path,  'derivatives', pp, 'texture_features.nii.gz')), mean_pat_shape)

        # print(mean_pat_texture)











