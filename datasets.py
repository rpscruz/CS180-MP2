import os
import nibabel
import matplotlib.pyplot as plt
import numpy as np
import skimage.segmentation as seg
from skimage.exposure import histogram
import cv2
from scipy.misc import imsave

dir_path = os.getcwd()


from scipy.ndimage import zoom
def load_scans(samples_len):
    scans = np.ndarray(shape=(0, 512, 512))
    masks = np.ndarray(shape=(0, 512, 512))
    print(scans.shape)
    print(scans)


    for i in range(0 + 1, samples_len + 1): # 40 + 1): # 1-indexed data
        print(str(i).rjust(2, '0'))
        scan_filename = "{}/dataset_preprocess/train/scans/Patient_{}.nii.gz".format(dir_path, str(i).rjust(2, '0'))
        mask_filename = "{}/dataset_preprocess/train/masks/Patient_{}_GT.nii.gz".format(dir_path, str(i).rjust(2, '0'))

        print("load_scans: loaded file")
        scan = nibabel.load(scan_filename)
        mask = nibabel.load(mask_filename)

        print("load_scans: loaded data")
        scan_data = scan.get_data()
        mask_data = mask.get_data()

        print(mask_data)
       
        print(mask_data.shape)
        scan_data = scan_data.T
        mask_data = mask_data.T
        print(mask_data.shape)
        '''
        for slide in range(len(scan_data)):
            print(scan_data[slide])
            imsave("data\\scans\\{}_{}.png".format(i, slide), scan_data[slide])
            imsave("data\\masks\\{}_{}.png".format(i, slide), mask_data[slide])
        '''
        for slide in range(len(mask_data)):
            print(mask_data[slide])
            imsave("data\\masks\\{}_{}.png".format(i, slide), mask_data[slide])


        # scans = np.concatenate((scans, scan_data), axis=0)
        '''
        plt.imshow(scans[0], 'gray')
        plt.show()
        '''

        print(scans.shape)
        print(scan_data.shape)

        # test = concatenate((scan_data, scans), axis=0)
    return scans, masks

   
scans, masks = load_scans(3)