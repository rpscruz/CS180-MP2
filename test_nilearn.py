import os
from nilearn.input_data import MultiNiftiMasker # NiftiMasker
from nilearn.plotting import plot_roi, show
from nilearn.image.image import mean_img


# We first create a masker, giving it the options that we care
# about. Here we use standardizing of the data, as it is often important
# for decoding
mask_filename = os.getcwd() + "/dataset/train/Patient_01/GT.nii.gz"
scan_filename = os.getcwd() + "/dataset/train/Patient_01/Patient_01.nii.gz"

masker = MultiNiftiMasker(mask_img=mask_filename, standardize=True)
print(masker)


# We give the masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn

masker.fit(scan_filename)
#masker.transform(scan_filename)
scan_masked = masker.fit_transform(scan_filename)

# calculate mean image for the background
mean_func_img = mean_img(scan_filename)
'''
plot_roi(masker.mask_img_, mean_func_img, display_mode='y', cut_coords=4, title="Mask")
show()
'''
# maxes = np.max(labelArray, axis=0)
# calculate mean image for the background
# mean_func_img = mean_img(filename)

# plot_roi(mask_img, mean_func_img, display_mode='y', cut_coords=4, title="Mask")

# https://nilearn.github.io/auto_examples/02_decoding/plot_simulated_data.html
# Baysian ridge
# https://scikit-learn.org/0.18/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge

