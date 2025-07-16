import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import cv2

# if DICOM is in raw format, we can transform the data to "human-friendly" (the radiologist's) view
# if dicom.PhotometricInterpretation == "MONOCHROME1", then pixel intensities must be reversed
# clip image to reduce brightness artifacts
# make data range from 0 to max_dtype_value
def read_xray(path, voi_lut = True, fix_monochrome = True, clip_eps = 1e-3):
    dicom = pydicom.dcmread(path)
    
    dtype = dicom.pixel_array.dtype

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    # clipping images can help dealing with bright artifacts, e.g., caused by nipple piercings.
    # Without clipping, other pixel values are shifted towards black when normalizing towards range [0,255]. See Raddar's comment in (https://www.kaggle.com/code/radek1/how-to-process-dicom-images-to-pngs/comments#2056978)
    if clip_eps > 0:
        qq = np.quantile(data,[clip_eps,1-clip_eps])
        data = data.clip(*qq).astype(dtype)

    data = data - np.min(data)
    # Normalize pixel values to full bit range, iff they are int
    #if dtype.kind == 'f':
    #    max_dtype_value = np.finfo(dtype).max
    if dtype.kind in ['u', 'i']:
        max_dtype_value = np.iinfo(dtype).max
        
        if np.max(data) > 0: # Handle blank images
            data = data / np.max(data)
        data = (data * max_dtype_value).astype(dtype)
        
    return dicom, data
    
# TODO: Andreas found cases with false flag information in the EMBED data set.
# A more robust but less clean idea is to consider an image as being shown on the right iff the mean pixel intensity on the right half space > the one on the left half space and flip accordingly.
def flip_image_left(dicom, image, verbose = True):

    # flip right views to the left
    if dicom.FieldOfViewHorizontalFlip == 'NO' and dicom.ImageLaterality == 'R':
        if verbose:
            print('flipping R view shown right to left')
        dicom.FieldOfViewHorizontalFlip = 'YES'
        image = cv2.flip(image,1)

    # flip left breasts viewed on the right to the left
    if dicom.FieldOfViewHorizontalFlip == 'YES' and dicom.ImageLaterality == 'L':
        if verbose:
            print('flipping L view shown right to left')
        dicom.FieldOfViewHorizontalFlip = 'NO'
        image = cv2.flip(image,1)
        
    return dicom, image

def image_resize(image, width = None, height = None, inter = cv2.INTER_LINEAR):

    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        width = int(w * r)
    if height is None:
        r = width / float(w)
        height = int(h * r)
    resized = cv2.resize(image, (width, height), interpolation = inter)

    return resized

# CLAHE Equalization (see https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html for explanation)
def hist_equalization(image, hist_type = 'CLAHE', clipLimit = None, castType = np.uint8):
    dtype = image.dtype
    if hist_type == 'CLAHE':
        if clipLimit is None:
            clipLimit = 10.0 #looks OK for uint8 casting
            if castType == np.uint16:
                clipLimit *= 2**8 #change clipLimit from uint8 to uint16
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
        img_max = image.max()
        max_castType_value = np.iinfo(castType).max
        image = castType(image/img_max*max_castType_value)
        image = clahe.apply(image)
        image = (image/max_castType_value*img_max)
    else:
        # Simple Histogram Equalization
        dtype = image.dtype
        if image.dtype.kind == 'f':
            max_dtype_value = np.finfo(dtype).max
        else:
            max_dtype_value = np.iinfo(dtype).max

        image = cv2.normalize(image, None, 0, max_dtype_value, cv2.NORM_MINMAX)

    return(image.astype(dtype))

def pad_to_aspect_ratio(image, target_aspect_ratio):
    current_ratio = image.shape[0]/image.shape[1]
    padded_image = image
    if target_aspect_ratio > current_ratio:
        # need to pad y-axis, up and down
        padding_size = int(image.shape[1] * target_aspect_ratio - image.shape[0])
        if padding_size % 2 == 0:
            ps1 = ps2 = padding_size // 2
        else:
            ps1 = (padding_size+1) // 2
            ps2 = ps1 + 1

        padded_image = np.vstack((np.tile(image[0].mean(), (ps1, image.shape[1])).astype(image.dtype), image, np.tile(image[-1].mean(), (ps2, image.shape[1])).astype(image.dtype)))
        padded_image.shape[0]/padded_image.shape[1]
    if target_aspect_ratio < current_ratio:
        # need to pad x-axis on the right
        padding_size = int(image.shape[0] / target_aspect_ratio - image.shape[1])
        padded_image = np.hstack((image, np.tile(image[:,-1].mean(), (image.shape[0],padding_size)).astype(image.dtype)))
    return padded_image

from ..cropping.crop_mammogram import crop_img_from_largest_connected, image_orientation
def crop_image(dicom, image):
    window_location, rightmost_points, bottommost_points, distance_from_starting_side = crop_img_from_largest_connected(image, image_orientation(dicom.FieldOfViewHorizontalFlip, dicom.ImageLaterality))

    top, bottom, left, right = window_location
    dicom.rightmost_points = rightmost_points
    dicom.bottommost_points = bottommost_points
    dicom.distance_from_starting_side = distance_from_starting_side

    return dicom, image[top:bottom, left:right]


import src.optimal_centers.calc_optimal_centers as calc_optimal_centers
# creates all feasible windows of a target image size and chooses the one with most share of non-zero pixels.
# Note: I think there was an error in the implementation of the NYU-guys, concerning the construction of the feasible windows. I corrected for it.
#       It only manifests if x-window-range becomes smaller than the actual cropped image size
def extract_center(dicom, image, target_dims = {'CC': (2677, 1942), 'MLO': (2974, 1748)}, corrected = True):
    """
    Compute the optimal center for an image
    """
    dicom, image = flip_image_left(dicom, image)
    if not hasattr(dicom, 'rightmost_points'):
        dicom, image = crop_image(dicom, image)
    
    if dicom.ViewPosition == "MLO":
        tl_br_constraint = calc_optimal_centers.get_bottomrightmost_pixel_constraint(
            rightmost_x=dicom.rightmost_points[1],
            bottommost_y=dicom.bottommost_points[0], corrected=corrected,
        )
    elif dicom.ViewPosition == "CC":
        tl_br_constraint = calc_optimal_centers.get_rightmost_pixel_constraint(
            rightmost_x=dicom.rightmost_points[1], corrected=corrected,
        )
    else:
        raise RuntimeError(dicom.ViewPosition)

    optimal_center = calc_optimal_centers.get_image_optimal_window_info(
        image,
        com=np.array(image.shape) // 2,
        window_dim=np.array(target_dims[dicom.ViewPosition]),
        tl_br_constraint=tl_br_constraint,
    )
    wy = optimal_center["window_dim_y"]
    wx = optimal_center["window_dim_x"]
    cy = optimal_center["best_center_y"]
    cx = optimal_center["best_center_x"]
    top, bottom, left, right = [cy - wy//2, cy + wy//2 + (wy % 2), cx - wx//2, cx + wx//2 + (wx % 2)]

    return dicom, image[top:bottom, left:right], [wy, wx], [cy, cx], optimal_center['fraction']