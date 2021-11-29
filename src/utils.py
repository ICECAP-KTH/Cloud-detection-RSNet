#!/usr/bin/env python
import numpy as np


def image_normalizer(img, params):
    """
    Clip an image at certain threshold value, and then normalize to values between 0 and 1.
    Threshold is used for contrast enhancement.
    """
    # Enhance contrast of entire image
    # The Sentinel-2 data has 15 significant bits, but normally maxes out between 10000-20000.
    # Here we clip and normalize to value between 0 and 1
    img_norm = np.clip(img, 0, params.norm_threshold)
    img_norm = img_norm / params.norm_threshold

    return img_norm


def patch_image(img, patch_size, overlap):
    """
    Split up an image into smaller overlapping patches
    """
    # TODO: Get the size of the padding right.
    # Add zeropadding around the image (has to match the overlap)
    img_shape = np.shape(img)
    img_padded = np.zeros((img_shape[0] + 2 * patch_size, img_shape[1] + 2 * patch_size, img_shape[2]))
    img_padded[overlap:overlap + img_shape[0], overlap:overlap + img_shape[1], :] = img

    # Find number of patches
    n_width = int((np.size(img_padded, axis=0) - patch_size) / (patch_size - overlap))
    n_height = int((np.size(img_padded, axis=1) - patch_size) / (patch_size - overlap))

    # Now cut into patches
    n_bands = np.size(img_padded, axis=2)
    img_patched = np.zeros((n_height * n_width, patch_size, patch_size, n_bands), dtype=img.dtype)
    for i in range(0, n_width):
        for j in range(0, n_height):
            id = n_height * i + j

            # Define "pixel coordinates" of the patches in the whole image
            xmin = patch_size * i - i * overlap
            xmax = patch_size * i + patch_size - i * overlap
            ymin = patch_size * j - j * overlap
            ymax = patch_size * j + patch_size - j * overlap

            # Cut out the patches.
            # img_patched[id, width , height, depth]
            img_patched[id, :, :, :] = img_padded[xmin:xmax, ymin:ymax, :]

    return img_patched, n_height, n_width  # n_height and n_width are necessary for stitching image back together


def stitch_image(img_patched, n_height, n_width, patch_size, overlap):
    """
    Stitch the overlapping patches together to one large image (the original format)
    """
    isz_overlap = patch_size - overlap  # i.e. remove the overlap

    n_bands = np.size(img_patched, axis=3)

    img = np.zeros((n_width * isz_overlap, n_height * isz_overlap, n_bands))

    # Define bbox of the interior of the patch to be stitched (not required if using Cropping2D layer in model)
    # xmin_overlap = int(overlap / 2)
    # xmax_overlap = int(patch_size - overlap / 2)
    # ymin_overlap = int(overlap / 2)
    # ymax_overlap = int(patch_size - overlap / 2)

    # Stitch the patches together
    for i in range(0, n_width):
        for j in range(0, n_height):
            id = n_height * i + j

            # Cut out the interior of the patch
            # interior_path = img_patched[id, xmin_overlap:xmax_overlap, ymin_overlap:ymax_overlap, :]
            interior_patch = img_patched[id, :, :, :]

            # Define "pixel coordinates" of the patches in the whole image
            xmin = isz_overlap * i
            xmax = isz_overlap * i + isz_overlap
            ymin = isz_overlap * j
            ymax = isz_overlap * j + isz_overlap

            # Insert the patch into the stitched image
            img[xmin:xmax, ymin:ymax, :] = interior_patch

    return img


def predict_img(model, params, img):
    """
    Run prediction on an full image
    """
    # Find dimensions
    img_shape = np.shape(img)

    # Normalize the product
    img = image_normalizer(img, params)

    # Patch the image in patch_size * patch_size pixel patches
    img_patched, n_height, n_width = patch_image(img, patch_size=params.patch_size, overlap=params.overlap)

    # Now find all completely black patches and inpaint partly black patches
    indices = []  # Used to ignore completely black patches during prediction
    for i in range(0, np.shape(img_patched)[0]):  # For all patches
        if np.any(img_patched[i, :, :, :] == 0):  # If any black pixels
            if np.mean(img_patched[i, :, :, :] != 0):  # Ignore completely black patches
                indices.append(i)  # Use the patch for prediction
                # Fill in zero pixels using the non-zero pixels in the patch
                for j in range(0, np.shape(img_patched)[3]):  # Loop over each spectral band in the patch
                    # Bands do not always overlap. Use mean of all bands if zero-slice is found, otherwise use
                    # mean of the specific band
                    if np.mean(img_patched[i, :, :, j]) == 0:
                        mean_value = np.mean(img_patched[i, img_patched[i, :, :, :] != 0])
                    else:
                        mean_value = np.mean(img_patched[i, img_patched[i, :, :, j] != 0, j])

                    img_patched[i, img_patched[i, :, :, j] == 0, j] = mean_value
        else:
            indices.append(i)  # Use the patch for prediction

    # Now do the cloud masking (on non-zero patches according to indices)
    # start_time = time.time()
    predicted_patches = np.zeros((np.shape(img_patched)[0],
                                  params.patch_size - params.overlap, params.patch_size - params.overlap, 1))
    predicted_patches[indices, :, :, :] = model.predict(img_patched[indices, :, :, :], params)
    # exec_time = str(time.time() - start_time)
    # print("Prediction of patches (not including splitting and stitching) finished in: " + exec_time + "s")

    # Stitch the patches back together
    predicted_stitched = stitch_image(predicted_patches, n_height, n_width, patch_size=params.patch_size,
                                      overlap=params.overlap)

    # Now throw away the padded sections from the overlap
    padding = int(params.overlap / 2)  # The overlap is over 2 patches, so you need to throw away overlap/2 on each
    predicted_mask = predicted_stitched[padding - 1:padding - 1 + img_shape[0],
                     # padding-1 because it is index in array
                     padding - 1:padding - 1 + img_shape[1], :]

    # Throw away the inpainting of the zero pixels in the individual patches
    # The summation is done to ensure that all pixels are included. The bands do not perfectly overlap (!)
    predicted_mask[np.sum(img, axis=2) == 0] = 0

    # Threshold the prediction
    predicted_binary_mask = predicted_mask >= np.float32(params.threshold)

    return predicted_mask, predicted_binary_mask

