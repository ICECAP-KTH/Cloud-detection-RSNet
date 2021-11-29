import os
import time
import numpy as np
import tifffile as tiff
import cv2
from PIL import Image
from src.utils import predict_img


def evaluate_test_set(model, params, test_dataset):
    # Get the name of all the products (scenes)
    data_path = params.project_path + "data/raw/KTH/" + test_dataset + '/'
    data_output_path = params.project_path + "data/output/KTH/" + test_dataset + '/'
    products = sorted(os.listdir(data_path))
    products = [p for p in products if '.tif' in p]

    start_time_dataset = time.time()

    for count, product in enumerate(products):
        # Time the prediction
        print('Predicting product ', count, ':', product)

        start_time_product = time.time()

        # Load data
        img_all_bands = tiff.imread(data_path + product)
        img_all_bands[:, :, 0:4] = tiff.imread(data_path + product)

        # Load the relevant bands and the mask
        img = np.zeros((np.shape(img_all_bands)[0], np.shape(img_all_bands)[1], np.size(params.bands)))
        img[:, :, 0] = img_all_bands[:, :, 0]

        # Pad the image for improved borders
        padding_size = params.overlap
        npad = ((padding_size, padding_size), (padding_size, padding_size), (0, 0))
        img_padded = np.pad(img, pad_width=npad, mode='symmetric')

        # Predict the images
        _, predicted_binary_mask_padded = predict_img(model, params, img_padded)

        # Remove padding
        predicted_binary_mask = predicted_binary_mask_padded[padding_size:-padding_size, padding_size:-padding_size, :]

        #   Apply morphological operations on the generated binary mask
        operation = params.morph_op
        kernel = np.ones((7, 7), np.uint16)

        arr = np.uint16(predicted_binary_mask[:, :, 0] * 65535)

        if operation == 'erode':
            arr = cv2.erode(arr, kernel, iterations=1)
        elif operation == 'open':
            arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)

        #   Output the predicted image
        array_buffer = arr.tobytes()
        img = Image.new("I", arr.T.shape)
        img.frombytes(array_buffer, 'raw', "I;16")
        img.save(data_output_path + '%s-prediction-%s.png' % (product[:-4], operation))

        exec_time = str(time.time() - start_time_product)
        print("Prediction finished in      : " + exec_time + "s")

        # Save original RGB image
        if not os.path.isfile(data_output_path + product):
            Image.open(data_path + 'jpg/' + product + '_RGB.jpeg').save(data_output_path + product)

    exec_time = str(time.time() - start_time_dataset)
    print("Dataset evaluated in: " + exec_time + "s")
    print("That is " + str(float(exec_time) / np.size(products)) + "s per image")
