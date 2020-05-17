#!/bin/env python

import cv2 as cv
import numpy as np
import glob
import os
import sys


def sort_contours(cnts):
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boxes) = zip(*sorted(zip(cnts, boxes), key=lambda b: b[1][1]))

    # return the list of sorted contours and bounding boxes
    return (cnts, boxes)


def box_extraction(img_for_box_extraction_path, cropped_dir_path):
    img = cv.imread(img_for_box_extraction_path, 0)  # Read the image

    (thresh, orig) = cv.threshold(img, 100, 255,
                                  cv.THRESH_BINARY)  # Thresholding the image

    # img = cv.blur(img, (3, 3))
    img_bin = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 11, 0)
    img_bin = 255-img_bin  # Invert the image

    # cv.imwrite("Image_bin.jpg", img_bin)

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//20

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv.getStructuringElement(
        cv.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv.dilate(img_temp1, verticle_kernel, iterations=3)
    # cv.imwrite("verticle_lines.jpg", verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv.dilate(img_temp2, hori_kernel, iterations=3)
    # cv.imwrite("horizontal_lines.jpg", horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv.addWeighted(
        verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv.threshold(
        img_final_bin, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    # cv.imwrite("img_final_bin.jpg", img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv.findContours(
        img_final_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours)

    idx = 0
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv.boundingRect(c)

        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (w > 100 and w < 200) and (h > 100 and h < 200):
            idx += 1
            new_img = img[y:y+h, x:x+w]
            resized = cv.resize(new_img, (32, 32), interpolation=cv.INTER_AREA)
            ret, resized = cv.threshold(resized, 180, 255, cv.THRESH_BINARY)
            cv.imwrite(cropped_dir_path+str(idx) + '.png', resized)

    print(idx)

# box_extraction("./1/mr.jpg", "./crop/")


def main():
    folder = "./photos"
    if len(sys.argv) == 2:
        folder = sys.argv[1]

    folder = os.path.abspath(folder)

    out = os.path.join(os.path.dirname(folder), "crop")
    try:
        os.mkdir(out)
    except FileExistsError:
        pass

    print("out", out)

    folders = os.listdir(folder)
    for idx, name in enumerate(folders):
        print(idx, name)
        process_folder(folder, name, out)


def process_folder(folder: str, name: str, out: str):
    files = glob.glob(os.path.join(folder, name)+"/*.*g")
    print(files)
    for filename in files:
        one = os.path.join(out, name, filename)
        try:
            os.mkdir(one)
        except FileExistsError:
            pass
        print("extract", os.path.splitext(one)[0])
        # box_extraction(os.path.join(folder, name, filename), out)


if __name__ == "__main__":
    sys.exit(main())
