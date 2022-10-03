#!/usr/bin/python3
import matplotlib
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import cv2
import numpy as np

# use a different backend for matplotlib since the default one causes problems with cv2
# On debian install "python3-pil python3-pil.imagetk" if this causes problems
matplotlib.use("tkagg")

ver = "0.0.1"
author = "Valentin Reichenbach"
description = f"""
TODO: Insert description
"""
epilog = f"""
Author: {author}
Version: {ver}
License: GPLv3+
"""

def getXValues(img: np.ndarray):
    """Returns the average light intensity for each row in the image"""
    xValues = []
    for x in range(img.shape[1]):
        currentVal = 0
        for y in range(img.shape[0]):
            currentVal += img[y, x]
        xValues.append(currentVal/img.shape[1])
    return xValues

def getYValues(img: np.ndarray):
    """Returns the average light intensity for each column in the image"""
    yValues = []
    for y in range(img.shape[0]):
        currentVal = 0
        for x in range(img.shape[1]):
            currentVal += img[y, x]
        yValues.append(currentVal/img.shape[0])
    return yValues

def main():
    # parse arguments
    parser = argparse.ArgumentParser(
        description=description, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', "--input", type=Path, required=True,
                        help="the input file with the beam positions")
    parser.add_argument('-b', '--background', type=Path,
                        required=True, help="the file with the background image")
    parser.add_argument('-o', '--output', type=Path,
                        default=Path('output.jpg'), help="the output file")
    parser.add_argument('-p', '--position', action='store_true', default=False,
                        help="add the position of the beam to the output image")
    parser.add_argument('-v','--visualize', action='count', default=0,
                        help="show the image")
    parser.add_argument('-V', '--version', action='version', version=f'{ver}')

    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: Input file {args.input} does not exist\nExiting...")
        exit()
    if not args.background.is_file():
        print(
            f"ERROR: Background file {args.background} does not exist\nExiting...")
        exit()

    
    # read in image
    img = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    # read in background
    background = cv2.imread(str(args.background), cv2.IMREAD_GRAYSCALE)
    # subtract background
    subtractedImg = cv2.subtract(img, background)

    xValues = getXValues(subtractedImg)
    yValues = getYValues(subtractedImg)

    if args.visualize:
        plt.plot(xValues, label="x")
        plt.plot(yValues, label="y")
        plt.legend()
        plt.show()

    if args.visualize >= 2:
        cv2.imshow("Input", img)
        cv2.imshow("Background", background)
        cv2.imshow("Subtracted", subtractedImg)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
