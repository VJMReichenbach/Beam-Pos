#!/usr/bin/python3
from multiprocessing.connection import wait
import matplotlib
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import curve_fit

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


def gaussian(x, a, x0, sigma):
    """The gaussian function used for fitting"""
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


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
    parser.add_argument('-v', '--visualize', action='count', default=0,
                        help="show the image")
    parser.add_argument('--markersize', type=float,
                        default=1.2, help="the size of the markers")
    parser.add_argument('-V', '--version', action='version', version=f'{ver}')

    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: Input file \"{args.input}\" does not exist\nExiting...")
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

    # fit gaussian
    x = np.arange(0, len(xValues), 1)
    popt, pcov = curve_fit(gaussian, x, xValues, p0=[1, 0, 1])
    xGauss = gaussian(x, *popt)
    y = np.arange(0, len(yValues), 1)
    popt, pcov = curve_fit(gaussian, y, yValues, p0=[1, 0, 1])
    yGauss = gaussian(y, *popt)

    # calcualte mean of gaussians
    xMean = np.sum(xGauss*x)/np.sum(xGauss)
    yMean = np.sum(yGauss*y)/np.sum(yGauss)
    print(f"X: {xMean}, Y: {yMean}")

    # plot the data and the fit if requested
    if args.visualize:
        fig, ax = plt.subplots(2)
        ax[0].plot(xValues, 'o', label="x", markersize=args.markersize, color='red')
        ax[1].plot(yValues, 'o', label="y", markersize=args.markersize, color='blue')
        ax[0].plot(xGauss, label="xGauss", color='orange')
        ax[1].plot(yGauss, label="yGauss", color='purple')
        fig.legend()
        plt.show()

    # show the images if requested
    if args.visualize >= 2:
        cv2.imshow("Input", img)
        cv2.imshow("Background", background)
        cv2.imshow("Subtracted", subtractedImg)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
