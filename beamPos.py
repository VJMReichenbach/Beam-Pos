#!/usr/bin/python3
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


def inputFile(string: str):
    """Checks if the input file is valid"""
    inputPath = Path(string)
    path = []
    if inputPath.is_dir():
        for file in inputPath.iterdir():
            if file.is_file():
                path.append(file)
    elif inputPath.is_file():
        path.append(inputPath)
    else:
        raise argparse.ArgumentTypeError(
            f"Input file \"{inputPath}\" does not exist\nThis can be either a single file or a directory containing multiple files")
    return path


def readInputImg(paths: list):
    """Reads in the input images and returns them as a list"""
    imgs = []
    for path in paths:
        imgs.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
    return imgs


def gaussian(x, a, x0, sigma):
    """The gaussian function used for fitting"""
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def getXValues(img: np.ndarray, i: int):
    """Returns the average light intensity for each row in image i"""
    xValues = []
    for x in range(img[i].shape[1]):
        currentVal = 0
        for y in range(img[i].shape[0]):
            currentVal += img[i][y, x]
        xValues.append(currentVal/img[i].shape[1])
    return xValues


def getYValues(img: np.ndarray, i: int):
    """Returns the average light intensity for each column in image i"""
    yValues = []
    for y in range(img[i].shape[0]):
        currentVal = 0
        for x in range(img[i].shape[1]):
            currentVal += img[i][y, x]
        yValues.append(currentVal/img[i].shape[0])
    return yValues


def fitGaussian(xValues: list):
    """Fits a gaussian to the data and returns the fit curve"""
    x = np.arange(0, len(xValues), 1)
    # increased the maxfev value since the fit failed otherwise
    popt, pcov = curve_fit(gaussian, x, xValues, p0=[1, 0, 1], maxfev=100000)
    xGauss = gaussian(x, *popt)
    return xGauss


def getBeamPos(args: argparse.Namespace):
    # read in image
    img = readInputImg(args.input)
    # read in background
    background = cv2.imread(str(args.background), cv2.IMREAD_GRAYSCALE)
    # subtract background
    subtractedImg = []
    for i in range(len(img)):
        subtractedImg.append(cv2.subtract(img[i], background))

    xValuesList = []
    yValuesList = []
    for i in range(len(subtractedImg)):
        xValuesList.append(getXValues(subtractedImg, i))
        yValuesList.append(getYValues(subtractedImg, i))

    # fit gaussian
    xGaussList = []
    yGaussList = []
    for i in range(len(subtractedImg)):
        xGaussList.append(fitGaussian(xValuesList[i]))
        yGaussList.append(fitGaussian(yValuesList[i]))

    # calcualte mean of gaussians
    # TODO: means are wrong
    xMeans = []
    yMeans = []
    for i in range(len(subtractedImg)):
        print(f"calculating mean of {args.input[i]}")
        xMeans.append(np.mean(xGaussList[i]))
        yMeans.append(np.mean(yGaussList[i]))

    # print results
    print("xMean\t\t\tyMean")
    for i in range(len(subtractedImg)):
        if not args.round:
            print(f"{xMeans[i]}\t{yMeans[i]}")
        else:
            print(
                f"{round(xMeans[i], ndigits=5)}\t{round(yMeans[i], ndigits=5)}")

    # plot the data and the fit if requested
    if args.visualize:
        if len(subtractedImg) > 1 and not args.force:
            print(
                f"WARNING: You are trying to visualize more than one image. This is not supported yet. Use the -f flag to force this")
            exit()
        for i in range(len(subtractedImg)):
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(xValuesList[i])
            ax1.plot(xGaussList[i])
            ax2.plot(yValuesList[i])
            ax2.plot(yGaussList[i])
            plt.show()

        # show the images if requested
        if args.visualize >= 2:
            for i in range(len(subtractedImg)):
                cv2.imshow("Input image", img[i])
                cv2.imshow("Subtracted image", subtractedImg[i])
                cv2.imshow("Background image", background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def main():
    # parse arguments
    parser = argparse.ArgumentParser(
        description=description, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', "--input", type=inputFile,
                        help="the input file with the beam positions. This can be either a single file or a directory containing multiple files")
    parser.add_argument('-b', '--background', type=Path,
                        help="the file with the background image. This can only be a single file")
    parser.add_argument('-p', '--position', action='store_true', default=False,
                        help="add the position of the beam to the output image")
    parser.add_argument('-v', '--visualize', action='count', default=0,
                        help="show the image")
    parser.add_argument('--markersize', type=float,
                        default=1.2, help="the size of the markers. The default is 1.2")
    parser.add_argument('--round', action='store_true',
                        default=False, help="round the beam position")
    parser.add_argument('-f', '--force', action='store_true',
                        default=False, help="skip all warnings")
    parser.add_argument('--test', action='store_true', default=False,
                        help="run the test to check if the script is working")
    parser.add_argument('-V', '--version', action='version', version=f'{ver}')

    args = parser.parse_args()

    if args.test:
        print("Running test")
        args.input = [Path("testFiles/testPic.jpg")]
        args.background = Path("testFiles/testBackground.jpg")

    if args.input is None:
        print("No input file specified")
        print("Use -h to see the help")
        exit()

    if args.background is None:
        print("No background file specified")
        print("Use -h to see the help")
        exit()

    if not args.background.is_file():
        print(
            f"ERROR: Background file {args.background} does not exist\nExiting...")
        exit()

    getBeamPos(args)


if __name__ == "__main__":
    main()
