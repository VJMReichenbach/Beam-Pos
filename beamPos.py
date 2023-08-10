#!/usr/bin/python3
import argparse
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import csv


ver = "0.0.1"
author = "Valentin Reichenbach"
description = f"""
This program takes a folder (name of folder will be used as the steerer name) as an input that contains multiple subfolders. Each subfolder must contain a "current.txt" file that contains only the mesured current and a set of one or more images of the corresponding target.

The program will first find the position of the beam in each image and then calculate the average x and y position for each current. 
"""
epilog = f"""
Author: {author}
Version: {ver}
License: GPLv3+
"""

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

def readCurrent(subfolder: Path) -> float:
    """Reads the current from the current.txt file in a given subfolder and returns it as a float. Returns None if it cant find current.txt"""
    for file in subfolder.iterdir():
        if file.is_file():
            if file.name == "current.txt":
                with open(file, "r") as f:
                    return float(f.read())
    return None

def readInputFolder(folder: Path, verbosity: int):
    """Takes a folder and returns a dictionary with the current as key and a list of images as value"""
    if verbosity > 0:
        print(f"Reading input folder {folder}...")
    data = {}
    for subfolder in folder.iterdir():
        if subfolder.is_dir():
            current = readCurrent(subfolder)  
            if current is not None:
                data[current] = readInputImg(subfolder, verbosity)
            else:
                print(f"ERROR: No current.txt file found in {subfolder}")
    return data

def readInputImg(subfolder: Path, verbosity: int) -> list:
    """Takes a folder and returns a list of all images in the folder in grayscale"""
    imgs = []
    for img in subfolder.iterdir():
        if img.is_file():
            if img.suffix in [".jpg", ".png", ".bmp"]:
                if verbosity > 1:
                    print(f"Reading {img}")
                imgs.append(cv2.imread(str(img), cv2.IMREAD_GRAYSCALE))
    return imgs

def writeData():
    """Writes the data to a csv file"""
    pass

def getBeamPos(imagesWithCurrent: dict, plot: bool, verbosity: int, figureName: str):
    """gets the x and y position of the beam in an image, by finding the mean of the gaussian fit of the x and y values.
    Returns a dict with current as key and the x and y position as an array as the value.
    Example: {0.0: [x, y], 0.1: [x, y], ...}
    If plot is True, it will plot the x and y values in 2 subplots."""
    beamPos = {}
    std = {}
    for current in imagesWithCurrent:
        beamPos[current] = []
        for i in range(len(imagesWithCurrent[current])):
            if verbosity > 1:
                print(f"Finding beam position for {figureName} with {current}A")
            # get the x and y values and fit a gaussian to them
            xValues = getXValues(imagesWithCurrent[current], i)
            yValues = getYValues(imagesWithCurrent[current], i)
            xGauss = fitGaussian(xValues)
            yGauss = fitGaussian(yValues)
            # calculate the mean of the gaussian fit
            # idk why, but only my old method works  ¯\_(ツ)_/¯
            # xMean = np.mean(xGauss)
            # yMean = np.mean(yGauss)
            xLen = np.arange(0, len(xGauss), 1)
            yLen = np.arange(0, len(yGauss), 1)
            xMean = np.sum(xGauss*xLen)/np.sum(xGauss)
            yMean = np.sum(yGauss*yLen)/np.sum(yGauss)
            # calculate the standard deviation of the gaussian fit 
            xStd = np.std(xGauss)
            yStd = np.std(yGauss)

            beamPos[current].append([xMean, yMean])
            std[current] = [xStd, yStd]

            if plot:
                # plot the x and y values in 2 subplots
                # title the whole plot after the folder name
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.suptitle(f"{figureName} with {current}A")
                ax1.plot(xValues, label="x values")
                ax1.plot(xGauss, label="x gaussian fit")
                ax2.plot(yValues, label="y values")
                ax2.plot(yGauss, label="y gaussian fit")

                ax1.set_title("X values")
                ax2.set_title("Y values")

                ax1.set_xlabel("Pixel")
                ax2.set_xlabel("Pixel")
                ax1.set_ylabel("Light intensity")
                ax2.set_ylabel("Light intensity") 

                ax1.legend()
                ax2.legend()
                plt.show()
    return beamPos, std

def main(inputFolder: Path, background: Path, output: Path, show: bool, plot_gauss: bool, round: bool, force: bool, verbosity: int):
    # figure name
    figureName = inputFolder.name
    # get a dict of the images as arrays with the current as key
    imagesWithCurrent = readInputFolder(inputFolder, verbosity)
    # read the background images
    background = cv2.imread(str(background), cv2.IMREAD_GRAYSCALE)
    # subtract the background from the imagesWithCurrent
    for current in imagesWithCurrent:
        for i in range(len(imagesWithCurrent[current])):
            imagesWithCurrent[current][i] = cv2.subtract(
                imagesWithCurrent[current][i], background)

    # show the subtracted images
    if show:
        for current in imagesWithCurrent:
            for img in imagesWithCurrent[current]:
                cv2.imshow(f"{figureName} with {current}A", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # get the beam position
    beamPos, std = getBeamPos(imagesWithCurrent, plot_gauss, verbosity, figureName)
    avgBeamPos = {}
    for current in beamPos:
        avgBeamPos[current] = np.mean(beamPos[current], axis=0)

    if verbosity > 2:
        print(f"beam position: {beamPos}")
        print(f"standard deviation: {std}")
        print(f"average beam position: {avgBeamPos}")

    with open(output, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["current", "xMean", "yMean", "xStd", "yStd"])
        for current in avgBeamPos:
            writer.writerow([current, avgBeamPos[current][0], avgBeamPos[current][1], std[current][0], std[current][1]])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)  

    # arguments
    parser.add_argument("-i", "--input", type=Path, help="Folder containing the subfolders with the images and the current.txt file")
    parser.add_argument("-b", "--background", help="Background image to be used for the subtraction")
    parser.add_argument("-o", "--output", type=Path, default=None, help=".csv file to save the results to. If not specified, the results will be stored in a file with the same name as the input folder")
    parser.add_argument('--show', action='store_true', default=False,help="show the image")
    parser.add_argument('--plot-gauss', action='store_true', default=False, help="plot the gaussian fit for each image")
    parser.add_argument('--round', action='store_true',
                        default=False, help="round the beam position")
    parser.add_argument('-f', '--force', action='store_true',
                        default=False, help="skip all warnings")
    parser.add_argument('--test', action='store_true', default=False,
                        help="run the test to check if the script is working")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="increase output verbosity")
    parser.add_argument('-V', '--version', action='version', version=f'{ver}')

    args = parser.parse_args()

    # check args
    if args.test:
        print("Running test")
        args.input = Path("testFiles/I0SH03/")
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

    if args.output is None:
        # if no output file is specified, use the input folder name
        outputFile = args.input / f"{args.input.name}.csv"
        print(f"No output file specified, using {outputFile}")
    else:
        outputFile = Path(args.output.stem + ".csv")

    if outputFile.is_file():
        if not args.force:
            print('ERROR: Output file already exists')
            print('Use --force to overwrite the file')
            print('Exiting...')
            exit()

    if args.verbose > 0:
        print(f"Args:\n{args}\n")

    try:
        main(args.input, args.background, outputFile, args.show, args.plot_gauss, args.round, args.force, args.verbose)
    except KeyboardInterrupt:
        print("Exiting...")
        exit()





