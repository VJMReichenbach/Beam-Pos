#!/usr/bin/python3
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageStat

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

def removeBackground(args):
    # Read the background image
    background = cv2.imread(str(args.background))
    img = cv2.imread(str(args.input))

    backSub = cv2.createBackgroundSubtractorMOG2()
    _ = backSub.apply(background)
    fgMask = backSub.apply(img)
    
    # write the img with background subtracted to file
    output = cv2.bitwise_and(img, img, mask=fgMask)
    cv2.imwrite(str(args.output), output)
    print(f"Output written to {args.output}")

def getXValues(img: Image):
    xValues = []
    for x in range(img.width):
        currentVal = 0
        for y in range(img.height):
            currentVal += img.getpixel((x,y))
        xValues.append(currentVal)
    return xValues

def main():
    parser = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', "--input",type=Path, required=True ,help="the input file with the beam positions")
    parser.add_argument('-b', '--background',type=Path ,required=True ,help="the file with the background image")
    parser.add_argument('-o', '--output',type=Path,default=Path('output.jpg') ,help="the output file")
    parser.add_argument('-p', '--position', action='store_true',default=False ,help="add the position of the beam to the output image")
    parser.add_argument('--visualize',action='store_true',help="show the image")
    parser.add_argument('-V', '--version', action='version', version=f'{ver}')

    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: Input file {args.input} does not exist\nExiting...")
        exit()
    if not args.background.is_file():
        print(f"ERROR: Background file {args.background} does not exist\nExiting...")
        exit()

    removeBackground(args)
    outputImg = Image.open(args.output).convert('L')
    
    if args.visualize:
        # show all 3 images
        # cv2.imshow("Input", img)
        # cv2.imshow("Background", background)
        # plot xvales
        xValues = getXValues(outputImg)
        plt.plot(xValues)
        plt.show()


if __name__ == "__main__":
    main()