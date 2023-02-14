#!/usr/bin/python3
import matplotlib
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import csv

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

def readFromInputFile(path: Path) -> list:
    """Reads the input file and returns the data as a list"""
    print(f"Reading input file {path}")
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader) # Skip header
        data = []
        for row in reader:
            data.append(row)
    return data

def main():
    parser = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", help="The input directory. It's name will be used as the figure name. Inside each of these folders has to be one or more .csv files. These names will be used as the value for the current", type=Path)

    args = parser.parse_args()
    print(args)

    # check if input folder exists and contains one or more .csv files
    if not args.input.is_dir():
        raise argparse.ArgumentTypeError(f"Input directory \"{args.input}\" does not exist")
    if not any(file.suffix == ".csv" for file in args.input.iterdir()):
        raise argparse.ArgumentTypeError(f"Input directory \"{args.input}\" does not contain any .csv files")
    
    # 2 subplots, one for the x values and one for the y values
    # current is x axis
    # xMean and yMean are y axis
    current = []
    xMean = []
    yMean = []

    # read input file
    data = []
    filenames = []
    for file in args.input.iterdir():
        if file.suffix == ".csv":
            data.append(readFromInputFile(file))
            filenames.append(file.name.split(".csv")[0]) # get the current from the file name

    # get the xMean and yMean values
    for i in range(len(data)):
        for j in range(len(data[i])):
            xMean.append(float(data[i][j][0]))
            yMean.append(float(data[i][j][1]))
            current.append(float(filenames[i]))
    print(xMean)
    print(yMean)
    print(current)

    # plot the data
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(args.input.name)
    ax1.plot(current, xMean, "o")
    ax1.set_title("xMean")
    ax1.set_xlabel("Current [A]")
    ax1.set_ylabel("xMean [px]")
    ax2.plot(current, yMean, "o")
    ax2.set_title("yMean")
    ax2.set_xlabel("Current [A]")
    ax2.set_ylabel("yMean [px]")
    plt.show()


if __name__ == "__main__":
    main()