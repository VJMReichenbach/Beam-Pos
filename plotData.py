#!/usr/bin/python3
import matplotlib
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import csv
import numpy as np
from sklearn.linear_model import LinearRegression

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

    parser.add_argument("input", type=Path, help="a csv file containing the data")  

    args = parser.parse_args()

    # check if the input file is valid
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist!")
        exit(1)
    if not args.input.suffix == ".csv":
        print(f"Error: Input file {args.input} is not a csv file!")
        exit(1)

    data = readFromInputFile(args.input)

    # 2 subplots, one for the x values and one for the y values
    # current is x axis
    # xMean and yMean are y axis
    # xStd and yStd are error bars
    current = []
    xMean = []
    yMean = []
    xStd = []
    yStd = []


    # read the data
    for row in data:
        current.append(float(row[0]))
        xMean.append(float(row[1]))
        yMean.append(float(row[2]))
        xStd.append(float(row[3]))
        yStd.append(float(row[4]))

    # linear regression for x and y
    xModel = LinearRegression().fit(np.array(current).reshape(-1, 1), np.array(xMean).reshape(-1, 1))
    yModel = LinearRegression().fit(np.array(current).reshape(-1, 1), np.array(yMean).reshape(-1, 1))
    rSqX = xModel.score(np.array(current).reshape(-1, 1), np.array(xMean).reshape(-1, 1))
    rSqY = yModel.score(np.array(current).reshape(-1, 1), np.array(yMean).reshape(-1, 1))
    
    # generate functions for x(I) and y(I)
    x_I = f"x(I) = {xModel.coef_[0][0]:.2f} * I + {xModel.intercept_[0]:.2f}"
    y_I = f"y(I) = {yModel.coef_[0][0]:.2f} * I + {yModel.intercept_[0]:.2f}"

    # invert the functions
    I_x = f"I(x) = {1/xModel.coef_[0][0]:.2f} * x + {xModel.intercept_[0]/xModel.coef_[0][0]:.2f}"
    I_y = f"I(y) = {1/yModel.coef_[0][0]:.2f} * y + {yModel.intercept_[0]/yModel.coef_[0][0]:.2f}"
    
    print("\nLinear regression:")
    print(f"R^2 x: {rSqX:.2f}")
    print(f"R^2 y: {rSqY:.2f}")
    print(x_I)
    print(y_I)
    print(I_x)
    print(I_y)

    # plot the data
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # figure name is input file name without extension
    fig.suptitle(args.input.stem)
    ax1.errorbar(current, xMean, xerr=0, yerr=xStd, fmt="o")
    ax1.set_title("x")
    ax1.set_ylabel("Position [px]")
    ax1.set_xlabel("Current [A]")
    ax2.errorbar(current, yMean, xerr=0, yerr=yStd, fmt="o")
    ax2.set_title("y")
    ax2.set_ylabel("Position [px]")
    ax2.set_xlabel("Current [A]")

    # plot the linear regression
    ax1.plot(current, xModel.predict(np.array(current).reshape(-1, 1)), label=x_I)
    ax2.plot(current, yModel.predict(np.array(current).reshape(-1, 1)), label=y_I)

    # add legend
    ax1.legend()
    ax2.legend()
    plt.show()

if __name__ == "__main__":
    main()
