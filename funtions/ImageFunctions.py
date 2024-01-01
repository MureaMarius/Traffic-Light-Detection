import re

from matplotlib import image as mtpimg
from skimage.exposure import exposure

from functions import Utility
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.ndimage.filters as filters
from PIL import Image
from skimage.feature import hog
import cv2
import numpy as np
import csv
import os
import sys


def brightness(image):
    gray = image
    average_color_per_row = np.average(gray, axis=0)
    average_color = np.average(average_color_per_row, axis=0)
    print(average_color)

    return average_color


def getPixelsList(image):
    im = Image.open(image)
    pixels = list(im.getdata())

    return pixels


def getMaximValueFromFourPoints(pathImage, xMin, xMax, yMin, yMax):
    image = Image.open(pathImage)
    image_data = np.asarray(image)

    maxValue = [-1, -1, -1]
    for i in range(yMin, yMax):
        for j in range(xMin, xMax):
            if (image_data[i][j] > maxValue).all():
                maxValue = image_data[i][j]

    return maxValue


def createBlankImage(width, height, rgb_color=(0, 0, 0)):
    image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color

    return image


def convertAnImageToGrayscale(imagePath):
    grayImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    return grayImage


def convertAnRGBImageToBinaryImage(imagePath):
    image = cv2.imread(imagePath, 2)

    ret, bw_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    bw = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # cv2.imshow("Result from RGB", bw_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return bw_image


def convertAnGrayScaleImageToBinaryImage(imagePath):
    imageInGray = cv2.imread(imagePath)
    (ret, grayToBinary) = cv2.threshold(imageInGray, 127, 256, cv2.THRESH_BINARY)

    cv2.imshow("Result from GrayScale", grayToBinary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getPixelWithMaximValue(imagePath):
    image = Image.open(imagePath)
    image_data = np.asarray(image)

    maximValue = [0, 0, 0]
    for i in range(len(image_data)):
        for j in range(len(image_data[0])):
            if (image_data[i][j] > maximValue).all():
                maximValue = image_data[i][j]

    return maximValue


def valueOfPixelsFromImage(imagePath):
    image = Image.open(imagePath)
    image_data = np.asarray(image)

    firstValue = 0
    secondValue = 0
    thirdValue = 0
    numberOfValidPixelsForAverage = 0

    for i in range(len(image_data)):
        for j in range(len(image_data[0])):
            if (image_data[i][j] > [100, 100, 100]).all():
                firstValue += image_data[i][j][0]
                secondValue += image_data[i][j][1]
                thirdValue += image_data[i][j][2]
                numberOfValidPixelsForAverage += 1

    if numberOfValidPixelsForAverage > 0:
        firstValue = int(firstValue / numberOfValidPixelsForAverage)
        secondValue = int(secondValue / numberOfValidPixelsForAverage)
        thirdValue = int(thirdValue / numberOfValidPixelsForAverage)
    elif numberOfValidPixelsForAverage == 0:
        maximValue = getPixelWithMaximValue(imagePath)
        firstValue = maximValue[0]
        secondValue = maximValue[1]
        thirdValue = maximValue[2]

    return [firstValue, secondValue, thirdValue]


def createAndPlotHistogram(imagePath):
    image = cv2.imread(imagePath, 0)

    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    plt.plot(histogram)
    plt.show()


def convertImageRGBtoHSV(imagePath):
    image = cv2.imread(imagePath)
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.imshow('Original image', image)
    cv2.imshow('HSV image', hsvImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return hsvImage


def plot_images_conversions(imagePath):
    rows = 2
    columns = 2

    fig = plt.figure(figsize=(10, 7))
    originalImage = cv2.imread(imagePath)
    imageForBinary = cv2.imread(imagePath, 2)

    ret, bw_image = cv2.threshold(imageForBinary, 127, 255, cv2.THRESH_BINARY)
    grayImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    hsvImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

    fig.add_subplot(rows, columns, 1)
    plt.imshow(originalImage)
    plt.axis('off')
    plt.title("Original image")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(bw_image)
    plt.axis('off')
    plt.title("Binary image")

    fig.add_subplot(rows, columns, 3)
    plt.imshow(grayImage)
    plt.axis('off')
    plt.title("Grayscale image")

    fig.add_subplot(rows, columns, 4)
    plt.imshow(hsvImage)
    plt.axis('off')
    plt.title("HSV image")

    plt.show()


def convertRGBtoHSVValues(red, green, blue):
    global hue
    red = red / 255.0
    green = green / 255.0
    blue = blue / 255.0

    maxim = max(red, max(green, blue))
    minim = min(red, min(green, blue))
    diffBetweenMaxMin = maxim - minim

    if maxim == minim:
        hue = 0
    elif maxim == red:
        hue = (60 * ((green - blue) / diffBetweenMaxMin) + 360) % 360
    elif maxim == green:
        hue = (60 * ((blue - red) / diffBetweenMaxMin) + 120) % 360
    elif maxim == blue:
        hue = (60 * ((red - green) / diffBetweenMaxMin) + 240) % 360

    if maxim == 0:
        saturation = 0
    else:
        saturation = (diffBetweenMaxMin / maxim) * 100

    value = maxim * 100
    return hue, saturation, value


def getHistogramArrayOfAnImage(imagePath):
    image = mtpimg.imread(imagePath)
    imageArray = image[:, :, 0]

    histogram = [0] * 256
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            positionInHistogram = imageArray[i, j]
            histogram[positionInHistogram] = histogram[positionInHistogram] + 1

    return histogram


def getRedOrGreenParts(imagePath):
    image = cv2.imread(imagePath)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (110, 255, 25), (70, 255, 255))
    iMask = mask > 0
    green = np.zeros_like(image, np.uint8)
    green[iMask] = image[iMask]

    cv2.imshow("Res", green)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getPixelFromColorZone(listOfTrafficLights):
    labelsFromTrafficLights = []
    listOfHSV = [(0, 0, 0)]
    listToBeReturned = [(0, 0, 0, 0)]

    for objectFromList in listOfTrafficLights:
        label = objectFromList.label
        labelsFromTrafficLights.append(label)

    imageNumber = 0

    for label in labelsFromTrafficLights:
        color = label
        imagePath = f'C:\\Users\\Marius\Desktop\\LicentaProiect\\trafficLightImagesRedGreen\\trafficLightImage_{imageNumber:04}.jpg'

        try:
            originalImage = Image.open(imagePath)
            image_data = np.asarray(originalImage)
        except:
            break

        width, height = originalImage.size
        imageNumber = imageNumber + 1

        firstValue = 0
        secondValue = 0
        thirdValue = 0
        numberOfValidPixelsForAverage = 0

        if color == "Red":
            for i in range(0, int(height / 2)):
                for j in range(0, width):
                    if (image_data[i][j] > [20, 20, 20]).all():
                        firstValue += image_data[i][j][0]
                        secondValue += image_data[i][j][1]
                        thirdValue += image_data[i][j][2]
                        numberOfValidPixelsForAverage += 1

            if numberOfValidPixelsForAverage > 0:
                firstValue = int(firstValue / numberOfValidPixelsForAverage)
                secondValue = int(secondValue / numberOfValidPixelsForAverage)
                thirdValue = int(thirdValue / numberOfValidPixelsForAverage)
            elif numberOfValidPixelsForAverage == 0:
                maximValue = getPixelWithMaximValue(imagePath)
                firstValue = maximValue[0]
                secondValue = maximValue[1]
                thirdValue = maximValue[2]

            value = [firstValue, secondValue, thirdValue, 0]
            h, s, v = convertRGBtoHSVValues(firstValue, secondValue, thirdValue)
            listOfHSV.append((int(h), int(s), int(v)))

            listToBeReturned.append(value)

        if color == "Green":
            for i in range(int(height / 2 + 1), height):
                for j in range(0, width):
                    if (image_data[i][j] > [20, 20, 20]).all():
                        firstValue += image_data[i][j][0]
                        secondValue += image_data[i][j][1]
                        thirdValue += image_data[i][j][2]
                        numberOfValidPixelsForAverage += 1

            if numberOfValidPixelsForAverage > 0:
                firstValue = int(firstValue / numberOfValidPixelsForAverage)
                secondValue = int(secondValue / numberOfValidPixelsForAverage)
                thirdValue = int(thirdValue / numberOfValidPixelsForAverage)
            elif numberOfValidPixelsForAverage == 0:
                maximValue = getPixelWithMaximValue(imagePath)
                firstValue = maximValue[0]
                secondValue = maximValue[1]
                thirdValue = maximValue[2]

            value = [firstValue, secondValue, thirdValue, 1]
            h, s, v = convertRGBtoHSVValues(firstValue, secondValue, thirdValue)
            listOfHSV.append((int(h), int(s), int(v)))

            listToBeReturned.append(value)

    listToBeReturned.remove((0, 0, 0, 0))
    listOfHSV.remove((0, 0, 0))

    return listToBeReturned, listOfHSV


def getANewImageWithTrafficLights(listOfTrafficLights):
    imagePath = listOfTrafficLights[0].path
    originalImage = Image.open(imagePath)

    width, height = originalImage.size
    white = (255, 255, 255)
    newImage = createBlankImage(width, height, rgb_color=white)

    for obj in listOfTrafficLights:
        xMax = int(obj.x_max)
        xMin = int(obj.x_min)
        yMax = int(obj.y_max)
        yMin = int(obj.y_min)

        for i in range(0, width):
            for j in range(0, height):
                if (xMin <= i <= xMax) and (yMin <= j <= yMax):
                    current_color = originalImage.getpixel((i, j))
                    newImage.put((i, j), current_color)

    return newImage


def RGBChannels(imagePath):
    image = mtpimg.imread(imagePath)

    blueChannel = image[:, :, 0]
    greenChannel = image[:, :, 1]
    redChannel = image[:, :, 2]

    return blueChannel, greenChannel, redChannel


def HSVChannels(imagePath):
    image = cv2.imread(imagePath)

    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsvImage[:, :, 0]
    saturation = hsvImage[:, :, 1]
    value = hsvImage[:, :, 2]

    return hue, saturation, value


def averageRedGreenChannelsOnTopAndBottom(imagePath):
    originalImage = Image.open(imagePath)
    image_data = np.asarray(originalImage)
    width, height = originalImage.size

    sumRedChannel = 0
    sumGreenChannel = 0
    numberOfValidPixelsTop = 0
    numberOfValidPixelsBottom = 0

    for i in range(0, int(height / 2)):
        for j in range(0, width):
            sumGreenChannel += image_data[i][j][1]
            sumRedChannel += image_data[i][j][2]
            numberOfValidPixelsTop += 1

    topAverageForRed = sumRedChannel / int(numberOfValidPixelsTop)
    topAverageForGreen = sumGreenChannel / int(numberOfValidPixelsTop)

    sumRedChannel = 0
    sumGreenChannel = 0
    for i in range(int(height / 2 + 1), height):
        for j in range(0, width):
            sumGreenChannel += image_data[i][j][1]
            sumRedChannel += image_data[i][j][2]
            numberOfValidPixelsBottom += 1

    bottomAverageForRed = sumRedChannel / int(numberOfValidPixelsBottom)
    bottomAverageForGreen = sumGreenChannel / int(numberOfValidPixelsBottom)

    return int(topAverageForGreen), int(topAverageForRed), int(bottomAverageForGreen), int(bottomAverageForRed)


def createHistogramForColorImage(imagePath):
    image = cv2.imread(imagePath, -1)
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    colorRGB = ('b', 'g', 'r')
    colorHSV = ('h', 's', 'v')

    blueChannel = []
    greenChannel = []
    redChannel = []

    hueChannel = []
    saturationChannel = []
    valueChannel = []

    finalArrayRGB = []
    for channel, col in enumerate(colorRGB):
        histogram = cv2.calcHist([image], [channel], None, [16], [0, 256])
        listOfValues = histogram.ravel().tolist()
        for x in listOfValues:
            finalArrayRGB.append(int(x))

    finalArrayHSV = []
    for channel, col in enumerate(colorHSV):
        histogram = cv2.calcHist([hsvImage], [channel], None, [16], [0, 256])
        listOfValues = histogram.ravel().tolist()
        for x in listOfValues:
            finalArrayHSV.append(int(x))

    for i in range(0, 16):
        blueChannel.append(finalArrayRGB[i])
        hueChannel.append(finalArrayHSV[i])

    for i in range(16, 32):
        greenChannel.append(finalArrayRGB[i])
        saturationChannel.append(finalArrayHSV[i])

    for i in range(32, len(finalArrayRGB)):
        redChannel.append(finalArrayRGB[i])
        valueChannel.append(finalArrayHSV[i])

    return blueChannel, greenChannel, redChannel, hueChannel, saturationChannel, valueChannel


def highPassFilter(image):
    highFilter = np.array([[-1 / 9, -1 / 9, -1 / 9], [-1 / 9, 8 / 9, -1 / 9], [-1 / 9, -1 / 9, -1 / 9]])

    return ndimage.convolve(image[:, :, 1], highFilter)


def getTrafficLights_MethodVersion_1(imagePath):
    image = cv2.imread(imagePath) / 255

    filterForImage = highPassFilter(image)

    # cv2.imshow("Filter", filterForImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    imageMaximum = filters.maximum_filter(filterForImage, 35)
    plt.figure(figsize=(10, 5))
    imageCoordinates = image.copy()
    imageCoordinates[imageMaximum > image[:, :, 1]] = [0, 1, 0]

    cv2.imshow("Result", imageCoordinates)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createHOGHistogram(imagePath):
    image = Image.open(imagePath)

    (fd, hogImage) = hog(image, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(1, 1), visualize=True,
                         multichannel=True)

    return fd, hogImage

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Original image')

    #Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hogImage, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    """


def createCSVWithPosition(imagePath, checker, csvNumber, typeOfTrafficLight):
    global j, i

    valueFromPatchCell = []
    row_list = [[]]

    valueFromPatchCell.append('Position')
    for i in range(0, 25):
        valueFromPatchCell.append(i)

    row_list[0] = valueFromPatchCell

    firstVersionOfImage = mtpimg.imread(imagePath)
    secondVersionOfImage = Image.open(imagePath)
    image_data = np.asarray(secondVersionOfImage)

    height = firstVersionOfImage.shape[0]
    width = firstVersionOfImage.shape[1]
    lineListPosition = []

    limitImage = True

    x = 0
    y = 0

    line = 0
    while limitImage:
        linesNumber = 0
        lineListPosition.append(line)
        for i in range(x, height):
            if linesNumber == 5:
                break

            countPositions = 0
            for j in range(y, width):
                if countPositions == 5:
                    break

                lineListPosition.append([i, j])

                countPositions += 1

            linesNumber += 1

        row_list.append(lineListPosition)
        lineListPosition = []

        if j == width - 1:
            x += 5
            y = 0
        else:
            y += 5

        if i == height - 1 and j == width - 1:
            break

        line += 1

    if typeOfTrafficLight == "Green":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                    'w',
                    newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\TestCSVs\\TestPosition.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 2:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition_Green.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
    elif typeOfTrafficLight == "Red":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\Desktop\\LicentaProiect\\RedTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                    'w',
                    newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestPosition.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 2:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition_Red.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
    elif typeOfTrafficLight == "Yellow":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\Desktop\\LicentaProiect\\YellowTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                    'w',
                    newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\TestPosition.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 2:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition_Yellow.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
    elif typeOfTrafficLight == "All":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\Desktop\\LicentaProiect\\AllTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                    'w',
                    newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)


def createCSVWithPatches_Short(imagePath, csvNumber):
    global j, i

    valueFromPatchCell = []
    row_list = [[]]

    for i in range(0, 75):
        valueFromPatchCell.append(i)

    row_list[0] = valueFromPatchCell

    firstVersionOfImage = mtpimg.imread(imagePath)
    secondVersionOfImage = Image.open(imagePath)
    image_data = np.asarray(secondVersionOfImage)

    height = firstVersionOfImage.shape[0]
    width = firstVersionOfImage.shape[1]
    lineListPosition = []

    limitImage = True

    x = 0
    y = 0
    while limitImage:
        linesNumber = 0
        for i in range(x, height):
            if linesNumber == 5:
                break

            countPositions = 0
            for j in range(y, width):
                if countPositions == 5:
                    break

                r_channel = image_data[i][j][0]
                g_channel = image_data[i][j][1]
                b_channel = image_data[i][j][2]
                h_channel, s_channel, v_channel = convertRGBtoHSVValues(r_channel, g_channel, b_channel)

                lineListPosition.append(int(h_channel))
                lineListPosition.append(int(s_channel))
                lineListPosition.append(int(v_channel))

                countPositions += 1

            linesNumber += 1

        row_list.append(lineListPosition)
        lineListPosition = []

        if j == width - 1:
            x += 5
            y = 0
        else:
            y += 5

        if i == height - 1 and j == width - 1:
            break

    with open(
            f"C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
            'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def createCSVWithPatches_Long(imagePath, checker, csvNumber, typeOfTrafficLight):
    global j, i

    valueFromPatchCell = []
    row_list = [[]]

    for i in range(0, 200):
        valueFromPatchCell.append(i)

    row_list[0] = valueFromPatchCell

    firstVersionOfImage = mtpimg.imread(imagePath)
    secondVersionOfImage = Image.open(imagePath)
    image_data = np.asarray(firstVersionOfImage)

    height = firstVersionOfImage.shape[0]
    width = firstVersionOfImage.shape[1]
    lineListPosition = []

    limitImage = True

    x = 0
    y = 0
    while limitImage:
        linesNumber = 0
        for i in range(x, height):
            if linesNumber == 5:
                break

            countPositions = 0
            for j in range(y, width):
                if countPositions == 5:
                    break

                r_channel = image_data[i][j][0]
                g_channel = image_data[i][j][1]
                b_channel = image_data[i][j][2]
                h_channel, s_channel, v_channel = convertRGBtoHSVValues(r_channel, g_channel, b_channel)
                brightnessPixel = sum([r_channel, g_channel, b_channel]) / 3
                averageHSV = (h_channel + s_channel + v_channel) / 3

                lineListPosition.append(r_channel)
                lineListPosition.append(g_channel)
                lineListPosition.append(b_channel)
                lineListPosition.append(h_channel)
                lineListPosition.append(s_channel)
                lineListPosition.append(v_channel)
                lineListPosition.append(brightnessPixel)
                lineListPosition.append(averageHSV)

                countPositions += 1

            linesNumber += 1

        row_list.append(lineListPosition)
        lineListPosition = []

        if j == width - 1:
            x += 5
            y = 0
        else:
            y += 5

        if i == height - 1 and j == width - 1:
            break

    if typeOfTrafficLight == "Green":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\TestCSVs\\TestPatches.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 2:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPatches_Green.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
    elif typeOfTrafficLight == "Red":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\Desktop\\LicentaProiect\\RedTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestPatches.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 2:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPatches_Red.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
    elif typeOfTrafficLight == "Yellow":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\Desktop\\LicentaProiect\\YellowTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\TestPatches.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 2:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPatches_Yellow.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
    elif typeOfTrafficLight == "All":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\Desktop\\LicentaProiect\\AllTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPatches.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)


def createCSVWithBinaryValues(imagePath, checker, csvNumber, typeOfTrafficLight):
    global j, i

    valueFromPatchCell = []
    row_list = [[]]

    for i in range(0, 25):
        valueFromPatchCell.append(i)

    row_list[0] = valueFromPatchCell

    firstVersionOfImage = mtpimg.imread(imagePath)
    secondVersionOfImage = Image.open(imagePath)
    image_data = np.asarray(secondVersionOfImage)

    height = firstVersionOfImage.shape[0]
    width = firstVersionOfImage.shape[1]
    lineListPosition = []

    limitImage = True

    x = 0
    y = 0

    while limitImage:
        linesNumber = 0
        for i in range(x, height):
            if linesNumber == 5:
                break

            countPositions = 0
            for j in range(y, width):
                if countPositions == 5:
                    break
                value = image_data[i][j]
                lineListPosition.append(value)

                countPositions += 1

            linesNumber += 1

        row_list.append(lineListPosition)
        lineListPosition = []

        if j == width - 1:
            x += 5
            y = 0
        else:
            y += 5

        if i == height - 1 and j == width - 1:
            break

    if typeOfTrafficLight == "Green":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\BinaryValuesCSV\\CSVwithBinaryValues_{csvNumber:04}.csv",
                    'w',
                    newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\TestCSVs\\TestBinaryPatches.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
    elif typeOfTrafficLight == "Red":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\BinaryValuesCSV\\CSVwithBinaryValues_{csvNumber:04}.csv",
                    'w',
                    newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestBinaryPatches.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
    elif typeOfTrafficLight == "Yellow":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\BinaryValuesCSV\\CSVwithBinaryValues_{csvNumber:04}.csv",
                    'w',
                    newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\TestBinaryPatches.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)


def createCSVOnlyForTrafficLights(imagePath, checker, csvNumber, typeOfTrafficLight):
    global j, i

    valueFromPatchCell = []
    row_list = [[]]

    for i in range(0, 25):
        valueFromPatchCell.append(i)

    row_list[0] = valueFromPatchCell

    firstVersionOfImage = mtpimg.imread(imagePath)
    secondVersionOfImage = Image.open(imagePath)
    image_data = np.asarray(secondVersionOfImage)

    height = firstVersionOfImage.shape[0]
    width = firstVersionOfImage.shape[1]
    lineListPosition = []

    limitImage = True

    x = 0
    y = 0

    while limitImage:
        linesNumber = 0
        for i in range(x, height):
            if linesNumber == 5:
                break

            countPositions = 0
            for j in range(y, width):
                if countPositions == 5:
                    break
                value = image_data[i][j]
                if value[0] == 255 or value[1] == 255 or value[2] == 255:
                    lineListPosition.append(255)
                else:
                    lineListPosition.append(0)

                countPositions += 1

            linesNumber += 1

        row_list.append(lineListPosition)
        lineListPosition = []

        if j == width - 1:
            x += 5
            y = 0
        else:
            y += 5

        if i == height - 1 and j == width - 1:
            break

    if typeOfTrafficLight == "Green":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\OnlyTrafficLightsCSV\\CSVwithOnlyTrafficLights_{csvNumber:04}.csv",
                    'w',
                    newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 2:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights_Green.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
    elif typeOfTrafficLight == "Red":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\OnlyTrafficLightsCSV\\CSVwithOnlyTrafficLights_{csvNumber:04}.csv",
                    'w',
                    newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 2:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights_Red.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
    elif typeOfTrafficLight == "Yellow":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\OnlyTrafficLightsCSV\\CSVwithOnlyTrafficLights_{csvNumber:04}.csv",
                    'w',
                    newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 2:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights_Yellow.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
    elif typeOfTrafficLight == "All":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\OnlyTrafficLightsCSV\\CSVwithOnlyTrafficLights_{csvNumber:04}.csv",
                    'w',
                    newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights.csv",
                    'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)


def resize(imagePath):
    size = 1024, 768
    im = Image.open(imagePath)
    im_resized = im.resize(size, Image.ANTIALIAS)
    im_resized.save("Sample.png", "PNG")


def detectGreenFromTrafficLights(limitImages):
    directory = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\Images'
    imageNumber = 0

    countImages = 0
    for filename in os.listdir(directory):
        if countImages == limitImages:
            break

        if filename.endswith(".png"):
            pathImage = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\Images\\' + filename
            originalImage = cv2.imread(pathImage)
            hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

            lower_green = np.array([60, 100, 100])
            upper_green = np.array([100, 255, 255])
            maskGreen = cv2.inRange(hsv, lower_green, upper_green)

            # toimage(maskGreen).show()
            finalImage = Image.fromarray(maskGreen)
            finalImage.save(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\BinaryImages\\Image_{imageNumber:04}.jpg")

            imageNumber = imageNumber + 1

        countImages += 1


def detectRedFromTrafficLights(limitImages):
    global result
    directory = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\Images'
    imageNumber = 0

    countImages = 0
    for filename in os.listdir(directory):
        if countImages == limitImages:
            break

        if filename.endswith(".png"):
            pathImage = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\Images\\' + filename
            originalImage = cv2.imread(pathImage)
            hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)
            result = originalImage.copy()

            lower_red1 = np.array([0, 100, 0])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 0])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            maskRed = cv2.add(mask1, mask2)

            # lower = np.array([140, 100, 0])
            # upper = np.array([179, 255, 255])
            # mask = cv2.inRange(hsv, lower, upper)
            # result = cv2.bitwise_and(result, result, mask=mask)

            # cv2.imshow('mask', mask)
            # cv2.imshow('result', result)
            # cv2.waitKey()

            # toimage(maskRed).show()
            finalImage = Image.fromarray(maskRed)
            finalImage.save(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\BinaryImages\\Image_{imageNumber:04}.jpg")

            imageNumber = imageNumber + 1

        countImages += 1


def detectYellowFromTrafficLights(limitImages):
    global result
    directory = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\Images'
    imageNumber = 0

    countImages = 0
    for filename in os.listdir(directory):
        if countImages == limitImages:
            break

        if filename.endswith(".png"):
            pathImage = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\Images\\' + filename
            originalImage = cv2.imread(pathImage)
            hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

            lower_yellow = np.array([15, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            finalImage = Image.fromarray(maskYellow)
            finalImage.save(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\BinaryImages\\Image_{imageNumber:04}.jpg")

            imageNumber = imageNumber + 1

        countImages += 1


def detectGreenFromTrafficLights_ForTestImage(imagePath):
    originalImage = cv2.imread(imagePath)
    hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

    lower_green = np.array([60, 100, 100])
    upper_green = np.array([100, 255, 255])
    maskGreen = cv2.inRange(hsv, lower_green, upper_green)

    # toimage(maskGreen).show()
    finalImage = Image.fromarray(maskGreen)
    finalImage.save(
        f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\BinaryImages\\BinaryImageTest.jpg")


def detectRedFromTrafficLights_ForTestImage(imagePath):
    originalImage = cv2.imread(imagePath)
    hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    maskRed = cv2.add(mask1, mask2)

    # toimage(maskRed).show()
    finalImage = Image.fromarray(maskRed)
    finalImage.save(
        f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\BinaryImages\\BinaryImageTest.jpg")


def detectYellowFromTrafficLights_ForTestImage(imagePath):
    originalImage = cv2.imread(imagePath)
    hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    finalImage = Image.fromarray(maskYellow)
    finalImage.save(
        f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\BinaryImages\\BinaryImageTest.jpg")


def createImageFromClassifierImageResult(arrayOfPointsCoordinates):
    listOfPoints = arrayOfPointsCoordinates

    image = Image.new('RGB', (1280, 720), color=(0, 0, 0))
    image_array = np.array(image)

    width, height = image.size

    for w in range(0, height - 1):
        for h in range(0, width - 1):
            checkerPoints = 0
            for x in range(0, len(listOfPoints)):
                listOfPositionNumbers = re.findall(r'\d+', listOfPoints[x])
                positionX = int(listOfPositionNumbers[0])
                positionY = int(listOfPositionNumbers[1])

                if positionX == w and positionY == h:
                    checkerPoints = 1
                    break

            if checkerPoints == 1:
                image_array[w][h] = [255, 255, 255]
            else:
                image_array[w][h] = [0, 0, 0]

    cv2.imshow("Result", image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createFirstImageResultFromClassifierPredictedPoints(arrayOfPointsCoordinates):
    listOfPoints = arrayOfPointsCoordinates

    image = Image.new('RGB', (1280, 720), color=(0, 0, 0))
    image_array = np.array(image)

    for x in range(0, len(listOfPoints)):
        listOfPositionNumbers = re.findall(r'\d+', listOfPoints[x])
        positionX = int(listOfPositionNumbers[0])
        positionY = int(listOfPositionNumbers[1])

        image_array[positionX][positionY] = [255, 255, 255]

    cv2.imshow("Result", image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createImageWithTrafficLightsFromPointsCoordinates(limitImages, typeOfTrafficLight):
    global directory, pathImage, listOfTrafficLights

    directoryForGreen = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\Images'
    directoryForRed = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\Images'
    directoryForYellow = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\Images'
    directoryForAll = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\Images'

    if typeOfTrafficLight == "Green":
        directory = directoryForGreen
    elif typeOfTrafficLight == "Red":
        directory = directoryForRed
    elif typeOfTrafficLight == "Yellow":
        directory = directoryForYellow
    elif typeOfTrafficLight == "All":
        directory = directoryForAll

    imageNumber = 0
    countImages = 0
    for filename in os.listdir(directory):
        if countImages == limitImages:
            break
        if filename.endswith(".png"):
            if typeOfTrafficLight == "Green":
                pathImage = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\Images\\' + filename
                listOfTrafficLights = Utility.extractSpecificBox(pathImage, 0, "Green")
            elif typeOfTrafficLight == "Red":
                pathImage = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\Images\\' + filename
                listOfTrafficLights = Utility.extractSpecificBox(pathImage, 0, "Red")
            elif typeOfTrafficLight == "Yellow":
                pathImage = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\Images\\' + filename
                listOfTrafficLights = Utility.extractSpecificBox(pathImage, 0, "Yellow")
            elif typeOfTrafficLight == "All":
                pathImage = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\Images\\' + filename
                listOfTrafficLights = Utility.extractSpecificBox(pathImage, 0, "All")

            image = Image.new('RGB', (1280, 720), color=(0, 0, 0))
            image_array = np.array(image)

            for x in range(len(image_array)):
                for y in range(len(image_array[0])):
                    for z in range(0, len(listOfTrafficLights)):
                        trafficLight = listOfTrafficLights[z]

                        if typeOfTrafficLight == "Green":
                            if trafficLight.label != "Green" and trafficLight.label != "GreenRight" and \
                                    trafficLight.label != "GreenLeft":
                                continue
                            else:
                                x_max = trafficLight.x_max
                                x_min = trafficLight.x_min
                                y_max = trafficLight.y_max
                                y_min = trafficLight.y_min
                                if x_min < y < x_max and y_min < x < y_max:
                                    image_array[x][y] = [255, 255, 255]
                        elif typeOfTrafficLight == "Red":
                            if trafficLight.label != "Red" and trafficLight.label != "RedRight" and \
                                    trafficLight.label != "RedLeft":
                                continue
                            else:
                                x_max = trafficLight.x_max
                                x_min = trafficLight.x_min
                                y_max = trafficLight.y_max
                                y_min = trafficLight.y_min
                                if x_min < y < x_max and y_min < x < y_max:
                                    image_array[x][y] = [255, 255, 255]
                        elif typeOfTrafficLight == "Yellow":
                            if trafficLight.label != "Yellow" and trafficLight.label != "YellowRight" and \
                                    trafficLight.label != "YellowLeft":
                                continue
                            else:
                                x_max = trafficLight.x_max
                                x_min = trafficLight.x_min
                                y_max = trafficLight.y_max
                                y_min = trafficLight.y_min
                                if x_min < y < x_max and y_min < x < y_max:
                                    image_array[x][y] = [255, 255, 255]
                        elif typeOfTrafficLight == "All":
                            if trafficLight.label != "Yellow" and trafficLight.label != "YellowRight" and \
                                    trafficLight.label != "YellowLeft" and trafficLight.label != "Green" and \
                                    trafficLight.label != "GreenRight" and trafficLight.label != "GreenLeft" and \
                                    trafficLight.label != "Red" and trafficLight.label != "RedLeft" and \
                                    trafficLight.label != "RedRight":
                                continue
                            else:
                                x_max = trafficLight.x_max
                                x_min = trafficLight.x_min
                                y_max = trafficLight.y_max
                                y_min = trafficLight.y_min
                                if x_min < y < x_max and y_min < x < y_max:
                                    image_array[x][y] = [255, 255, 255]

            finalImage = Image.fromarray(image_array)

            if typeOfTrafficLight == "Green":
                finalImage.save(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\TrafficLightsBinary\\Image_{imageNumber:04}.png")
            elif typeOfTrafficLight == "Red":
                finalImage.save(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\TrafficLightsBinary\\Image_{imageNumber:04}.png")
            elif typeOfTrafficLight == "Yellow":
                finalImage.save(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\TrafficLightsBinary\\Image_{imageNumber:04}.png")
            elif typeOfTrafficLight == "All":
                finalImage.save(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\TrafficLightsBinary\\Image_{imageNumber:04}.png")

            imageNumber = imageNumber + 1
        countImages += 1


def createImageWithTrafficLightsFromPointsCoordinates_ForALL(limitImages):
    global directory, pathImage, listOfTrafficLights

    directoryForAll = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\Images'

    imageNumber = 0
    countImages = 0
    for filename in os.listdir(directoryForAll):
        if countImages == limitImages:
            break
        if filename.endswith(".png"):
            pathImage = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\Images\\' + filename
            listOfTrafficLights = Utility.extractSpecificBox(pathImage, 0, "All")

            maximValuesForEachTrafficLight = []
            for t in range(0, len(listOfTrafficLights)):
                trafficLight = listOfTrafficLights[t]
                x_max = trafficLight.x_max
                x_min = trafficLight.x_min
                y_max = trafficLight.y_max
                y_min = trafficLight.y_min
                maxValue = getMaximValueFromFourPoints(pathImage, int(x_min), int(x_max), int(y_min),
                                                       int(y_max))
                maximValuesForEachTrafficLight.append(maxValue)

            image = Image.new('RGB', (1280, 720), color=(0, 0, 0))
            image_array = np.array(image)

            for x in range(len(image_array)):
                for y in range(len(image_array[0])):
                    for z in range(0, len(listOfTrafficLights)):
                        trafficLight = listOfTrafficLights[z]

                        if trafficLight.label == "Green" or trafficLight.label != "GreenRight" or trafficLight.label == "GreenLeft":
                            x_max = trafficLight.x_max
                            x_min = trafficLight.x_min
                            y_max = trafficLight.y_max
                            y_min = trafficLight.y_min

                            if x_min < y < x_max and y_min < x < y_max:
                                image_array[x][y] = maximValuesForEachTrafficLight[z]

                        if trafficLight.label == "Red" or trafficLight.label == "RedRight" or trafficLight.label == "RedLeft":
                            x_max = trafficLight.x_max
                            x_min = trafficLight.x_min
                            y_max = trafficLight.y_max
                            y_min = trafficLight.y_min

                            if x_min < y < x_max and y_min < x < y_max:
                                image_array[x][y] = maximValuesForEachTrafficLight[z]

                        if trafficLight.label == "Yellow" or trafficLight.label == "YellowRight" or trafficLight.label == "YellowLeft":
                            x_max = trafficLight.x_max
                            x_min = trafficLight.x_min
                            y_max = trafficLight.y_max
                            y_min = trafficLight.y_min

                            if x_min < y < x_max and y_min < x < y_max:
                                image_array[x][y] = maximValuesForEachTrafficLight[z]

            finalImage = Image.fromarray(image_array)
            finalImage.save(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\TrafficLightsBinary\\Image_{imageNumber:04}.png")

            imageNumber = imageNumber + 1
        countImages += 1


def detectTrafficLights_ForTestImage(imagePath, typeOfTrafficLight, checker):
    global listOfTrafficLights

    if typeOfTrafficLight == "Green":
        if checker == 0:
            listOfTrafficLights = Utility.extractSpecificBox(imagePath, 1, "Green")
        elif checker == 1:
            listOfTrafficLights = Utility.extractSpecificBox(imagePath, 2, "Green")
    elif typeOfTrafficLight == "Red":
        if checker == 0:
            listOfTrafficLights = Utility.extractSpecificBox(imagePath, 1, "Red")
        elif checker == 1:
            listOfTrafficLights = Utility.extractSpecificBox(imagePath, 2, "Red")
    elif typeOfTrafficLight == "Yellow":
        if checker == 0:
            listOfTrafficLights = Utility.extractSpecificBox(imagePath, 1, "Yellow")
        elif checker == 1:
            listOfTrafficLights = Utility.extractSpecificBox(imagePath, 2, "Yellow")
    elif typeOfTrafficLight == "All":
        listOfTrafficLights = Utility.extractSpecificBox(imagePath, 1, "All")

    image = Image.new('RGB', (1280, 720), color=(0, 0, 0))
    image_array = np.array(image)

    for x in range(len(image_array)):
        for y in range(len(image_array[0])):
            for z in range(0, len(listOfTrafficLights)):
                trafficLight = listOfTrafficLights[z]

                if typeOfTrafficLight == "Green":
                    if trafficLight.label != "Green" and trafficLight.label != "GreenRight" and trafficLight.label != "GreenLeft":
                        continue
                    else:
                        x_max = trafficLight.x_max
                        x_min = trafficLight.x_min
                        y_max = trafficLight.y_max
                        y_min = trafficLight.y_min
                        if x_min < y < x_max and y_min < x < y_max:
                            image_array[x][y] = [255, 255, 255]
                elif typeOfTrafficLight == "Red":
                    if trafficLight.label != "Red" and trafficLight.label != "RedRight" and trafficLight.label != "RedLeft":
                        continue
                    else:
                        x_max = trafficLight.x_max
                        x_min = trafficLight.x_min
                        y_max = trafficLight.y_max
                        y_min = trafficLight.y_min
                        if x_min < y < x_max and y_min < x < y_max:
                            image_array[x][y] = [255, 255, 255]
                elif typeOfTrafficLight == "Yellow":
                    if trafficLight.label != "Yellow" and trafficLight.label != "YellowRight" and trafficLight.label != "YellowLeft":
                        continue
                    else:
                        x_max = trafficLight.x_max
                        x_min = trafficLight.x_min
                        y_max = trafficLight.y_max
                        y_min = trafficLight.y_min
                        if x_min < y < x_max and y_min < x < y_max:
                            image_array[x][y] = [255, 255, 255]
                elif typeOfTrafficLight == "All":
                    if trafficLight.label != "Yellow" and trafficLight.label != "YellowRight" and \
                            trafficLight.label != "YellowLeft" and trafficLight.label != "Red" and \
                            trafficLight.label != "RedLeft" and trafficLight.label != "RedRight" and \
                            trafficLight.label != "Green" and trafficLight.label != "GreenLeft" and \
                            trafficLight.label != "GreenRight":
                        continue
                    else:
                        x_max = trafficLight.x_max
                        x_min = trafficLight.x_min
                        y_max = trafficLight.y_max
                        y_min = trafficLight.y_min
                        if x_min < y < x_max and y_min < x < y_max:
                            image_array[x][y] = [255, 255, 255]

    if checker == 0:
        if typeOfTrafficLight == "Green":
            finalImage = Image.fromarray(image_array)
            finalImage.save(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight.jpg")
        elif typeOfTrafficLight == "Red":
            finalImage = Image.fromarray(image_array)
            finalImage.save(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight.jpg")
        elif typeOfTrafficLight == "Yellow":
            finalImage = Image.fromarray(image_array)
            finalImage.save(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight.jpg")
        elif typeOfTrafficLight == "All":
            finalImage = Image.fromarray(image_array)
            finalImage.save(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight.jpg")
    elif checker == 1:
        if typeOfTrafficLight == "Green":
            finalImage = Image.fromarray(image_array)
            finalImage.save(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight_Green.jpg")
        elif typeOfTrafficLight == "Red":
            finalImage = Image.fromarray(image_array)
            finalImage.save(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight_Red.jpg")
        elif typeOfTrafficLight == "Yellow":
            finalImage = Image.fromarray(image_array)
            finalImage.save(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight_Yellow.jpg")


def createImageFromClassifierImageResult_Final_InCircle(arrayOfPointsCoordinates, imagePath):
    image = cv2.imread(imagePath)
    listOfPoints = arrayOfPointsCoordinates

    sortedListByTrafficLights = [[], []]

    firstPositions = re.findall(r'\d+', listOfPoints[0])
    firstX = int(firstPositions[0])
    firstY = int(firstPositions[1])
    sortedListByTrafficLights[0].append(firstPositions)

    limitX = firstX + 200
    limitY = firstY + 100
    for x in range(1, len(listOfPoints)):
        listOfPositionNumbers = re.findall(r'\d+', listOfPoints[x])
        positionX = int(listOfPositionNumbers[0])
        positionY = int(listOfPositionNumbers[1])

        if firstX - 50 < positionX < limitX and firstY - 50 < positionY < limitY:
            sortedListByTrafficLights[0].append(listOfPositionNumbers)
        else:
            sortedListByTrafficLights[1].append(listOfPositionNumbers)

    limitList = 0
    while limitList < len(sortedListByTrafficLights):
        currentList = sortedListByTrafficLights[limitList]

        xMinimPosition = sys.maxsize
        xMaximPosition = -sys.maxsize - 1
        yMinimPosition = sys.maxsize
        yMaximPosition = -sys.maxsize - 1
        for x in range(0, len(currentList)):
            positionX = int(currentList[x][0])
            positionY = int(currentList[x][1])

            if positionX < xMinimPosition:
                xMinimPosition = positionX
            if positionX > xMaximPosition:
                xMaximPosition = positionX
            if positionY < yMinimPosition:
                yMinimPosition = positionY
            if positionY > yMaximPosition:
                yMaximPosition = positionY

        middleXPosition = int((xMaximPosition + xMinimPosition) / 2)
        middleYPosition = int((yMaximPosition + yMinimPosition) / 2)

        cv2.circle(image, (middleYPosition, middleXPosition), 50, (0, 0, 255), 5)
        limitList += 1

    cv2.imshow('Test image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createImageFromClassifierImageResult_Final_InRectangle(arrayOfPointsCoordinates, imagePath, colorTL):
    global yForXMin, yForXMax, xForYMin, xForYMax, color
    image = cv2.imread(imagePath)
    listOfPoints = arrayOfPointsCoordinates

    sortedListByTrafficLights = [[], []]

    firstPositions = re.findall(r'\d+', listOfPoints[0])
    firstX = int(firstPositions[0])
    firstY = int(firstPositions[1])
    sortedListByTrafficLights[0].append(firstPositions)

    limitX = firstX + 200
    limitY = firstY + 100
    for x in range(1, len(listOfPoints)):
        listOfPositionNumbers = re.findall(r'\d+', listOfPoints[x])
        positionX = int(listOfPositionNumbers[0])
        positionY = int(listOfPositionNumbers[1])

        if firstX - 50 < positionX < limitX and firstY - 50 < positionY < limitY:
            sortedListByTrafficLights[0].append(listOfPositionNumbers)
        else:
            sortedListByTrafficLights[1].append(listOfPositionNumbers)

    limitList = 0
    while limitList < len(sortedListByTrafficLights):
        currentList = sortedListByTrafficLights[limitList]

        if not sortedListByTrafficLights[limitList]:
            limitList += 1
            continue

        xMinimPosition = sys.maxsize
        xMaximPosition = -sys.maxsize - 1
        yMinimPosition = sys.maxsize
        yMaximPosition = -sys.maxsize - 1
        for x in range(0, len(currentList)):
            positionX = int(currentList[x][0])
            positionY = int(currentList[x][1])

            if positionX < xMinimPosition:
                xMinimPosition = positionX
            if positionX > xMaximPosition:
                xMaximPosition = positionX
            if positionY < yMinimPosition:
                yMinimPosition = positionY
            if positionY > yMaximPosition:
                yMaximPosition = positionY

        point_1 = (yMinimPosition, xMinimPosition)
        point_2 = (yMinimPosition, xMaximPosition)
        point_3 = (yMaximPosition, xMinimPosition)
        point_4 = (yMaximPosition, xMaximPosition)

        if colorTL == "Green":
            color = (0, 255, 0)
        elif colorTL == "Red":
            color = (0, 0, 255)
        elif colorTL == "Yellow":
            color = (0, 255, 255)

        thickness = 2

        image = cv2.line(image, point_1, point_2, color, thickness)
        image = cv2.line(image, point_1, point_3, color, thickness)
        image = cv2.line(image, point_2, point_4, color, thickness)
        image = cv2.line(image, point_3, point_4, color, thickness)

        limitList += 1

    cv2.imshow('Test image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createImageFromClassifierImageResult_Final_InRectangle_ForAll(listOfGreen, listOfRed, imagePath):
    global yForXMin, yForXMax, xForYMin, xForYMax
    image = cv2.imread(imagePath)
    listOfPointsFromGreen = listOfGreen
    listOfPointsFromRed = listOfRed

    sortedListByTrafficLights_Green = [[], []]
    sortedListByTrafficLights_Red = [[], []]

    firstPositions = re.findall(r'\d+', listOfPointsFromGreen[0])
    firstX = int(firstPositions[0])
    firstY = int(firstPositions[1])
    sortedListByTrafficLights_Green[0].append(firstPositions)

    limitX = firstX + 200
    limitY = firstY + 100
    for x in range(1, len(listOfPointsFromGreen)):
        listOfPositionNumbers = re.findall(r'\d+', listOfPointsFromGreen[x])
        positionX = int(listOfPositionNumbers[0])
        positionY = int(listOfPositionNumbers[1])

        if firstX - 50 < positionX < limitX and firstY - 50 < positionY < limitY:
            sortedListByTrafficLights_Green[0].append(listOfPositionNumbers)
        else:
            sortedListByTrafficLights_Green[1].append(listOfPositionNumbers)

    firstPositions = re.findall(r'\d+', listOfPointsFromRed[0])
    firstX = int(firstPositions[0])
    firstY = int(firstPositions[1])
    sortedListByTrafficLights_Red[0].append(firstPositions)

    limitX = firstX + 200
    limitY = firstY + 100
    for x in range(1, len(listOfPointsFromRed)):
        listOfPositionNumbers = re.findall(r'\d+', listOfPointsFromRed[x])
        positionX = int(listOfPositionNumbers[0])
        positionY = int(listOfPositionNumbers[1])

        if firstX - 50 < positionX < limitX and firstY - 50 < positionY < limitY:
            sortedListByTrafficLights_Red[0].append(listOfPositionNumbers)
        else:
            sortedListByTrafficLights_Red[1].append(listOfPositionNumbers)

    limitList = 0
    while limitList < len(sortedListByTrafficLights_Red):
        currentList = sortedListByTrafficLights_Red[limitList]

        if not sortedListByTrafficLights_Red[limitList]:
            limitList += 1
            continue

        xMinimPosition = sys.maxsize
        xMaximPosition = -sys.maxsize - 1
        yMinimPosition = sys.maxsize
        yMaximPosition = -sys.maxsize - 1
        for x in range(0, len(currentList)):
            positionX = int(currentList[x][0])
            positionY = int(currentList[x][1])

            if positionX < xMinimPosition:
                xMinimPosition = positionX
            if positionX > xMaximPosition:
                xMaximPosition = positionX
            if positionY < yMinimPosition:
                yMinimPosition = positionY
            if positionY > yMaximPosition:
                yMaximPosition = positionY

        point_1 = (yMinimPosition, xMinimPosition)
        point_2 = (yMinimPosition, xMaximPosition)
        point_3 = (yMaximPosition, xMinimPosition)
        point_4 = (yMaximPosition, xMaximPosition)

        color = (0, 0, 255)
        thickness = 2

        image = cv2.line(image, point_1, point_2, color, thickness)
        image = cv2.line(image, point_1, point_3, color, thickness)
        image = cv2.line(image, point_2, point_4, color, thickness)
        image = cv2.line(image, point_3, point_4, color, thickness)

        limitList += 1

    limitList = 0
    while limitList < len(sortedListByTrafficLights_Green):
        currentList = sortedListByTrafficLights_Green[limitList]

        if not sortedListByTrafficLights_Green[limitList]:
            limitList += 1
            continue

        xMinimPosition = sys.maxsize
        xMaximPosition = -sys.maxsize - 1
        yMinimPosition = sys.maxsize
        yMaximPosition = -sys.maxsize - 1
        for x in range(0, len(currentList)):
            positionX = int(currentList[x][0])
            positionY = int(currentList[x][1])

            if positionX < xMinimPosition:
                xMinimPosition = positionX
            if positionX > xMaximPosition:
                xMaximPosition = positionX
            if positionY < yMinimPosition:
                yMinimPosition = positionY
            if positionY > yMaximPosition:
                yMaximPosition = positionY

        point_1 = (yMinimPosition, xMinimPosition)
        point_2 = (yMinimPosition, xMaximPosition)
        point_3 = (yMaximPosition, xMinimPosition)
        point_4 = (yMaximPosition, xMaximPosition)

        color = (0, 255, 0)
        thickness = 2

        image = cv2.line(image, point_1, point_2, color, thickness)
        image = cv2.line(image, point_1, point_3, color, thickness)
        image = cv2.line(image, point_2, point_4, color, thickness)
        image = cv2.line(image, point_3, point_4, color, thickness)

        limitList += 1

    cv2.imshow('Test image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
