import csv
import glob
import math
import os
import re
import pandas as pd
from csv import reader
from os import path, listdir

import cv2 as cv
import keyboard
import yaml
import itertools
from PIL import Image
from matplotlib import image as mtpimg
from skimage.util.shape import view_as_blocks

from data.TrafficLight import TrafficLight
from functions import ImageFunctions


def cleanPathString(pathImage):
    newPathImage = pathImage.replace("/", "\\")
    newPathImage = newPathImage.replace(".\\",
                                        "C:\\Users\\Marius\\Desktop\\LicentaProiect\\dateLicenta\\dataset_train_rgb.zip\\")

    return newPathImage


def get_image(image_path):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def parseYamlFileAndPrintThem():
    with open(r'C:\Users\Marius\Desktop\LicentaProiect\dateLicenta\dataset_train_rgb.zip\train.yaml') as file:
        documents = yaml.safe_load(file)

        for i in documents:
            if len(i['boxes']) > 0:
                for j in i['boxes']:
                    print(j)
                print(i['path'])
            print('\n')


def createAListOfObjectTrafficLights(numberOfValidImages, onlyTwoColors, checker):
    if checker == 0:
        with open(r'C:\Users\Marius\Desktop\LicentaProiect\dateLicenta\dataset_train_rgb.zip\train.yaml') as file:
            documents = yaml.safe_load(file)

            listOfTrafficLights = []
            listOfTrafficLightsRedGreen = []

            for i in documents:
                if len(i['boxes']) > 0:
                    for j in i['boxes']:
                        label = j['label']
                        occluded = j['occluded']
                        x_max = j['x_max']
                        x_min = j['x_min']
                        y_max = j['y_max']
                        y_min = j['y_min']
                        pathImage = cleanPathString(i['path'])
                        listOfTrafficLights.append(TrafficLight(label, occluded, x_max, x_min, y_max, y_min, pathImage))
                        numberOfValidImages = numberOfValidImages + 1

                        if onlyTwoColors == 1:
                            if label == "Red" or label == "Green":
                                listOfTrafficLightsRedGreen.append(
                                    TrafficLight(label, occluded, x_max, x_min, y_max, y_min, pathImage))
        return listOfTrafficLights, numberOfValidImages, listOfTrafficLightsRedGreen
    elif checker == 1:
        with open(r'C:\Users\Marius\Desktop\LicentaProiect\GreenTL\position.yaml') as file:
            documents = yaml.safe_load(file)

            listOfGreenTrafficLights = []
            numberOfBoxes = 0
            numberOfTrafficLightsPerBox = []
            listOfTrafficLightsFromSameImages = []

            for i in documents:
                if len(i['boxes']) > 0:
                    numberOfTrafficLights = 0
                    for j in i['boxes']:
                        label = j['label']
                        occluded = j['occluded']
                        x_max = j['x_max']
                        x_min = j['x_min']
                        y_max = j['y_max']
                        y_min = j['y_min']
                        pathImage = i['path']
                        listOfGreenTrafficLights.append(
                            TrafficLight(label, occluded, x_max, x_min, y_max, y_min, pathImage))
                        numberOfValidImages = numberOfValidImages + 1
                        numberOfTrafficLights += 1

                    numberOfBoxes += 1
                    numberOfTrafficLightsPerBox.append(numberOfTrafficLights)
                    listOfTrafficLightsFromSameImages.append(listOfGreenTrafficLights)

        return listOfGreenTrafficLights, numberOfValidImages, [numberOfBoxes,
                                                               numberOfTrafficLightsPerBox], listOfTrafficLightsFromSameImages


def createAListOfTrafficLightFromSameImages(checker):
    if checker == 0:
        with open(r'C:\Users\Marius\Desktop\LicentaProiect\dateLicenta\dataset_train_rgb.zip\train.yaml') as file:
            documents = yaml.safe_load(file)
    elif checker == 1:
        with open(r'C:\Users\Marius\Desktop\LicentaProiect\GreenTL\position.yaml') as file:
            documents = yaml.safe_load(file)

    listOfTrafficLightsFromSameImages = []

    for i in documents:
        if len(i['boxes']) > 0:
            listOfTrafficLights = []
            for j in i['boxes']:
                label = j['label']
                occluded = j['occluded']
                x_max = j['x_max']
                x_min = j['x_min']
                y_max = j['y_max']
                y_min = j['y_min']
                pathImage = cleanPathString(i['path'])
                listOfTrafficLights.append(TrafficLight(label, occluded, x_max, x_min, y_max, y_min, pathImage))
                listOfTrafficLightsFromSameImages.append(listOfTrafficLights)

    return listOfTrafficLightsFromSameImages


def extractSpecificBox(imagePath, checker, typeOfTrafficLight):
    if typeOfTrafficLight == "Green":
        if checker == 0:
            with open(r'C:\Users\Marius\Desktop\LicentaProiect\GreenTL\position.yaml') as file:
                documents = yaml.safe_load(file)
        elif checker == 1:
            with open(r'C:\Users\Marius\Desktop\LicentaProiect\GreenTL\testImages\positionTest.yaml') as file:
                documents = yaml.safe_load(file)
        elif checker == 2:
            with open(r'C:\Users\Marius\Desktop\LicentaProiect\AllTL\testImages\positionTest.yaml') as file:
                documents = yaml.safe_load(file)
    elif typeOfTrafficLight == "Red":
        if checker == 0:
            with open(r'C:\Users\Marius\Desktop\LicentaProiect\RedTL\position.yaml') as file:
                documents = yaml.safe_load(file)
        elif checker == 1:
            with open(r'C:\Users\Marius\Desktop\LicentaProiect\RedTL\testImages\positionTest.yaml') as file:
                documents = yaml.safe_load(file)
        elif checker == 2:
            with open(r'C:\Users\Marius\Desktop\LicentaProiect\AllTL\testImages\positionTest.yaml') as file:
                documents = yaml.safe_load(file)
    elif typeOfTrafficLight == "Yellow":
        if checker == 0:
            with open(r'C:\Users\Marius\Desktop\LicentaProiect\YellowTL\position.yaml') as file:
                documents = yaml.safe_load(file)
        elif checker == 1:
            with open(r'C:\Users\Marius\Desktop\LicentaProiect\YellowTL\testImages\positionTest.yaml') as file:
                documents = yaml.safe_load(file)
        elif checker == 2:
            with open(r'C:\Users\Marius\Desktop\LicentaProiect\AllTL\testImages\positionTest.yaml') as file:
                documents = yaml.safe_load(file)
    elif typeOfTrafficLight == "All":
        if checker == 0:
            with open(r'C:\Users\Marius\Desktop\LicentaProiect\AllTL\position.yaml') as file:
                documents = yaml.safe_load(file)
        elif checker == 1:
            with open(r'C:\Users\Marius\Desktop\LicentaProiect\AllTL\testImages\positionTest.yaml') as file:
                documents = yaml.safe_load(file)

    listOfTrafficLightsFromTheSpecificBox = []
    for i in documents:
        if len(i['boxes']) > 0:
            if i['path'] == imagePath:
                for j in i['boxes']:
                    label = j['label']
                    occluded = j['occluded']
                    x_max = j['x_max']
                    x_min = j['x_min']
                    y_max = j['y_max']
                    y_min = j['y_min']
                    pathImage = i['path']
                    listOfTrafficLightsFromTheSpecificBox.append(TrafficLight(label, occluded, x_max, x_min, y_max,
                                                                              y_min, pathImage))
                break

    return listOfTrafficLightsFromTheSpecificBox


def cropImagesFromListOfObjects(listOfTrafficLight, onlyTwoColors):
    numberImage = 0
    numberImageRedGreen = 0
    listOfLabels = []
    listOfTrafficLightRedGreen = []

    for obj in listOfTrafficLight:
        if not path.exists(obj.path):
            continue

        originalImage = Image.open(obj.path, "r")
        left = obj.x_min
        right = obj.x_max
        top = obj.y_min
        bottom = obj.y_max
        if right > left and bottom > top:
            cropImage = originalImage.crop((left, top, right, bottom))

            cropImage.save(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\trafficLightImages\\trafficLightImage_{numberImage:04}.jpg",
                quality=95)

            if onlyTwoColors == 1:
                if obj.label == "Red" or obj.label == "Green":
                    cropImage.save(
                        f"C:\\Users\\Marius\Desktop\\LicentaProiect\\trafficLightImagesRedGreen\\trafficLightImage_{numberImageRedGreen:04}.jpg",
                        quality=95)
                    numberImageRedGreen = numberImageRedGreen + 1
                    listOfTrafficLightRedGreen.append(obj)

            numberImage = numberImage + 1
            listOfLabels.append(obj.label)

    listOfGoodLabels = listOfLabels
    return listOfGoodLabels, listOfTrafficLightRedGreen


def makeAListWithAllLabelsForCropImages(listOfTrafficLight):
    listOfLabels = []
    for obj in listOfTrafficLight:
        if not path.exists(obj.path):
            continue

        left = obj.x_min
        right = obj.x_max
        top = obj.y_min
        bottom = obj.y_max

        if right > left and bottom > top:
            listOfLabels.append(obj.label)

    return listOfLabels


def createVersion_1_ofDataset(listOfTrafficLight, dimension):
    listOfLabels = makeAListWithAllLabelsForCropImages(listOfTrafficLight)
    row_list = [["R", "G", "B", "color"]]
    numberImage = 0

    for counter in range(dimension):
        pathImage = f'C:\\Users\\Marius\Desktop\\LicentaProiect\\trafficLightImages\\trafficLightImage_{numberImage:04}.jpg'
        pixelArray = ImageFunctions.valueOfPixelsFromImage(pathImage)
        numberImage += 1

        valueOfColor = -1
        firstValue = pixelArray[0]
        secondValue = pixelArray[1]
        thirdValue = pixelArray[2]
        color = listOfLabels[counter]

        if color == 'Red':
            valueOfColor = 0
        elif color == 'Green':
            valueOfColor = 1

        color = valueOfColor

        if color == 1 or color == 0:
            valueToPutInListOfCSV = [firstValue, secondValue, thirdValue, color]
            row_list.append(valueToPutInListOfCSV)

    with open('Version_1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def modifyAllImagesToHaveOnlyTrafficLights(listOfAllImagesWithTrafficLights):
    numberImage = 0
    for obj in listOfAllImagesWithTrafficLights:
        newImage = ImageFunctions.getANewImageWithTrafficLights(obj)

        cv.imwrite(f"C:\\Users\\Marius\Desktop\\LicentaProiect\\imageWithOnlyTrafficLights\\image_{numberImage:04}.jpg",
                   newImage)
        numberImage = numberImage + 1


def convertImagesFromRGBtoHSV():
    numberImage = 0
    directory = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\trafficLightImages'

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            pathImage = f'C:\\Users\\Marius\Desktop\\LicentaProiect\\trafficLightImages\\trafficLightImage_{numberImage:04}.jpg'
            hsvImage = ImageFunctions.convertImageRGBtoHSV(pathImage)

            cv.imwrite(f"C:\\Users\\Marius\Desktop\\LicentaProiect\\trafficLightsInHSV\\image_{numberImage:04}.jpg",
                       hsvImage)
            numberImage = numberImage + 1


def createVersion_2_ofDataset(listWithRGBValues, listOfHSVValues):
    indicesForHistogram = []
    row_list = [[]]

    for i in range(0, 256):
        indicesForHistogram.append(i)

    totalIndices = indicesForHistogram
    totalIndices.append("R")
    totalIndices.append("G")
    totalIndices.append("B")
    totalIndices.append("H")
    totalIndices.append("S")
    totalIndices.append("V")
    totalIndices.append("Clasa")

    row_list[0] = totalIndices

    numberImage = 0
    dimension = len(listWithRGBValues)
    for counter in range(dimension):
        valueToPutInList = []
        pathImage = f'C:\\Users\\Marius\Desktop\\LicentaProiect\\trafficLightImagesRedGreen\\trafficLightImage_{numberImage:04}.jpg'
        histogram = ImageFunctions.getHistogramArrayOfAnImage(pathImage)

        numberImage += 1
        for h in histogram:
            valueToPutInList.append(h)

        arrayOfRGBValue = listWithRGBValues[counter]
        r = arrayOfRGBValue[0]
        g = arrayOfRGBValue[1]
        b = arrayOfRGBValue[2]
        clasa = arrayOfRGBValue[3]

        arrayOfHSVValue = listOfHSVValues[counter]
        h = arrayOfHSVValue[0]
        s = arrayOfHSVValue[1]
        v = arrayOfHSVValue[2]

        valueToPutInList.append(r)
        valueToPutInList.append(g)
        valueToPutInList.append(b)
        valueToPutInList.append(h)
        valueToPutInList.append(s)
        valueToPutInList.append(v)
        valueToPutInList.append(clasa)

        row_list.append(valueToPutInList)

    with open('Version_2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def createVersion_3_ofDataset(listWithRGBValues, listOfHSVValues):
    indicesForHistograms = []
    row_list = [[]]

    for i in range(0, 384):
        indicesForHistograms.append(i)

    totalIndices = indicesForHistograms
    totalIndices.append("R")
    totalIndices.append("G")
    totalIndices.append("B")
    totalIndices.append("H")
    totalIndices.append("S")
    totalIndices.append("V")
    totalIndices.append("Average Red Top")
    totalIndices.append("Average Red Bottom")
    totalIndices.append("Average Green Top")
    totalIndices.append("Average Green Bottom")
    totalIndices.append("Clasa")

    row_list[0] = totalIndices
    numberImage = 0
    dimension = len(listWithRGBValues)
    for counter in range(dimension):
        valueToPutInList = []
        pathImage = f'C:\\Users\\Marius\Desktop\\LicentaProiect\\trafficLightImagesRedGreen\\trafficLightImage_{numberImage:04}.jpg'
        blueChannel, greenChannel, redChannel, hueChannel, saturationChannel, valueChannel = ImageFunctions.createHistogramForColorImage(
            pathImage)

        numberImage += 1
        for x in blueChannel:
            valueToPutInList.append(x)

        for x in greenChannel:
            valueToPutInList.append(x)

        for x in redChannel:
            valueToPutInList.append(x)

        for x in hueChannel:
            valueToPutInList.append(x)

        for x in saturationChannel:
            valueToPutInList.append(x)

        for x in valueChannel:
            valueToPutInList.append(x)

        arrayOfRGBValue = listWithRGBValues[counter]
        r = arrayOfRGBValue[0]
        g = arrayOfRGBValue[1]
        b = arrayOfRGBValue[2]
        clasa = arrayOfRGBValue[3]

        arrayOfHSVValue = listOfHSVValues[counter]
        h = arrayOfHSVValue[0]
        s = arrayOfHSVValue[1]
        v = arrayOfHSVValue[2]

        topAverageGreen, bottomAverageGreen, topAverageRed, bottomAverageRed = ImageFunctions.averageRedGreenChannelsOnTopAndBottom(
            pathImage)

        valueToPutInList.append(r)
        valueToPutInList.append(g)
        valueToPutInList.append(b)
        valueToPutInList.append(h)
        valueToPutInList.append(s)
        valueToPutInList.append(v)
        valueToPutInList.append(topAverageRed)
        valueToPutInList.append(bottomAverageRed)
        valueToPutInList.append(topAverageGreen)
        valueToPutInList.append(bottomAverageGreen)
        valueToPutInList.append(clasa)

        row_list.append(valueToPutInList)

    with open('Version_3.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def createVersion_4_ofDataset(listWithRGBValues, listOfHSVValues):
    indicesForHistograms = []
    row_list = [[]]

    for i in range(0, 96):
        indicesForHistograms.append(i)

    totalIndices = indicesForHistograms
    totalIndices.append("R")
    totalIndices.append("G")
    totalIndices.append("B")
    totalIndices.append("H")
    totalIndices.append("S")
    totalIndices.append("V")
    totalIndices.append("Average Red Top")
    totalIndices.append("Average Red Bottom")
    totalIndices.append("Average Green Top")
    totalIndices.append("Average Green Bottom")
    totalIndices.append("Clasa")

    row_list[0] = totalIndices
    numberImage = 0
    dimension = len(listWithRGBValues)
    for counter in range(dimension):
        valueToPutInList = []
        pathImage = f'C:\\Users\\Marius\Desktop\\LicentaProiect\\trafficLightImagesRedGreen\\trafficLightImage_{numberImage:04}.jpg'
        blueChannel, greenChannel, redChannel, hueChannel, saturationChannel, valueChannel = ImageFunctions.createHistogramForColorImage(
            pathImage)

        numberImage += 1
        for x in blueChannel:
            valueToPutInList.append(x)

        for x in greenChannel:
            valueToPutInList.append(x)

        for x in redChannel:
            valueToPutInList.append(x)

        for x in hueChannel:
            valueToPutInList.append(x)

        for x in saturationChannel:
            valueToPutInList.append(x)

        for x in valueChannel:
            valueToPutInList.append(x)

        arrayOfRGBValue = listWithRGBValues[counter]
        r = arrayOfRGBValue[0]
        g = arrayOfRGBValue[1]
        b = arrayOfRGBValue[2]
        clasa = arrayOfRGBValue[3]

        arrayOfHSVValue = listOfHSVValues[counter]
        h = arrayOfHSVValue[0]
        s = arrayOfHSVValue[1]
        v = arrayOfHSVValue[2]

        topAverageGreen, bottomAverageGreen, topAverageRed, bottomAverageRed = ImageFunctions.averageRedGreenChannelsOnTopAndBottom(
            pathImage)

        valueToPutInList.append(r)
        valueToPutInList.append(g)
        valueToPutInList.append(b)
        valueToPutInList.append(h)
        valueToPutInList.append(s)
        valueToPutInList.append(v)
        valueToPutInList.append(topAverageRed)
        valueToPutInList.append(bottomAverageRed)
        valueToPutInList.append(topAverageGreen)
        valueToPutInList.append(bottomAverageGreen)
        valueToPutInList.append(clasa)

        row_list.append(valueToPutInList)

    with open('Version_4.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def cropImagesByHalf():
    numberImage = 0
    directory = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\Images'

    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            pathImage = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\Images\\' + filename
            image = cv.imread(pathImage)

            height = image.shape[0]
            width = image.shape[1]

            cropImage = image[0:int(height / 2), 0:width]
            newPath = f"C:\\Users\\Marius\Desktop\\LicentaProiect\\CropImages\\image_{numberImage:04}.jpg"
            cv.imwrite(newPath, cropImage)
            numberImage = numberImage + 1


def createAFolderWithGrayscaleImages():
    numberImageForGray = 0
    directory = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\CropImages'

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            pathImage = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\CropImages\\' + filename

            grayscale = ImageFunctions.convertAnImageToGrayscale(pathImage)
            cv.imwrite(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\CropedImagesInGray\\image_{numberImageForGray:04}.jpg",
                grayscale)
            numberImageForGray = numberImageForGray + 1


def makeCSVWithPosition(imagePath):
    image = Image.open(imagePath)

    width, height = image.size
    row_list = [[]]

    firstLine = []
    for i in range(0, 1280):
        firstLine.append(i)

    row_list[0] = firstLine
    lineListPosition = []

    for i in range(0, height):
        for j in range(0, width):
            lineListPosition.append([i, j])
        row_list.append(lineListPosition)
        lineListPosition = []

    with open('CSVwithPositions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def createCSVWithPatches(imagePath, imageType):
    global resultFromPatches

    valueFromPatchCell = []
    row_list = [[]]

    for i in range(0, 25):
        valueFromPatchCell.append(i)

    row_list[0] = valueFromPatchCell

    image = mtpimg.imread(imagePath)

    if imageType == 0:
        imageArray = image[:, :, 0]
        resultFromPatches = view_as_blocks(imageArray, block_shape=(5, 5))
    elif imageType == 1:
        resultFromPatches = view_as_blocks(image, block_shape=(5, 5))

    height = resultFromPatches.shape[0]
    width = resultFromPatches.shape[1]

    for x in range(0, height):
        setOfPatches = resultFromPatches[x]
        for i in range(0, width):
            currentPatchToPutInCSV = []
            for j in range(0, 5):
                currentArrayFromPatch = setOfPatches[i][j]
                for y in range(0, 5):
                    currentPatchToPutInCSV.append(currentArrayFromPatch[y])

            row_list.append(currentPatchToPutInCSV)

    if imageType == 0:
        with open('CSVwithPatches.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)
    elif imageType == 1:
        with open('CSVwithPatchesForMaskedImage.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)


def populateRows_V1(rowList, list_1, list_2, list_3, position):
    currentPatch = list_1[position]
    currentMaskedList = list_2[position]
    ##currentPositionList = list_3[position]

    for x in range(0, len(currentPatch)):
        value = int(currentPatch[x])
        rowList.append(value)

    for x in range(0, len(currentMaskedList)):
        value = int(currentMaskedList[x])
        rowList.append(value)

    return rowList


def populateRows_V2(rowList, list_1, position):
    currentPatch = list_1[position]

    for x in range(0, len(currentPatch)):
        value = int(currentPatch[x])
        rowList.append(value)

    return currentPatch


def divideChunks(myList, l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

    n = 9
    x = list(divideChunks(myList, n))
    return x


def CreateCSVFromHOG(imagePath):
    (fd, hogImage) = ImageFunctions.createHOGHistogram(imagePath)

    dimensionForEachHistogram = 9
    myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                     for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

    valueFromPatchCell = []
    row_list = [[]]

    for i in range(0, 9):
        valueFromPatchCell.append(i)
    row_list[0] = valueFromPatchCell

    for i in range(0, len(myListFromHOG)):
        row_list.append(myListFromHOG[i])

    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\TestCSVs\\HOGForTestImage.csv",
            'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def createCSVForTrafficLightClassification_Version1():
    valueFromPatchCell = []
    row_list = [[]]

    for i in range(0, 50):
        valueFromPatchCell.append(i)
    valueFromPatchCell.append('Clasa')

    row_list[0] = valueFromPatchCell

    listOfPatches = [[]]
    listOfPatchesForMaskedImage = [[]]
    listOfPositionPatches = [[]]

    with open('CSVwithPatches.csv', 'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfPatches.append(row)

    with open('CSVwithPatchesForMaskedImage.csv', 'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfPatchesForMaskedImage.append(row)

    with open('CSVwithPositionPatches.csv', 'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfPositionPatches.append(row)

    validPatch = 0
    validPatches = []
    dimensionCSVs = len(listOfPatches)

    for i in range(1, len(listOfPatchesForMaskedImage)):
        currentMaskedImageList = listOfPatchesForMaskedImage[i]

        counterWhitePixels = 0
        for j in range(0, len(currentMaskedImageList)):
            currentMaskedPixel = int(currentMaskedImageList[j])
            listOfPositionNumbers = re.findall(r'\d+', listOfPositionPatches[i][j])

            positionX = int(listOfPositionNumbers[0])
            positionY = int(listOfPositionNumbers[1])

            if currentMaskedPixel > 100 and 50 < positionX < 300 and 200 < positionY < 1000:
                counterWhitePixels += 1

        if counterWhitePixels > 5:
            validPatch += 1
            validPatches.append(i)

    for i in range(1, dimensionCSVs):
        valueToPutInList = []
        if validPatches.__contains__(i):
            valueToPutInList = populateRows_V1(valueToPutInList, listOfPatches, listOfPatchesForMaskedImage,
                                               listOfPositionPatches, i)
            valueToPutInList.append(1)
        else:
            valueToPutInList = populateRows_V1(valueToPutInList, listOfPatches, listOfPatchesForMaskedImage,
                                               listOfPositionPatches, i)
            valueToPutInList.append(0)

        row_list.append(valueToPutInList)

    with open('Version_1_trafficLight_train.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def createCSVForTrafficLightClassification_Version2(limitImages):
    directory = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\Images\\'
    files = listdir(directory)
    images_list = [i for i in files if i.endswith('.png')]

    currentBox = 0
    csvNumber = 0
    countImages = 0

    totalValidPatches = 0
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break

        pathImage = directory + image
        ImageFunctions.createCSVWithPatches_Long(pathImage, 0, csvNumber, "Green")
        ImageFunctions.createCSVWithPosition(pathImage, 0, csvNumber, "Green")

        listOfPatches = []
        listOfPositions = []
        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)

        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])

        listOfTrafficLightForCurrentImage = extractSpecificBox(pathImage, 0, "Green")

        validPatch = 0
        validPatches = []

        for i in range(0, len(listOfPositions)):
            currentListOfPositions = listOfPositions[i]

            okPatch = 0
            for j in range(0, len(listOfTrafficLightForCurrentImage)):
                trafficLight = listOfTrafficLightForCurrentImage[j]
                if trafficLight.label != "Green" and trafficLight.label != "GreenRight" and trafficLight.label != "GreenLeft":
                    continue
                else:
                    x_max = trafficLight.x_max
                    x_min = trafficLight.x_min
                    y_max = trafficLight.y_max
                    y_min = trafficLight.y_min

                    for p in range(0, len(currentListOfPositions)):
                        listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                        positionX = int(listOfPositionNumbers[0])
                        positionY = int(listOfPositionNumbers[1])

                        if x_min < positionY < x_max and y_min < positionX < y_max:
                            okPatch = 1
                            validPatch += 1
                            validPatches.append(i)
                            break
                if okPatch == 1:
                    break

        print("Valid patches for ", pathImage, " --------- ", validPatch)
        totalValidPatches += validPatch

        valueFromPatchCell = []
        row_list = [[]]
        dimensionCSV = len(listOfPatches)

        valueFromPatchCell.append('Position')
        for i in range(0, 209):
            valueFromPatchCell.append(i)
        valueFromPatchCell.append('Clasa')

        row_list[0] = valueFromPatchCell

        (fd, hogImage) = ImageFunctions.createHOGHistogram(pathImage)

        dimensionForEachHistogram = 9
        myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                         for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

        numberList = 0
        for i in range(0, dimensionCSV):
            valueToPutInList = []
            if validPatches.__contains__(i):
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                for x in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[x]
                    valueToPutInList.append(value)

                # print(myListFromHOG[numberList])
                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))
                valueToPutInList.append(1)
            else:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                for x in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[x]
                    valueToPutInList.append(value)

                # print(myListFromHOG[numberList])
                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))
                valueToPutInList.append(0)

            row_list.append(valueToPutInList)
            numberList += 1

        # print(myListFromHOG[len(myListFromHOG)-1])
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\finalTrainCSVs\\FinalCSVForTrain_{csvNumber:04}.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        csvNumber += 1
        currentBox += 1
        countImages += 1

    print("\n")
    print("Total number of valid patches ------- ", totalValidPatches)

    pathTrain = "C:/Users/Marius/Desktop/LicentaProiect/GreenTL/finalTrainCSVs"

    all_files = glob.glob(os.path.join(pathTrain, "FinalCSVForTrain_*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)

    df_merged.to_csv("Version_2_trafficLight_train.csv", index=False)


def createCSVForClassificationOfTrafficLights_Green(limitImages):
    global directoryForTrainImages, directoryForBinaryImages, pathImage, listOfTrafficLightForCurrentImage, listOfTrafficLightForCurrentImage, pathTrain

    directoryForTrainImages = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\Images\\'
    directoryForBinaryImages = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\BinaryImages\\'

    files = listdir(directoryForTrainImages)
    images_list = [i for i in files if i.endswith('.png')]

    ImageFunctions.detectGreenFromTrafficLights(limitImages)

    csvNumber = 0
    countImages = 0
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForTrainImages + image

        ImageFunctions.createCSVWithPatches_Long(pathImage, 0, csvNumber, "Green")
        ImageFunctions.createCSVWithPosition(pathImage, 0, csvNumber, "Green")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    countImages = 0

    files = listdir(directoryForBinaryImages)
    images_list = [i for i in files if i.endswith('.jpg')]
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForBinaryImages + image

        ImageFunctions.createCSVWithBinaryValues(pathImage, 0, csvNumber, "Green")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    imageNumber = 0
    totalValidPatches = 0
    totalValidPatchesAfterBinaryTriage = 0
    for x in range(0, limitImages):

        listOfPatches = []
        listOfPositions = []
        listOfBinaryPatches = []

        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)
        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\BinaryValuesCSV\\CSVwithBinaryValues_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfBinaryPatches.append(row)

        pathImage = f"C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\Images\\Image_{imageNumber:04}.png"

        (fd, hogImage) = ImageFunctions.createHOGHistogram(pathImage)
        dimensionForEachHistogram = 9
        myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                         for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

        listOfTrafficLightForCurrentImage = extractSpecificBox(pathImage, 0, "Green")

        validPatch = 0
        validPatches = []

        for i in range(0, len(listOfPositions)):
            currentListOfPositions = listOfPositions[i]

            okPatch = 0
            for j in range(0, len(listOfTrafficLightForCurrentImage)):
                trafficLight = listOfTrafficLightForCurrentImage[j]
                if trafficLight.label != "Green" and trafficLight.label != "GreenRight" and trafficLight.label != "GreenLeft":
                    continue
                else:
                    x_max = trafficLight.x_max
                    x_min = trafficLight.x_min
                    y_max = trafficLight.y_max
                    y_min = trafficLight.y_min

                    for p in range(0, len(currentListOfPositions)):
                        listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                        positionX = int(listOfPositionNumbers[0])
                        positionY = int(listOfPositionNumbers[1])

                        if x_min < positionY < x_max and y_min < positionX < y_max:
                            okPatch = 1
                            validPatch += 1
                            validPatches.append(i)
                            break
                if okPatch == 1:
                    break

        print("Valid patches before binary triage for ", pathImage, " --------- ", validPatch)
        totalValidPatches += validPatch

        valueFromPatchCell = []
        row_list = [[]]
        dimensionCSV = len(listOfPatches)

        valueFromPatchCell.append('Position')
        for i in range(0, 234):
            valueFromPatchCell.append(i)
        valueFromPatchCell.append('Clasa')

        row_list[0] = valueFromPatchCell

        numberList = 0
        validPatchAfterBinaryTriage = 0
        for i in range(0, dimensionCSV):
            valueToPutInList = []
            counterOfWhitePixels = 0
            if validPatches.__contains__(i):
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfBinaryPatches[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                # print(myListFromHOG[numberList])
                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = arrayOfBinaryPatches[y]
                    if 100 <= int(value) <= 255:
                        counterOfWhitePixels += 1

                if counterOfWhitePixels >= 1:
                    for y in range(0, len(arrayOfBinaryPatches)):
                        value = arrayOfBinaryPatches[y]
                        valueToPutInList.append(value)
                    valueToPutInList.append(1)
                else:
                    for y in range(0, 25):
                        value = 0
                        valueToPutInList.append(value)
                    valueToPutInList.append(0)
            else:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfBinaryPatches[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                # print(myListFromHOG[numberList])
                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = arrayOfBinaryPatches[y]
                    valueToPutInList.append(value)

                valueToPutInList.append(0)

            row_list.append(valueToPutInList)
            numberList += 1
            totalValidPatchesAfterBinaryTriage += validPatchAfterBinaryTriage
            validPatchAfterBinaryTriage = 0

        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\finalTrainCSVs\\FinalCSVForTrain_{csvNumber:04}.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        csvNumber += 1
        imageNumber += 1

    print("\n")
    print("Total number of valid patches ------- without binary triage: ", totalValidPatches)
    print("Total number of valid patches ------- with binary triage: ", totalValidPatchesAfterBinaryTriage)

    pathTrain = "C:/Users/Marius/Desktop/LicentaProiect/GreenTL/finalTrainCSVs"

    all_files = glob.glob(os.path.join(pathTrain, "FinalCSVForTrain_*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)

    df_merged.to_csv("DatasetForAllImages_Green.csv", index=False)


def FinalMethodToCreateCSVForClassificationOfGreenTrafficLight(limitImages):
    global directoryForTrainImages, directoryForBinaryImages, pathImage, listOfTrafficLightForCurrentImage, listOfTrafficLightForCurrentImage, pathTrain, OK

    directoryForTrainImages = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\Images\\'
    directoryForBinaryImages = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\TrafficLightsBinary\\'

    files = listdir(directoryForTrainImages)
    images_list = [i for i in files if i.endswith('.png')]

    ImageFunctions.createImageWithTrafficLightsFromPointsCoordinates(limitImages, "Green")

    csvNumber = 0
    countImages = 0
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForTrainImages + image

        ImageFunctions.createCSVWithPatches_Long(pathImage, 0, csvNumber, "Green")
        ImageFunctions.createCSVWithPosition(pathImage, 0, csvNumber, "Green")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    countImages = 0

    files = listdir(directoryForBinaryImages)
    images_list = [i for i in files if i.endswith('.png')]
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForBinaryImages + image

        ImageFunctions.createCSVOnlyForTrafficLights(pathImage, 0, csvNumber, "Green")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    imageNumber = 0
    counterGoodPatches = 0
    totalNumberOfGoodPatches = 0
    for x in range(0, limitImages):

        listOfPatches = []
        listOfPositions = []
        listOfOnlyTrafficLights = []

        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)
        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\OnlyTrafficLightsCSV\\CSVwithOnlyTrafficLights_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfOnlyTrafficLights.append(row)

        pathImage = f"C:\\Users\\Marius\Desktop\\LicentaProiect\\GreenTL\\Images\\Image_{imageNumber:04}.png"

        (fd, hogImage) = ImageFunctions.createHOGHistogram(pathImage)
        dimensionForEachHistogram = 9
        myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                         for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

        valueFromPatchCell = []
        row_list = [[]]
        dimensionCSV = len(listOfPatches)

        valueFromPatchCell.append('Position')
        for i in range(0, 234):
            valueFromPatchCell.append(i)
        valueFromPatchCell.append('Clasa')

        row_list[0] = valueFromPatchCell

        numberList = 0
        counterGoodPatches = 0
        for i in range(0, dimensionCSV):
            OK = 0
            valueToPutInList = []

            checkerList = listOfOnlyTrafficLights[i]
            for valueFromList in checkerList:
                if int(valueFromList) == 255:
                    OK = 1

            if OK == 1:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))
                # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
                # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
                # valueToPutInList.append(int(max(arrayOfBinaryPatches)))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = int(arrayOfBinaryPatches[y])
                    valueToPutInList.append(value)

                valueToPutInList.append(1)
                counterGoodPatches += 1
            elif OK == 0:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))
                # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
                # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
                # valueToPutInList.append(int(max(arrayOfBinaryPatches)))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = int(arrayOfBinaryPatches[y])
                    valueToPutInList.append(value)

                valueToPutInList.append(0)

            row_list.append(valueToPutInList)
            numberList += 1

        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\finalTrainCSVs\\FinalCSVForTrain_{csvNumber:04}.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        csvNumber += 1
        imageNumber += 1

        print("Valid patches for ", pathImage, " --------- ", counterGoodPatches)
        totalNumberOfGoodPatches += counterGoodPatches

    print("Total number of patches with class = 1: ", totalNumberOfGoodPatches)

    pathTrain = "C:/Users/Marius/Desktop/LicentaProiect/GreenTL/finalTrainCSVs"

    all_files = glob.glob(os.path.join(pathTrain, "FinalCSVForTrain_*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)

    df_merged.to_csv("DatasetForAllImages_Green.csv", index=False)


def FinalMethodToCreateCSVForClassificationOfRedTrafficLight(limitImages):
    global directoryForTrainImages, directoryForBinaryImages, pathImage, listOfTrafficLightForCurrentImage, listOfTrafficLightForCurrentImage, pathTrain, OK

    directoryForTrainImages = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\RedTL\\Images\\'
    directoryForBinaryImages = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\TrafficLightsBinary\\'

    files = listdir(directoryForTrainImages)
    images_list = [i for i in files if i.endswith('.png')]

    ImageFunctions.createImageWithTrafficLightsFromPointsCoordinates(limitImages, "Red")

    csvNumber = 0
    countImages = 0
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForTrainImages + image

        ImageFunctions.createCSVWithPatches_Long(pathImage, 0, csvNumber, "Red")
        ImageFunctions.createCSVWithPosition(pathImage, 0, csvNumber, "Red")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    countImages = 0

    files = listdir(directoryForBinaryImages)
    images_list = [i for i in files if i.endswith('.png')]
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForBinaryImages + image

        ImageFunctions.createCSVOnlyForTrafficLights(pathImage, 0, csvNumber, "Red")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    imageNumber = 0
    counterGoodPatches = 0
    totalNumberOfGoodPatches = 0
    for x in range(0, limitImages):

        listOfPatches = []
        listOfPositions = []
        listOfOnlyTrafficLights = []

        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\RedTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)
        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\RedTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\OnlyTrafficLightsCSV\\CSVwithOnlyTrafficLights_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfOnlyTrafficLights.append(row)

        pathImage = f"C:\\Users\\Marius\Desktop\\LicentaProiect\\RedTL\\Images\\Image_{imageNumber:04}.png"

        (fd, hogImage) = ImageFunctions.createHOGHistogram(pathImage)
        dimensionForEachHistogram = 9
        myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                         for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

        valueFromPatchCell = []
        row_list = [[]]
        dimensionCSV = len(listOfPatches)

        valueFromPatchCell.append('Position')
        for i in range(0, 234):
            valueFromPatchCell.append(i)
        valueFromPatchCell.append('Clasa')

        row_list[0] = valueFromPatchCell

        numberList = 0
        counterGoodPatches = 0
        for i in range(0, dimensionCSV):
            OK = 0
            valueToPutInList = []

            checkerList = listOfOnlyTrafficLights[i]
            for valueFromList in checkerList:
                if int(valueFromList) == 255:
                    OK = 1

            if OK == 1:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))
                # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
                # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
                # valueToPutInList.append(int(max(arrayOfBinaryPatches)))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = int(arrayOfBinaryPatches[y])
                    valueToPutInList.append(value)

                valueToPutInList.append(1)
                counterGoodPatches += 1
            elif OK == 0:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))
                # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
                # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
                # valueToPutInList.append(int(max(arrayOfBinaryPatches)))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = int(arrayOfBinaryPatches[y])
                    valueToPutInList.append(value)

                valueToPutInList.append(0)

            row_list.append(valueToPutInList)
            numberList += 1

        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\finalTrainCSVs\\FinalCSVForTrain_{csvNumber:04}.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        csvNumber += 1
        imageNumber += 1

        print("Valid patches for ", pathImage, " --------- ", counterGoodPatches)
        totalNumberOfGoodPatches += counterGoodPatches

    print("Total number of patches with class = 1: ", totalNumberOfGoodPatches)

    pathTrain = "C:/Users/Marius/Desktop/LicentaProiect/RedTL/finalTrainCSVs"

    all_files = glob.glob(os.path.join(pathTrain, "FinalCSVForTrain_*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)

    df_merged.to_csv("DatasetForAllImages_Red.csv", index=False)


def FinalMethodToCreateCSVForClassificationOfYellowTrafficLight(limitImages):
    global directoryForTrainImages, directoryForBinaryImages, pathImage, listOfTrafficLightForCurrentImage, listOfTrafficLightForCurrentImage, pathTrain, OK

    directoryForTrainImages = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\YellowTL\\Images\\'
    directoryForBinaryImages = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\TrafficLightsBinary\\'

    files = listdir(directoryForTrainImages)
    images_list = [i for i in files if i.endswith('.png')]

    ImageFunctions.createImageWithTrafficLightsFromPointsCoordinates(limitImages, "Yellow")

    csvNumber = 0
    countImages = 0
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForTrainImages + image

        ImageFunctions.createCSVWithPatches_Long(pathImage, 0, csvNumber, "Yellow")
        ImageFunctions.createCSVWithPosition(pathImage, 0, csvNumber, "Yellow")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    countImages = 0

    files = listdir(directoryForBinaryImages)
    images_list = [i for i in files if i.endswith('.png')]
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForBinaryImages + image

        ImageFunctions.createCSVOnlyForTrafficLights(pathImage, 0, csvNumber, "Yellow")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    imageNumber = 0
    counterGoodPatches = 0
    totalNumberOfGoodPatches = 0
    for x in range(0, limitImages):

        listOfPatches = []
        listOfPositions = []
        listOfOnlyTrafficLights = []

        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\YellowTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)
        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\YellowTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\OnlyTrafficLightsCSV\\CSVwithOnlyTrafficLights_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfOnlyTrafficLights.append(row)

        pathImage = f"C:\\Users\\Marius\Desktop\\LicentaProiect\\YellowTL\\Images\\Image_{imageNumber:04}.png"

        (fd, hogImage) = ImageFunctions.createHOGHistogram(pathImage)
        dimensionForEachHistogram = 9
        myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                         for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

        valueFromPatchCell = []
        row_list = [[]]
        dimensionCSV = len(listOfPatches)

        valueFromPatchCell.append('Position')
        for i in range(0, 234):
            valueFromPatchCell.append(i)
        valueFromPatchCell.append('Clasa')

        row_list[0] = valueFromPatchCell

        numberList = 0
        counterGoodPatches = 0
        for i in range(0, dimensionCSV):
            OK = 0
            valueToPutInList = []

            checkerList = listOfOnlyTrafficLights[i]
            for valueFromList in checkerList:
                if int(valueFromList) == 255:
                    OK = 1

            if OK == 1:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = int(arrayOfBinaryPatches[y])
                    valueToPutInList.append(value)

                valueToPutInList.append(1)
                counterGoodPatches += 1
            elif OK == 0:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = int(arrayOfBinaryPatches[y])
                    valueToPutInList.append(value)

                valueToPutInList.append(0)

            row_list.append(valueToPutInList)
            numberList += 1

        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\finalTrainCSVs\\FinalCSVForTrain_{csvNumber:04}.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        csvNumber += 1
        imageNumber += 1

        print("Valid patches for ", pathImage, " --------- ", counterGoodPatches)
        totalNumberOfGoodPatches += counterGoodPatches

    print("Total number of patches with class = 1: ", totalNumberOfGoodPatches)

    pathTrain = "C:/Users/Marius/Desktop/LicentaProiect/YellowTL/finalTrainCSVs"

    all_files = glob.glob(os.path.join(pathTrain, "FinalCSVForTrain_*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)

    df_merged.to_csv("DatasetForAllImages_Yellow.csv", index=False)


def FinalMethodToCreateCSVForClassificationOfAllTrafficLights(limitImages):
    global directoryForTrainImages, directoryForBinaryImages, pathImage, listOfTrafficLightForCurrentImage, listOfTrafficLightForCurrentImage, pathTrain, OK

    directoryForTrainImages = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\AllTL\\Images\\'
    directoryForBinaryImages = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\TrafficLightsBinary\\'

    files = listdir(directoryForTrainImages)
    images_list = [i for i in files if i.endswith('.png')]

    ImageFunctions.createImageWithTrafficLightsFromPointsCoordinates(limitImages, "All")

    csvNumber = 0
    countImages = 0
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForTrainImages + image

        ImageFunctions.createCSVWithPatches_Long(pathImage, 0, csvNumber, "All")
        ImageFunctions.createCSVWithPosition(pathImage, 0, csvNumber, "All")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    countImages = 0

    files = listdir(directoryForBinaryImages)
    images_list = [i for i in files if i.endswith('.png')]
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForBinaryImages + image

        ImageFunctions.createCSVOnlyForTrafficLights(pathImage, 0, csvNumber, "All")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    imageNumber = 0
    totalNumberOfGoodPatches = 0
    totalOfRedPatches = 0
    totalOfGreenPatches = 0
    totalOfYellowPatches = 0
    for x in range(0, limitImages):

        listOfPatches = []
        listOfPositions = []
        listOfOnlyTrafficLights = []

        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\AllTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)
        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\AllTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\OnlyTrafficLightsCSV\\CSVwithOnlyTrafficLights_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfOnlyTrafficLights.append(row)

        pathImage = f"C:\\Users\\Marius\Desktop\\LicentaProiect\\AllTL\\Images\\Image_{imageNumber:04}.png"

        (fd, hogImage) = ImageFunctions.createHOGHistogram(pathImage)
        dimensionForEachHistogram = 9
        myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                         for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

        listOfTrafficLightForCurrentImage = extractSpecificBox(pathImage, 0, "All")

        validPatch = 0
        validPatches = []
        for i in range(0, len(listOfPositions)):
            currentListOfPositions = listOfPositions[i]

            for j in range(0, len(listOfTrafficLightForCurrentImage)):
                trafficLight = listOfTrafficLightForCurrentImage[j]

                if trafficLight.label == "Green" or trafficLight.label == "GreenRight" or trafficLight.label == "GreenLeft":
                    x_max = trafficLight.x_max
                    x_min = trafficLight.x_min
                    y_max = trafficLight.y_max
                    y_min = trafficLight.y_min

                    for p in range(0, len(currentListOfPositions)):
                        listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                        positionX = int(listOfPositionNumbers[0])
                        positionY = int(listOfPositionNumbers[1])

                        if x_min < positionY < x_max and y_min < positionX < y_max:
                            validPatch += 1
                            validPatches.append([i, "Green"])
                            break
                elif trafficLight.label == "Red" or trafficLight.label == "RedRight" or trafficLight.label == "RedLeft":
                    x_max = trafficLight.x_max
                    x_min = trafficLight.x_min
                    y_max = trafficLight.y_max
                    y_min = trafficLight.y_min

                    for p in range(0, len(currentListOfPositions)):
                        listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                        positionX = int(listOfPositionNumbers[0])
                        positionY = int(listOfPositionNumbers[1])

                        if x_min < positionY < x_max and y_min < positionX < y_max:
                            validPatch += 1
                            validPatches.append([i, "Red"])
                            break
                elif trafficLight.label == "Yellow" or trafficLight.label == "YellowRight" or trafficLight.label == "YellowLeft":
                    x_max = trafficLight.x_max
                    x_min = trafficLight.x_min
                    y_max = trafficLight.y_max
                    y_min = trafficLight.y_min

                    for p in range(0, len(currentListOfPositions)):
                        listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                        positionX = int(listOfPositionNumbers[0])
                        positionY = int(listOfPositionNumbers[1])

                        if x_min < positionY < x_max and y_min < positionX < y_max:
                            validPatch += 1
                            validPatches.append([i, "Yellow"])
                            break

        valueFromPatchCell = []
        row_list = [[]]
        dimensionCSV = len(listOfPatches)

        valueFromPatchCell.append('Position')
        for i in range(0, 234):
            valueFromPatchCell.append(i)
        valueFromPatchCell.append('Clasa')

        row_list[0] = valueFromPatchCell

        numberList = 0
        counter_1 = 0
        counter_2 = 0
        counter_3 = 0
        for i in range(0, dimensionCSV):
            OK = 0
            valueToPutInList = []

            checkerList = listOfOnlyTrafficLights[i]
            for valueFromList in checkerList:
                if int(valueFromList) == 255:
                    OK = 1

            if OK == 1:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = int(arrayOfBinaryPatches[y])
                    valueToPutInList.append(value)

                for q in range(0, len(validPatches)):
                    patch = int(validPatches[q][0])
                    if i == patch:
                        typeOfTrafficLight = validPatches[q]
                        if typeOfTrafficLight[1] == "Red":
                            valueToPutInList.append(1)
                            counter_1 += 1
                            break
                        elif typeOfTrafficLight[1] == "Green":
                            valueToPutInList.append(2)
                            counter_2 += 1
                            break
                        elif typeOfTrafficLight[1] == "Yellow":
                            valueToPutInList.append(3)
                            counter_3 += 1
                            break
            elif OK == 0:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = int(arrayOfBinaryPatches[y])
                    valueToPutInList.append(value)

                valueToPutInList.append(0)

            row_list.append(valueToPutInList)
            numberList += 1

        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\finalTrainCSVs\\FinalCSVForTrain_{csvNumber:04}.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        csvNumber += 1
        imageNumber += 1

        print("Valid patches for ", pathImage, " --------- Red = ", counter_1, "--------- Green = ", counter_2, "--------- Yellow = ", counter_3)
        totalNumberOfGoodPatches += counter_1 + counter_2 + counter_3
        totalOfRedPatches += counter_1
        totalOfGreenPatches += counter_2
        totalOfYellowPatches += counter_3

    print("Total number of patches with class = 1: ", totalOfRedPatches)
    print("Total number of patches with class = 2: ", totalOfGreenPatches)
    print("Total number of patches with class = 3: ", totalOfYellowPatches)
    print("Total number of patches: ", totalNumberOfGoodPatches)

    pathTrain = "C:/Users/Marius/Desktop/LicentaProiect/AllTL/finalTrainCSVs"

    all_files = glob.glob(os.path.join(pathTrain, "FinalCSVForTrain_*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)

    df_merged.to_csv("DatasetForAllImages_ForAll.csv", index=False)


def FinalMethodToCreateCSVForClassificationOfAllTrafficLights_V2(limitImages):
    global directoryForTrainImages, directoryForBinaryImages, pathImage, listOfTrafficLightForCurrentImage, listOfTrafficLightForCurrentImage, pathTrain, OK

    directoryForTrainImages = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\AllTL\\Images\\'
    directoryForBinaryImages = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\TrafficLightsBinary\\'

    files = listdir(directoryForTrainImages)
    images_list = [i for i in files if i.endswith('.png')]

    ImageFunctions.createImageWithTrafficLightsFromPointsCoordinates_ForALL(limitImages)

    csvNumber = 0
    countImages = 0
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForTrainImages + image

        ImageFunctions.createCSVWithPatches_Long(pathImage, 0, csvNumber, "All")
        ImageFunctions.createCSVWithPosition(pathImage, 0, csvNumber, "All")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    countImages = 0

    files = listdir(directoryForBinaryImages)
    images_list = [i for i in files if i.endswith('.png')]
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForBinaryImages + image

        ImageFunctions.createCSVOnlyForTrafficLights(pathImage, 0, csvNumber, "All")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    imageNumber = 0
    totalNumberOfGoodPatches = 0
    totalOfRedPatches = 0
    totalOfGreenPatches = 0
    totalOfYellowPatches = 0
    for x in range(0, limitImages):

        listOfPatches = []
        listOfPositions = []
        listOfOnlyTrafficLights = []

        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\AllTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)
        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\AllTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\OnlyTrafficLightsCSV\\CSVwithOnlyTrafficLights_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfOnlyTrafficLights.append(row)

        pathImage = f"C:\\Users\\Marius\Desktop\\LicentaProiect\\AllTL\\Images\\Image_{imageNumber:04}.png"

        (fd, hogImage) = ImageFunctions.createHOGHistogram(pathImage)
        dimensionForEachHistogram = 9
        myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                         for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

        listOfTrafficLightForCurrentImage = extractSpecificBox(pathImage, 0, "All")

        validPatch = 0
        validPatches = []
        for i in range(0, len(listOfPositions)):
            currentListOfPositions = listOfPositions[i]

            for j in range(0, len(listOfTrafficLightForCurrentImage)):
                trafficLight = listOfTrafficLightForCurrentImage[j]

                if trafficLight.label == "Green" or trafficLight.label == "GreenRight" or trafficLight.label == "GreenLeft":
                    x_max = trafficLight.x_max
                    x_min = trafficLight.x_min
                    y_max = trafficLight.y_max
                    y_min = trafficLight.y_min

                    for p in range(0, len(currentListOfPositions)):
                        listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                        positionX = int(listOfPositionNumbers[0])
                        positionY = int(listOfPositionNumbers[1])

                        if x_min < positionY < x_max and y_min < positionX < y_max:
                            validPatch += 1
                            validPatches.append([i, "Green"])
                            break
                elif trafficLight.label == "Red" or trafficLight.label == "RedRight" or trafficLight.label == "RedLeft":
                    x_max = trafficLight.x_max
                    x_min = trafficLight.x_min
                    y_max = trafficLight.y_max
                    y_min = trafficLight.y_min

                    for p in range(0, len(currentListOfPositions)):
                        listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                        positionX = int(listOfPositionNumbers[0])
                        positionY = int(listOfPositionNumbers[1])

                        if x_min < positionY < x_max and y_min < positionX < y_max:
                            validPatch += 1
                            validPatches.append([i, "Red"])
                            break
                elif trafficLight.label == "Yellow" or trafficLight.label == "YellowRight" or trafficLight.label == "YellowLeft":
                    x_max = trafficLight.x_max
                    x_min = trafficLight.x_min
                    y_max = trafficLight.y_max
                    y_min = trafficLight.y_min

                    for p in range(0, len(currentListOfPositions)):
                        listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                        positionX = int(listOfPositionNumbers[0])
                        positionY = int(listOfPositionNumbers[1])

                        if x_min < positionY < x_max and y_min < positionX < y_max:
                            validPatch += 1
                            validPatches.append([i, "Yellow"])
                            break

        valueFromPatchCell = []
        row_list = [[]]
        dimensionCSV = len(listOfPatches)

        valueFromPatchCell.append('Position')
        for i in range(0, 234):
            valueFromPatchCell.append(i)
        valueFromPatchCell.append('Clasa')

        row_list[0] = valueFromPatchCell

        numberList = 0
        counter_1 = 0
        counter_2 = 0
        counter_3 = 0
        for i in range(0, dimensionCSV):
            OK = 0
            valueToPutInList = []

            checkerList = listOfOnlyTrafficLights[i]
            for valueFromList in checkerList:
                if int(valueFromList) == 255:
                    OK = 1

            if OK == 1:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = int(arrayOfBinaryPatches[y])
                    valueToPutInList.append(value)

                for q in range(0, len(validPatches)):
                    patch = int(validPatches[q][0])
                    if i == patch:
                        typeOfTrafficLight = validPatches[q]
                        if typeOfTrafficLight[1] == "Red":
                            valueToPutInList.append(1)
                            counter_1 += 1
                            break
                        elif typeOfTrafficLight[1] == "Green":
                            valueToPutInList.append(2)
                            counter_2 += 1
                            break
                        elif typeOfTrafficLight[1] == "Yellow":
                            valueToPutInList.append(3)
                            counter_3 += 1
                            break
            elif OK == 0:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = int(arrayOfBinaryPatches[y])
                    valueToPutInList.append(value)

                valueToPutInList.append(0)

            row_list.append(valueToPutInList)
            numberList += 1

        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\finalTrainCSVs\\FinalCSVForTrain_{csvNumber:04}.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        csvNumber += 1
        imageNumber += 1

        print("Valid patches for ", pathImage, " --------- Red = ", counter_1, "--------- Green = ", counter_2, "--------- Yellow = ", counter_3)
        totalNumberOfGoodPatches += counter_1 + counter_2 + counter_3
        totalOfRedPatches += counter_1
        totalOfGreenPatches += counter_2
        totalOfYellowPatches += counter_3

    print("Total number of patches with class = 1: ", totalOfRedPatches)
    print("Total number of patches with class = 2: ", totalOfGreenPatches)
    print("Total number of patches with class = 3: ", totalOfYellowPatches)
    print("Total number of patches: ", totalNumberOfGoodPatches)

    pathTrain = "C:/Users/Marius/Desktop/LicentaProiect/AllTL/finalTrainCSVs"

    all_files = glob.glob(os.path.join(pathTrain, "FinalCSVForTrain_*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)

    df_merged.to_csv("DatasetForAllImages_ForAll.csv", index=False)


def createCSVForClassificationOfTrafficLights_Red(limitImages):
    global directoryForTrainImages, directoryForBinaryImages, pathImage, listOfTrafficLightForCurrentImage, listOfTrafficLightForCurrentImage, pathTrain

    directoryForTrainImages = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\RedTL\\Images\\'
    directoryForBinaryImages = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\BinaryImages\\'

    files = listdir(directoryForTrainImages)
    images_list = [i for i in files if i.endswith('.png')]

    ImageFunctions.detectRedFromTrafficLights(limitImages)

    csvNumber = 0
    countImages = 0
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForTrainImages + image

        ImageFunctions.createCSVWithPatches_Long(pathImage, 0, csvNumber, "Red")
        ImageFunctions.createCSVWithPosition(pathImage, 0, csvNumber, "Red")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    countImages = 0

    files = listdir(directoryForBinaryImages)
    images_list = [i for i in files if i.endswith('.jpg')]
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForBinaryImages + image

        ImageFunctions.createCSVWithBinaryValues(pathImage, 0, csvNumber, "Red")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    imageNumber = 0
    totalValidPatches = 0
    totalValidPatchesAfterBinaryTriage = 0
    for x in range(0, limitImages):

        listOfPatches = []
        listOfPositions = []
        listOfBinaryPatches = []

        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\RedTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)
        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\RedTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\BinaryValuesCSV\\CSVwithBinaryValues_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfBinaryPatches.append(row)

        pathImage = f"C:\\Users\\Marius\Desktop\\LicentaProiect\\RedTL\\Images\\Image_{imageNumber:04}.png"

        (fd, hogImage) = ImageFunctions.createHOGHistogram(pathImage)
        dimensionForEachHistogram = 9
        myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                         for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

        listOfTrafficLightForCurrentImage = extractSpecificBox(pathImage, 0, "Red")

        validPatch = 0
        validPatches = []

        for i in range(0, len(listOfPositions)):
            currentListOfPositions = listOfPositions[i]

            okPatch = 0
            for j in range(0, len(listOfTrafficLightForCurrentImage)):
                trafficLight = listOfTrafficLightForCurrentImage[j]

                if trafficLight.label != "Red" and trafficLight.label != "RedRight" and trafficLight.label != "RedLeft":
                    continue
                else:
                    x_max = trafficLight.x_max
                    x_min = trafficLight.x_min
                    y_max = trafficLight.y_max
                    y_min = trafficLight.y_min

                    for p in range(0, len(currentListOfPositions)):
                        listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                        positionX = int(listOfPositionNumbers[0])
                        positionY = int(listOfPositionNumbers[1])

                        if x_min < positionY < x_max and y_min < positionX < y_max:
                            okPatch = 1
                            validPatch += 1
                            validPatches.append(i)
                            break
                if okPatch == 1:
                    break

        print("Valid patches before binary triage for ", pathImage, " --------- ", validPatch)
        totalValidPatches += validPatch

        valueFromPatchCell = []
        row_list = [[]]
        dimensionCSV = len(listOfPatches)

        valueFromPatchCell.append('Position')
        for i in range(0, 234):
            valueFromPatchCell.append(i)
        valueFromPatchCell.append('Clasa')

        row_list[0] = valueFromPatchCell

        numberList = 0
        validPatchAfterBinaryTriage = 0
        for i in range(0, dimensionCSV):
            valueToPutInList = []
            counterOfWhitePixels = 0
            if validPatches.__contains__(i):
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfBinaryPatches[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                # print(myListFromHOG[numberList])
                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = arrayOfBinaryPatches[y]
                    if 100 <= int(value) <= 255:
                        counterOfWhitePixels += 1

                if counterOfWhitePixels >= 1:
                    for y in range(0, len(arrayOfBinaryPatches)):
                        value = 255
                        valueToPutInList.append(value)
                    valueToPutInList.append(1)
                else:
                    for y in range(0, len(arrayOfBinaryPatches)):
                        value = arrayOfBinaryPatches[y]
                        valueToPutInList.append(value)
                    valueToPutInList.append(0)
            else:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfBinaryPatches[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                # print(myListFromHOG[numberList])
                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = 0
                    valueToPutInList.append(value)

                valueToPutInList.append(0)

            row_list.append(valueToPutInList)
            numberList += 1
            totalValidPatchesAfterBinaryTriage += validPatchAfterBinaryTriage
            validPatchAfterBinaryTriage = 0

        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\finalTrainCSVs\\FinalCSVForTrain_{csvNumber:04}.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        csvNumber += 1
        imageNumber += 1

    print("\n")
    print("Total number of valid patches ------- without binary triage: ", totalValidPatches)
    print("Total number of valid patches ------- with binary triage: ", totalValidPatchesAfterBinaryTriage)

    pathTrain = "C:/Users/Marius/Desktop/LicentaProiect/RedTL/finalTrainCSVs"

    all_files = glob.glob(os.path.join(pathTrain, "FinalCSVForTrain_*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)

    df_merged.to_csv("DatasetForAllImages_Red.csv", index=False)


def createCSVForClassificationOfTrafficLights_Yellow(limitImages):
    global directoryForTrainImages, directoryForBinaryImages, pathImage, listOfTrafficLightForCurrentImage, listOfTrafficLightForCurrentImage, pathTrain

    directoryForTrainImages = 'C:\\Users\\Marius\Desktop\\LicentaProiect\\YellowTL\\Images\\'
    directoryForBinaryImages = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\BinaryImages\\'

    files = listdir(directoryForTrainImages)
    images_list = [i for i in files if i.endswith('.png')]

    ImageFunctions.detectYellowFromTrafficLights(limitImages)

    csvNumber = 0
    countImages = 0
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForTrainImages + image

        ImageFunctions.createCSVWithPatches_Long(pathImage, 0, csvNumber, "Yellow")
        ImageFunctions.createCSVWithPosition(pathImage, 0, csvNumber, "Yellow")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    countImages = 0

    files = listdir(directoryForBinaryImages)
    images_list = [i for i in files if i.endswith('.jpg')]
    for idx, image in enumerate(images_list):
        if countImages == limitImages:
            break
        pathImage = directoryForBinaryImages + image

        ImageFunctions.createCSVWithBinaryValues(pathImage, 0, csvNumber, "Yellow")

        countImages += 1
        csvNumber += 1

    csvNumber = 0
    imageNumber = 0
    totalValidPatches = 0
    totalValidPatchesAfterBinaryTriage = 0
    for x in range(0, limitImages):

        listOfPatches = []
        listOfPositions = []
        listOfBinaryPatches = []

        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\YellowTL\\trainCSVs\\CSVwithHSVPatches_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)
        with open(
                f"C:\\Users\\Marius\Desktop\\LicentaProiect\\YellowTL\\positionCSVs\\CSVwithPosition_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\BinaryValuesCSV\\CSVwithBinaryValues_{csvNumber:04}.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfBinaryPatches.append(row)

        pathImage = f"C:\\Users\\Marius\Desktop\\LicentaProiect\\YellowTL\\Images\\Image_{imageNumber:04}.png"

        (fd, hogImage) = ImageFunctions.createHOGHistogram(pathImage)
        dimensionForEachHistogram = 9
        myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                         for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

        listOfTrafficLightForCurrentImage = extractSpecificBox(pathImage, 0, "Yellow")

        validPatch = 0
        validPatches = []

        for i in range(0, len(listOfPositions)):
            currentListOfPositions = listOfPositions[i]

            okPatch = 0
            for j in range(0, len(listOfTrafficLightForCurrentImage)):
                trafficLight = listOfTrafficLightForCurrentImage[j]

                if trafficLight.label != "Yellow" and trafficLight.label != "YellowRight" and trafficLight.label != "YellowLeft":
                    continue
                else:
                    x_max = trafficLight.x_max
                    x_min = trafficLight.x_min
                    y_max = trafficLight.y_max
                    y_min = trafficLight.y_min

                    for p in range(0, len(currentListOfPositions)):
                        listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                        positionX = int(listOfPositionNumbers[0])
                        positionY = int(listOfPositionNumbers[1])

                        if x_min < positionY < x_max and y_min < positionX < y_max:
                            okPatch = 1
                            validPatch += 1
                            validPatches.append(i)
                            break
                if okPatch == 1:
                    break

        print("Valid patches before binary triage for ", pathImage, " --------- ", validPatch)
        totalValidPatches += validPatch

        valueFromPatchCell = []
        row_list = [[]]
        dimensionCSV = len(listOfPatches)

        valueFromPatchCell.append('Position')
        for i in range(0, 234):
            valueFromPatchCell.append(i)
        valueFromPatchCell.append('Clasa')

        row_list[0] = valueFromPatchCell

        numberList = 0
        validPatchAfterBinaryTriage = 0
        for i in range(0, dimensionCSV):
            valueToPutInList = []
            counterOfWhitePixels = 0
            if validPatches.__contains__(i):
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfBinaryPatches[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                # print(myListFromHOG[numberList])
                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = arrayOfBinaryPatches[y]
                    valueToPutInList.append(value)

                    if 1 <= int(value) <= 255:
                        counterOfWhitePixels += 1

                if counterOfWhitePixels >= 1:
                    validPatchAfterBinaryTriage += 1
                    valueToPutInList.append(1)
                else:
                    valueToPutInList.append(0)
            else:
                valueToPutInList.append(i)

                arrayOfPatches = listOfPatches[i]
                arrayOfBinaryPatches = listOfBinaryPatches[i]

                for y in range(0, len(arrayOfPatches)):
                    value = arrayOfPatches[y]
                    valueToPutInList.append(value)

                # print(myListFromHOG[numberList])
                valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

                for y in range(0, len(arrayOfBinaryPatches)):
                    value = arrayOfBinaryPatches[y]
                    valueToPutInList.append(value)

                valueToPutInList.append(0)

            row_list.append(valueToPutInList)
            numberList += 1
            totalValidPatchesAfterBinaryTriage += validPatchAfterBinaryTriage
            validPatchAfterBinaryTriage = 0

        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\finalTrainCSVs\\FinalCSVForTrain_{csvNumber:04}.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)

        csvNumber += 1
        imageNumber += 1

    print("\n")
    print("Total number of valid patches ------- without binary triage: ", totalValidPatches)
    print("Total number of valid patches ------- with binary triage: ", totalValidPatchesAfterBinaryTriage)

    pathTrain = "C:/Users/Marius/Desktop/LicentaProiect/YellowTL/finalTrainCSVs"

    all_files = glob.glob(os.path.join(pathTrain, "FinalCSVForTrain_*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)

    df_merged.to_csv("DatasetForAllImages_Yellow.csv", index=False)


def FinalMethodToCreateCSVForClassificationOfGreenTrafficLight_ForTestImage(imagePath, checker):
    global listOfPositions, listOfOnlyTrafficLights, listOfPatches
    if checker == 0:
        ImageFunctions.detectTrafficLights_ForTestImage(imagePath, "Green", 0)
        ImageFunctions.createCSVWithPatches_Long(imagePath, 1, 0, "Green")
        ImageFunctions.createCSVWithPosition(imagePath, 1, 0, "Green")
    elif checker == 1:
        ImageFunctions.detectTrafficLights_ForTestImage(imagePath, "Green", 1)
        ImageFunctions.createCSVWithPatches_Long(imagePath, 2, 0, "Green")
        ImageFunctions.createCSVWithPosition(imagePath, 2, 0, "Green")

    if checker == 0:
        imagePathForBinary = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight.jpg'
        ImageFunctions.createCSVOnlyForTrafficLights(imagePathForBinary, 1, 0, "Green")
    elif checker == 1:
        imagePathForBinary = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight_Green.jpg'
        ImageFunctions.createCSVOnlyForTrafficLights(imagePathForBinary, 2, 0, "Green")

    if checker == 0:
        listOfPatches = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\TestCSVs\\TestPatches.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)

        listOfPositions = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\TestCSVs\\TestPosition.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])

        listOfOnlyTrafficLights = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfOnlyTrafficLights.append(row)
    elif checker == 1:
        listOfPatches = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPatches_Green.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)

        listOfPositions = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition_Green.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])

        listOfOnlyTrafficLights = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights_Green.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfOnlyTrafficLights.append(row)

    valueFromPatchCell = []
    row_list = [[]]
    dimensionCSV = len(listOfPatches)

    valueFromPatchCell.append('Position')
    for i in range(0, 234):
        valueFromPatchCell.append(i)
    valueFromPatchCell.append('Clasa')

    row_list[0] = valueFromPatchCell

    (fd, hogImage) = ImageFunctions.createHOGHistogram(imagePath)

    dimensionForEachHistogram = 9
    myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                     for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

    numberList = 0
    for i in range(0, dimensionCSV):
        OK = 0
        valueToPutInList = []

        checkerList = listOfOnlyTrafficLights[i]
        for valueFromList in checkerList:
            if int(valueFromList) == 255:
                OK = 1

        if OK == 1:
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = int(arrayOfBinaryPatches[y])
                valueToPutInList.append(value)

            valueToPutInList.append(1)
        elif OK == 0:
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = int(arrayOfBinaryPatches[y])
                valueToPutInList.append(value)

            valueToPutInList.append(0)

        row_list.append(valueToPutInList)
        numberList += 1

    if checker == 0:
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\TestCSVs\\FinalCSVForTestImage.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)
    elif checker == 1:
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\FinalCSVForTestImage_Green.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)


def FinalMethodToCreateCSVForClassificationOfRedTrafficLight_ForTestImage(imagePath, checker):
    global listOfPatches, listOfOnlyTrafficLights, listOfPositions

    if checker == 0:
        ImageFunctions.detectTrafficLights_ForTestImage(imagePath, "Red", 0)
        ImageFunctions.createCSVWithPatches_Long(imagePath, 1, 0, "Red")
        ImageFunctions.createCSVWithPosition(imagePath, 1, 0, "Red")
    elif checker == 1:
        ImageFunctions.detectTrafficLights_ForTestImage(imagePath, "Red", 1)
        ImageFunctions.createCSVWithPatches_Long(imagePath, 2, 0, "Red")
        ImageFunctions.createCSVWithPosition(imagePath, 2, 0, "Red")

    if checker == 0:
        imagePathForBinary = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight.jpg'
        ImageFunctions.createCSVOnlyForTrafficLights(imagePathForBinary, 1, 0, "Red")
    elif checker == 1:
        imagePathForBinary = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight_Red.jpg'
        ImageFunctions.createCSVOnlyForTrafficLights(imagePathForBinary, 2, 0, "Red")

    if checker == 0:
        listOfPatches = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestPatches.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)

        listOfPositions = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestPosition.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])

        listOfOnlyTrafficLights = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfOnlyTrafficLights.append(row)
    elif checker == 1:
        listOfPatches = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPatches_Red.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)

        listOfPositions = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition_Red.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])

        listOfOnlyTrafficLights = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights_Red.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfOnlyTrafficLights.append(row)

    valueFromPatchCell = []
    row_list = [[]]
    dimensionCSV = len(listOfPatches)

    valueFromPatchCell.append('Position')
    for i in range(0, 234):
        valueFromPatchCell.append(i)
    valueFromPatchCell.append('Clasa')

    row_list[0] = valueFromPatchCell

    (fd, hogImage) = ImageFunctions.createHOGHistogram(imagePath)

    dimensionForEachHistogram = 9
    myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                     for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

    numberList = 0
    for i in range(0, dimensionCSV):
        OK = 0
        valueToPutInList = []

        checkerList = listOfOnlyTrafficLights[i]
        for valueFromList in checkerList:
            if int(valueFromList) == 255:
                OK = 1

        if OK == 1:
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = int(arrayOfBinaryPatches[y])
                valueToPutInList.append(value)

            valueToPutInList.append(1)
        elif OK == 0:
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = int(arrayOfBinaryPatches[y])
                valueToPutInList.append(value)

            valueToPutInList.append(0)

        row_list.append(valueToPutInList)
        numberList += 1

    if checker == 0:
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\FinalCSVForTestImage.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)
    elif checker == 1:
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\FinalCSVForTestImage_Red.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)


def FinalMethodToCreateCSVForClassificationOfYellowTrafficLight_ForTestImage(imagePath, checker):
    global listOfPatches
    if checker == 0:
        ImageFunctions.detectTrafficLights_ForTestImage(imagePath, "Yellow", 0)
        ImageFunctions.createCSVWithPatches_Long(imagePath, 1, 0, "Yellow")
        ImageFunctions.createCSVWithPosition(imagePath, 1, 0, "Yellow")
    elif checker == 1:
        ImageFunctions.detectTrafficLights_ForTestImage(imagePath, "Yellow", 1)
        ImageFunctions.createCSVWithPatches_Long(imagePath, 2, 0, "Yellow")
        ImageFunctions.createCSVWithPosition(imagePath, 2, 0, "Yellow")

    if checker == 0:
        imagePathForBinary = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight.jpg'
        ImageFunctions.createCSVOnlyForTrafficLights(imagePathForBinary, 1, 0, "Green")
    elif checker == 1:
        imagePathForBinary = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight_Yellow.jpg'
        ImageFunctions.createCSVOnlyForTrafficLights(imagePathForBinary, 2, 0, "Yellow")

    imagePathForBinary = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight.jpg'
    ImageFunctions.createCSVOnlyForTrafficLights(imagePathForBinary, 1, 0, "Yellow")

    if checker == 0:
        listOfPatches = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\TestPatches.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)

        listOfPositions = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\TestPosition.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])

        listOfOnlyTrafficLights = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfOnlyTrafficLights.append(row)
    elif checker == 1:
        listOfPatches = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPatches_Yellow.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPatches.append(row)

        listOfPositions = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition_Yellow.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row[1:])

        listOfOnlyTrafficLights = []
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights_Yellow.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfOnlyTrafficLights.append(row)

    valueFromPatchCell = []
    row_list = [[]]
    dimensionCSV = len(listOfPatches)

    valueFromPatchCell.append('Position')
    for i in range(0, 234):
        valueFromPatchCell.append(i)
    valueFromPatchCell.append('Clasa')

    row_list[0] = valueFromPatchCell

    (fd, hogImage) = ImageFunctions.createHOGHistogram(imagePath)

    dimensionForEachHistogram = 9
    myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                     for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

    numberList = 0
    for i in range(0, dimensionCSV):
        OK = 0
        valueToPutInList = []

        checkerList = listOfOnlyTrafficLights[i]
        for valueFromList in checkerList:
            if int(valueFromList) == 255:
                OK = 1

        if OK == 1:
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = int(arrayOfBinaryPatches[y])
                valueToPutInList.append(value)

            valueToPutInList.append(1)
        elif OK == 0:
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))
            # valueToPutInList.append(int(max(arrayOfBinaryPatches)))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = int(arrayOfBinaryPatches[y])
                valueToPutInList.append(value)

            valueToPutInList.append(0)

        row_list.append(valueToPutInList)
        numberList += 1

    if checker == 0:
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\FinalCSVForTestImage.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)
    elif checker == 1:
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\FinalCSVForTestImage_Yellow.csv",
                'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)


def FinalMethodToCreateCSVForClassificationOfAllTrafficLights_ForTestImage(imagePath):
    ImageFunctions.detectTrafficLights_ForTestImage(imagePath, "All")

    ImageFunctions.createCSVWithPatches_Long(imagePath, 1, 0, "All")
    ImageFunctions.createCSVWithPosition(imagePath, 1, 0, "All")

    imagePathForBinary = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\OnlyTrafficLightImage\\OnlyTrafficLight.jpg'
    ImageFunctions.createCSVOnlyForTrafficLights(imagePathForBinary, 1, 0, "All")

    listOfPatches = []
    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPatches.csv",
            'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfPatches.append(row)

    listOfPositions = []
    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition.csv",
            'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfPositions.append(row[1:])

    listOfOnlyTrafficLights = []
    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\CSVwithOnlyTrafficLights.csv",
            'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfOnlyTrafficLights.append(row)

    valueFromPatchCell = []
    row_list = [[]]
    dimensionCSV = len(listOfPatches)

    valueFromPatchCell.append('Position')
    for i in range(0, 234):
        valueFromPatchCell.append(i)
    valueFromPatchCell.append('Clasa')

    row_list[0] = valueFromPatchCell

    (fd, hogImage) = ImageFunctions.createHOGHistogram(imagePath)

    dimensionForEachHistogram = 9
    myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                     for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

    listOfTrafficLightForCurrentImage = extractSpecificBox(imagePath, 1, "All")

    validPatch = 0
    validPatches = []
    for i in range(0, len(listOfPositions)):
        currentListOfPositions = listOfPositions[i]

        for j in range(0, len(listOfTrafficLightForCurrentImage)):
            trafficLight = listOfTrafficLightForCurrentImage[j]

            if trafficLight.label == "Green" or trafficLight.label == "GreenRight" or trafficLight.label == "GreenLeft":
                x_max = trafficLight.x_max
                x_min = trafficLight.x_min
                y_max = trafficLight.y_max
                y_min = trafficLight.y_min

                for p in range(0, len(currentListOfPositions)):
                    listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                    positionX = int(listOfPositionNumbers[0])
                    positionY = int(listOfPositionNumbers[1])

                    if x_min < positionY < x_max and y_min < positionX < y_max:
                        validPatch += 1
                        validPatches.append([i, "Green"])
                        break
            elif trafficLight.label == "Red" or trafficLight.label == "RedRight" or trafficLight.label == "RedLeft":
                x_max = trafficLight.x_max
                x_min = trafficLight.x_min
                y_max = trafficLight.y_max
                y_min = trafficLight.y_min

                for p in range(0, len(currentListOfPositions)):
                    listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                    positionX = int(listOfPositionNumbers[0])
                    positionY = int(listOfPositionNumbers[1])

                    if x_min < positionY < x_max and y_min < positionX < y_max:
                        validPatch += 1
                        validPatches.append([i, "Red"])
                        break
            elif trafficLight.label == "Yellow" or trafficLight.label == "YellowRight" or trafficLight.label == "YellowLeft":
                x_max = trafficLight.x_max
                x_min = trafficLight.x_min
                y_max = trafficLight.y_max
                y_min = trafficLight.y_min

                for p in range(0, len(currentListOfPositions)):
                    listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                    positionX = int(listOfPositionNumbers[0])
                    positionY = int(listOfPositionNumbers[1])

                    if x_min < positionY < x_max and y_min < positionX < y_max:
                        validPatch += 1
                        validPatches.append([i, "Yellow"])
                        break

    numberList = 0
    for i in range(0, dimensionCSV):
        OK = 0
        valueToPutInList = []

        checkerList = listOfOnlyTrafficLights[i]
        for valueFromList in checkerList:
            if int(valueFromList) == 255:
                OK = 1

        if OK == 1:
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = int(arrayOfBinaryPatches[y])
                valueToPutInList.append(value)

            for q in range(0, len(validPatches)):
                patch = int(validPatches[q][0])
                if i == patch:
                    typeOfTrafficLight = validPatches[q]
                    if typeOfTrafficLight[1] == "Red":
                        valueToPutInList.append(1)
                        break
                    elif typeOfTrafficLight[1] == "Green":
                        valueToPutInList.append(2)
                        break
                    elif typeOfTrafficLight[1] == "Yellow":
                        valueToPutInList.append(3)
                        break
        elif OK == 0:
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfOnlyTrafficLights[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = int(arrayOfBinaryPatches[y])
                valueToPutInList.append(value)

            valueToPutInList.append(0)

        row_list.append(valueToPutInList)
        numberList += 1

    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\FinalCSVForTestImage.csv",
            'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def createCSVForTestImage_ForRedTrafficLights(imagePath):
    ImageFunctions.detectRedFromTrafficLights_ForTestImage(imagePath)

    ImageFunctions.createCSVWithPatches_Long(imagePath, 1, 0, "Red")
    ImageFunctions.createCSVWithPosition(imagePath, 1, 0, "Red")

    imagePathForBinary = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\BinaryImages\\BinaryImageTest.jpg'
    ImageFunctions.createCSVWithBinaryValues(imagePathForBinary, 1, 0, "Red")

    listOfPatches = []
    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestPatches.csv",
            'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfPatches.append(row)

    listOfPositions = []
    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestPosition.csv",
            'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfPositions.append(row[1:])

    listOfBinaryPatches = []
    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestBinaryPatches.csv",
            'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfBinaryPatches.append(row)

    listOfTrafficLightForCurrentImage = extractSpecificBox(imagePath, 1, "Red")
    validPatch = 0
    validPatches = []

    for i in range(0, len(listOfPositions)):
        currentListOfPositions = listOfPositions[i]

        okPatch = 0
        for j in range(0, len(listOfTrafficLightForCurrentImage)):
            trafficLight = listOfTrafficLightForCurrentImage[j]
            if trafficLight.label != "Red" and trafficLight.label != "RedRight" and trafficLight.label != "RedLeft":
                continue
            else:
                x_max = trafficLight.x_max
                x_min = trafficLight.x_min
                y_max = trafficLight.y_max
                y_min = trafficLight.y_min

                for p in range(0, len(currentListOfPositions)):
                    listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                    positionX = int(listOfPositionNumbers[0])
                    positionY = int(listOfPositionNumbers[1])

                    if x_min < positionY < x_max and y_min < positionX < y_max:
                        okPatch = 1
                        validPatch += 1
                        validPatches.append(i)
                        break
            if okPatch == 1:
                break

    valueFromPatchCell = []
    row_list = [[]]
    dimensionCSV = len(listOfPatches)

    valueFromPatchCell.append('Position')
    for i in range(0, 234):
        valueFromPatchCell.append(i)
    valueFromPatchCell.append('Clasa')

    row_list[0] = valueFromPatchCell

    (fd, hogImage) = ImageFunctions.createHOGHistogram(imagePath)

    dimensionForEachHistogram = 9
    myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                     for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

    numberList = 0
    for i in range(0, dimensionCSV):
        valueToPutInList = []
        counterOfWhitePixels = 0
        if validPatches.__contains__(i):
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfBinaryPatches[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            # print(myListFromHOG[numberList])
            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = arrayOfBinaryPatches[y]
                if 100 <= int(value) <= 255:
                    counterOfWhitePixels += 1

            if counterOfWhitePixels >= 1:
                for y in range(0, len(arrayOfBinaryPatches)):
                    value = 255
                    valueToPutInList.append(value)
                valueToPutInList.append(1)
            else:
                for y in range(0, len(arrayOfBinaryPatches)):
                    value = arrayOfBinaryPatches[y]
                    valueToPutInList.append(value)
                valueToPutInList.append(0)
        else:
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfBinaryPatches[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            # print(myListFromHOG[numberList])
            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = 0
                valueToPutInList.append(value)

            valueToPutInList.append(0)

        row_list.append(valueToPutInList)
        numberList += 1

    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\FinalCSVForTestImage.csv",
            'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def createCSVForTestImage_ForRedTrafficLights_V2(imagePath):
    ImageFunctions.detectRedFromTrafficLights_ForTestImage(imagePath)

    ImageFunctions.createCSVWithPatches_Long(imagePath, 1, 0, "Red")
    ImageFunctions.createCSVWithPosition(imagePath, 1, 0, "Red")

    imagePathForBinary = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\BinaryImages\\BinaryImageTest.jpg'
    ImageFunctions.createCSVWithBinaryValues(imagePathForBinary, 1, 0, "Red")

    listOfPatches = []
    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestPatches.csv",
            'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfPatches.append(row)

    listOfPositions = []
    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestPosition.csv",
            'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfPositions.append(row[1:])

    listOfBinaryPatches = []
    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestBinaryPatches.csv",
            'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfBinaryPatches.append(row)

    listOfTrafficLightForCurrentImage = extractSpecificBox(imagePath, 1, "Red")
    validPatch = 0
    validPatches = []

    for i in range(0, len(listOfPositions)):
        currentListOfPositions = listOfPositions[i]

        okPatch = 0
        for j in range(0, len(listOfTrafficLightForCurrentImage)):
            trafficLight = listOfTrafficLightForCurrentImage[j]
            if trafficLight.label != "Red" and trafficLight.label != "RedRight" and trafficLight.label != "RedLeft":
                continue
            else:
                x_max = trafficLight.x_max
                x_min = trafficLight.x_min
                y_max = trafficLight.y_max
                y_min = trafficLight.y_min

                for p in range(0, len(currentListOfPositions)):
                    listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                    positionX = int(listOfPositionNumbers[0])
                    positionY = int(listOfPositionNumbers[1])

                    if x_min < positionY < x_max and y_min < positionX < y_max:
                        okPatch = 1
                        validPatch += 1
                        validPatches.append(i)
                        break
            if okPatch == 1:
                break

    valueFromPatchCell = []
    row_list = [[]]
    dimensionCSV = len(listOfPatches)

    valueFromPatchCell.append('Position')
    for i in range(0, 234):
        valueFromPatchCell.append(i)
    valueFromPatchCell.append('Clasa')

    row_list[0] = valueFromPatchCell

    (fd, hogImage) = ImageFunctions.createHOGHistogram(imagePath)

    dimensionForEachHistogram = 9
    myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                     for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

    numberList = 0
    for i in range(0, dimensionCSV):
        valueToPutInList = []
        counterOfWhitePixels = 0
        if validPatches.__contains__(i):
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfBinaryPatches[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            # print(myListFromHOG[numberList])
            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = arrayOfBinaryPatches[y]
                if 100 <= int(value) <= 255:
                    counterOfWhitePixels += 1

            if counterOfWhitePixels >= 1:
                for y in range(0, len(arrayOfBinaryPatches)):
                    value = arrayOfBinaryPatches[y]
                    valueToPutInList.append(value)
                if i < int(dimensionCSV / 2):
                    valueToPutInList.append(1)
                else:
                    valueToPutInList.append(0)
        else:
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfBinaryPatches[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            # print(myListFromHOG[numberList])
            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = arrayOfBinaryPatches[y]
                valueToPutInList.append(value)

            valueToPutInList.append(0)

        row_list.append(valueToPutInList)
        numberList += 1

    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\FinalCSVForTestImage.csv",
            'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def createCSVForTestImage_ForYellowTrafficLights(imagePath):
    ImageFunctions.detectYellowFromTrafficLights_ForTestImage(imagePath)

    ImageFunctions.createCSVWithPatches_Long(imagePath, 1, 0, "Yellow")
    ImageFunctions.createCSVWithPosition(imagePath, 1, 0, "Yellow")

    imagePathForBinary = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\BinaryImages\\BinaryImageTest.jpg'
    ImageFunctions.createCSVWithBinaryValues(imagePathForBinary, 1, 0, "Yellow")

    listOfPatches = []
    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\TestPatches.csv",
            'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfPatches.append(row)

    listOfPositions = []
    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\TestPosition.csv",
            'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfPositions.append(row[1:])

    listOfBinaryPatches = []
    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\TestBinaryPatches.csv",
            'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfBinaryPatches.append(row)

    listOfTrafficLightForCurrentImage = extractSpecificBox(imagePath, 1, "Yellow")
    validPatch = 0
    validPatches = []

    for i in range(0, len(listOfPositions)):
        currentListOfPositions = listOfPositions[i]

        okPatch = 0
        for j in range(0, len(listOfTrafficLightForCurrentImage)):
            trafficLight = listOfTrafficLightForCurrentImage[j]
            if trafficLight.label != "Yellow" and trafficLight.label != "YellowRight" and trafficLight.label != "YellowLeft":
                continue
            else:
                x_max = trafficLight.x_max
                x_min = trafficLight.x_min
                y_max = trafficLight.y_max
                y_min = trafficLight.y_min

                for p in range(0, len(currentListOfPositions)):
                    listOfPositionNumbers = re.findall(r'\d+', currentListOfPositions[p])

                    positionX = int(listOfPositionNumbers[0])
                    positionY = int(listOfPositionNumbers[1])

                    if x_min < positionY < x_max and y_min < positionX < y_max:
                        okPatch = 1
                        validPatch += 1
                        validPatches.append(i)
                        break
            if okPatch == 1:
                break

    valueFromPatchCell = []
    row_list = [[]]
    dimensionCSV = len(listOfPatches)

    valueFromPatchCell.append('Position')
    for i in range(0, 234):
        valueFromPatchCell.append(i)
    valueFromPatchCell.append('Clasa')

    row_list[0] = valueFromPatchCell

    (fd, hogImage) = ImageFunctions.createHOGHistogram(imagePath)

    dimensionForEachHistogram = 9
    myListFromHOG = [fd[i * dimensionForEachHistogram:(i + 1) * dimensionForEachHistogram]
                     for i in range((len(fd) + dimensionForEachHistogram - 1) // dimensionForEachHistogram)]

    numberList = 0
    for i in range(0, dimensionCSV):
        valueToPutInList = []
        counterOfWhitePixels = 0
        if validPatches.__contains__(i):
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfBinaryPatches[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            # print(myListFromHOG[numberList])
            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = arrayOfBinaryPatches[y]
                valueToPutInList.append(value)

                if 1 <= int(value) <= 255:
                    counterOfWhitePixels += 1

            if counterOfWhitePixels >= 1:
                valueToPutInList.append(1)
            else:
                valueToPutInList.append(0)
        else:
            valueToPutInList.append(i)

            arrayOfPatches = listOfPatches[i]
            arrayOfBinaryPatches = listOfBinaryPatches[i]

            for y in range(0, len(arrayOfPatches)):
                value = arrayOfPatches[y]
                valueToPutInList.append(value)

            # print(myListFromHOG[numberList])
            valueToPutInList = list(itertools.chain(valueToPutInList, myListFromHOG[numberList]))

            for y in range(0, len(arrayOfBinaryPatches)):
                value = arrayOfBinaryPatches[y]
                valueToPutInList.append(value)

            valueToPutInList.append(0)

        row_list.append(valueToPutInList)
        numberList += 1

    with open(
            f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\FinalCSVForTestImage.csv",
            'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def FinalMethod(imagePath):
    FinalMethodToCreateCSVForClassificationOfGreenTrafficLight_ForTestImage(imagePath, 1)
    FinalMethodToCreateCSVForClassificationOfRedTrafficLight_ForTestImage(imagePath, 1)
    #FinalMethodToCreateCSVForClassificationOfYellowTrafficLight_ForTestImage(imagePath, 1)


def detectTheArrayOfPointsFromPredictedCSV(filename, typeOfTrafficLight, checker):
    df = pd.read_csv(filename)
    matrix = df[df.columns[0]].to_numpy()
    positionListFromPredictedCSV = matrix.tolist()

    positionListFromPredictedCSV.sort()

    listOfPositions = []
    if typeOfTrafficLight == "Green":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\TestCSVs\\TestPosition.csv",
                    'r') as readObj:
                csvReader = reader(readObj)
                header = next(csvReader)
                if header is not None:
                    for row in csvReader:
                        listOfPositions.append(row)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition_Green.csv",
                    'r') as readObj:
                csvReader = reader(readObj)
                header = next(csvReader)
                if header is not None:
                    for row in csvReader:
                        listOfPositions.append(row)
    elif typeOfTrafficLight == "Red":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestPosition.csv",
                    'r') as readObj:
                csvReader = reader(readObj)
                header = next(csvReader)
                if header is not None:
                    for row in csvReader:
                        listOfPositions.append(row)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition_Red.csv",
                    'r') as readObj:
                csvReader = reader(readObj)
                header = next(csvReader)
                if header is not None:
                    for row in csvReader:
                        listOfPositions.append(row)
    elif typeOfTrafficLight == "Yellow":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\TestPosition.csv",
                    'r') as readObj:
                csvReader = reader(readObj)
                header = next(csvReader)
                if header is not None:
                    for row in csvReader:
                        listOfPositions.append(row)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition_Yellow.csv",
                    'r') as readObj:
                csvReader = reader(readObj)
                header = next(csvReader)
                if header is not None:
                    for row in csvReader:
                        listOfPositions.append(row)
    elif typeOfTrafficLight == "All":
        with open(
                f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition.csv",
                'r') as readObj:
            csvReader = reader(readObj)
            header = next(csvReader)
            if header is not None:
                for row in csvReader:
                    listOfPositions.append(row)

    listToBeReturned = []
    for row in range(0, len(listOfPositions)):
        okPositionList = 0

        currentList = listOfPositions[row]
        positionValueFromCurrentList = int(currentList[0])

        for i in range(0, len(positionListFromPredictedCSV)):
            if positionValueFromCurrentList == positionListFromPredictedCSV[i]:
                okPositionList = 1

        if okPositionList != 1:
            continue
        elif okPositionList == 1:
            for j in range(1, len(currentList)):
                listToBeReturned.append(currentList[j])

    return listToBeReturned


def cleanThePredictedCSV(filename, typeOfTrafficLight, checker):
    df = pd.read_csv(filename)
    matrix = df.to_numpy()
    listsFromPredictedTrafficLightsCSV = matrix.tolist()

    listOfWrongPredictedTrafficLightsPatches = []
    for x in range(0, len(listsFromPredictedTrafficLightsCSV)):
        currentList = listsFromPredictedTrafficLightsCSV[x]
        counterWhitePixels = 0

        for y in range(210, len(currentList)):
            value = currentList[y]
            if value == 255:
                counterWhitePixels += 1

        if counterWhitePixels == 0:
            listOfWrongPredictedTrafficLightsPatches.append(x)

    if len(listOfWrongPredictedTrafficLightsPatches) > 0:
        for i in range(0, len(listOfWrongPredictedTrafficLightsPatches)):
            listsFromPredictedTrafficLightsCSV.pop(i)

    listOfPositions = []
    if typeOfTrafficLight == "Green":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\TestCSVs\\TestPosition.csv",
                    'r') as readObj:
                csvReader = reader(readObj)
                header = next(csvReader)
                if header is not None:
                    for row in csvReader:
                        listOfPositions.append(row)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition_Green.csv",
                    'r') as readObj:
                csvReader = reader(readObj)
                header = next(csvReader)
                if header is not None:
                    for row in csvReader:
                        listOfPositions.append(row)
    elif typeOfTrafficLight == "Red":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\TestPosition.csv",
                    'r') as readObj:
                csvReader = reader(readObj)
                header = next(csvReader)
                if header is not None:
                    for row in csvReader:
                        listOfPositions.append(row)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition_Red.csv",
                    'r') as readObj:
                csvReader = reader(readObj)
                header = next(csvReader)
                if header is not None:
                    for row in csvReader:
                        listOfPositions.append(row)
    elif typeOfTrafficLight == "Yellow":
        if checker == 0:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\TestPosition.csv",
                    'r') as readObj:
                csvReader = reader(readObj)
                header = next(csvReader)
                if header is not None:
                    for row in csvReader:
                        listOfPositions.append(row)
        elif checker == 1:
            with open(
                    f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\TestPosition_Yellow.csv",
                    'r') as readObj:
                csvReader = reader(readObj)
                header = next(csvReader)
                if header is not None:
                    for row in csvReader:
                        listOfPositions.append(row)

    listToBeReturned = []
    for row in range(0, len(listOfPositions)):
        okPositionList = 0

        currentList = listOfPositions[row]
        positionValueFromCurrentList = int(currentList[0])

        for i in range(0, len(listsFromPredictedTrafficLightsCSV)):
            position = int(listsFromPredictedTrafficLightsCSV[i][0])
            if positionValueFromCurrentList == position:
                okPositionList = 1

        if okPositionList != 1:
            continue
        elif okPositionList == 1:
            for j in range(1, len(currentList)):
                listToBeReturned.append(currentList[j])

    return listToBeReturned
