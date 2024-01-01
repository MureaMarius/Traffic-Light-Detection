from cv2 import cv2
from cv2.cv2 import imread, imshow
from skimage.filters import prewitt_h, prewitt_v

from functions import Utility
from functions import ImageFunctions
from functions import Classifier
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt


def createTrainDataset(typeOfTrafficLight):
    if typeOfTrafficLight == "Green":
        Utility.createCSVForClassificationOfTrafficLights_Green(80)
        Utility.FinalMethodToCreateCSVForClassificationOfGreenTrafficLight(80)
        Classifier.shuffleDataFrame(
            'C:\\Users\\Marius\\Desktop\\LicentaProiect\\DatasetForAllImages_Green.csv', "Green")
        Classifier.shuffleLinesInDataFrame('C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Green.csv',
                                           "Green")
        Classifier.classifierKNN("Green")
    elif typeOfTrafficLight == "Red":
        Utility.createCSVForClassificationOfTrafficLights_Red(80)
        Utility.FinalMethodToCreateCSVForClassificationOfRedTrafficLight(80)
        Classifier.shuffleDataFrame(
            'C:\\Users\\Marius\\Desktop\\LicentaProiect\\DatasetForAllImages_Red.csv', "Red")
        Classifier.shuffleLinesInDataFrame('C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Red.csv',
                                           "Red")
        Classifier.classifierKNN("Red")
    elif typeOfTrafficLight == "Yellow":
        Utility.FinalMethodToCreateCSVForClassificationOfYellowTrafficLight(11)
        Classifier.shuffleDataFrame(
            'C:\\Users\\Marius\\Desktop\\LicentaProiect\\DatasetForAllImages_Yellow.csv', "Yellow")
        Classifier.shuffleLinesInDataFrame('C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Yellow.csv',
                                           "Yellow")
        Classifier.classifierKNN("Yellow")
    elif typeOfTrafficLight == "All":
        Utility.FinalMethodToCreateCSVForClassificationOfAllTrafficLights(22)
        Classifier.shuffleDataFrame('C:\\Users\\Marius\\Desktop\\LicentaProiect\\DatasetForAllImages_ForAll.csv', "All")
        Classifier.shuffleLinesInDataFrame('C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_AllTrafficLights.csv', "All")

        Classifier.classifierKNN("All")


def selectTestImageFromFolder_ForGreenTrafficLights():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(parent=root,
                                           initialdir="C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages",
                                           title='Please select a directory')

    file_path = file_path.replace('/', '\\')
    Utility.FinalMethodToCreateCSVForClassificationOfGreenTrafficLight_ForTestImage(file_path, 0)

    print("CSV-ul de test este gata!!")
    CSVFilenameForPredict = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\GreenTL\\testImages\\TestCSVs\\FinalCSVForTestImage.csv'

    print("\nRezultatele predictiei: ")
    Classifier.PredictOnTestImage(CSVFilenameForPredict, "Green", 0)

    CSVFilenameForFinalImage = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\PredictedPointsForGreenTrafficLights.csv'
    listOfPoints = Utility.detectTheArrayOfPointsFromPredictedCSV(CSVFilenameForFinalImage, "Green", 0)
    listOfCleanedPoints = Utility.cleanThePredictedCSV(CSVFilenameForFinalImage, "Green", 0)

    ImageFunctions.createFirstImageResultFromClassifierPredictedPoints(listOfPoints)
    ImageFunctions.createImageFromClassifierImageResult_Final_InRectangle(listOfCleanedPoints, file_path, "Green")

    print("\nPuteti alege o noua imagine de test!\n")


def selectTestImageFromFolder_ForRedTrafficLights():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(parent=root,
                                           initialdir="C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages",
                                           title='Please select a directory')
    file_path = file_path.replace('/', '\\')
    Utility.FinalMethodToCreateCSVForClassificationOfRedTrafficLight_ForTestImage(file_path, 0)

    print("CSV-ul de test este gata!!")
    CSVFilenameForPredict = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\RedTL\\testImages\\TestCSVs\\FinalCSVForTestImage.csv'

    print("\nRezultatele predictiei: ")
    Classifier.PredictOnTestImage(CSVFilenameForPredict, "Red", 0)

    CSVFilenameForFinalImage = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\PredictedPointsForRedTrafficLights.csv'
    listOfPoints = Utility.detectTheArrayOfPointsFromPredictedCSV(CSVFilenameForFinalImage, "Red", 0)
    listOfCleanedPoints = Utility.cleanThePredictedCSV(CSVFilenameForFinalImage, "Red", 0)

    ImageFunctions.createFirstImageResultFromClassifierPredictedPoints(listOfPoints)
    ImageFunctions.createImageFromClassifierImageResult_Final_InRectangle(listOfCleanedPoints, file_path, "Red")

    print("\nPuteti alege o noua imagine de test!\n")


def selectTestImageFromFolder_ForYellowTrafficLights():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(parent=root,
                                           initialdir="C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages",
                                           title='Please select a directory')
    file_path = file_path.replace('/', '\\')
    Utility.FinalMethodToCreateCSVForClassificationOfYellowTrafficLight_ForTestImage(file_path)

    print("CSV-ul de test este gata!!")
    CSVFilenameForPredict = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\YellowTL\\testImages\\TestCSVs\\FinalCSVForTestImage.csv'

    print("\nRezultatele predictiei: ")
    Classifier.PredictOnTestImage(CSVFilenameForPredict, "Yellow", 0)

    CSVFilenameForFinalImage = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\PredictedPointsForYellowTrafficLights.csv'
    listOfPoints = Utility.detectTheArrayOfPointsFromPredictedCSV(CSVFilenameForFinalImage, "Yellow", 0)
    listOfCleanedPoints = Utility.cleanThePredictedCSV(CSVFilenameForFinalImage, "Yellow", 0)

    ImageFunctions.createFirstImageResultFromClassifierPredictedPoints(listOfPoints)
    ImageFunctions.createImageFromClassifierImageResult_Final_InRectangle(listOfCleanedPoints, file_path, "Yellow")

    print("\nPuteti alege o noua imagine de test!\n")


def selectTestImageFromFolder_ForAllTrafficLights():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(parent=root,
                                           initialdir="C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages",
                                           title='Please select a directory')
    file_path = file_path.replace('/', '\\')
    Utility.FinalMethod(file_path)

    print("CSV-ul de test este gata!!")
    CSVFilenameForPredictForGreen = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\FinalCSVForTestImage_Green.csv'
    CSVFilenameForPredictForRed = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\FinalCSVForTestImage_Red.csv'

    print("\nRezultatele predictiei (---- Green Traffic Light ----): ")
    Classifier.PredictOnTestImage(CSVFilenameForPredictForGreen, "Green", 1)

    print("\nRezultatele predictiei (---- Red Traffic Light ----): ")
    Classifier.PredictOnTestImage(CSVFilenameForPredictForRed, "Red", 1)

    CSVFilenameForFinalImage_Green = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\PredictedPointsForGreenTrafficLights.csv'
    CSVFilenameForFinalImage_Red = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\PredictedPointsForRedTrafficLights.csv'
    listOfPointsForGreen = Utility.detectTheArrayOfPointsFromPredictedCSV(CSVFilenameForFinalImage_Green, "Green", 1)
    listOfPointsForRed = Utility.detectTheArrayOfPointsFromPredictedCSV(CSVFilenameForFinalImage_Red, "Red", 1)

    listOfCleanedPointsGreen = Utility.cleanThePredictedCSV(CSVFilenameForFinalImage_Green, "Green", 1)
    listOfCleanedPointsRed = Utility.cleanThePredictedCSV(CSVFilenameForFinalImage_Red, "Red", 1)

    finalList = listOfCleanedPointsGreen + listOfCleanedPointsRed

    ImageFunctions.createFirstImageResultFromClassifierPredictedPoints(finalList)
    ImageFunctions.createImageFromClassifierImageResult_Final_InRectangle_ForAll(listOfCleanedPointsGreen,
                                                                                 listOfCleanedPointsRed, file_path)

    print("\nPuteti alege o noua imagine de test!\n")


def GUI():
    master = tk.Tk()
    master.title("Traffic Lights Detection")
    master.geometry("350x200")

    button1 = tk.Button(master, text="Red Traffic Lights", command=selectTestImageFromFolder_ForRedTrafficLights)
    button1.place(x=25, y=100)

    button2 = tk.Button(master, text="Green Traffic Lights", command=selectTestImageFromFolder_ForGreenTrafficLights)
    button2.place(x=25, y=25)

    button3 = tk.Button(master, text="Yellow Traffic Lights", command=selectTestImageFromFolder_ForYellowTrafficLights)
    button3.place(x=175, y=25)

    button4 = tk.Button(master, text="All Traffic Lights", command=selectTestImageFromFolder_ForAllTrafficLights)
    button4.place(x=175, y=100)

    master.mainloop()


def main():
    # createTrainDataset("Green")
    GUI()


main()
