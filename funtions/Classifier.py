import csv
import pickle
import random
from cmath import sqrt
from csv import reader
from os import listdir

import joblib
import pandas as pd
import numpy as np
from skimage.metrics import mean_squared_error
from sklearn import metrics, model_selection
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from yellowbrick.regressor import PredictionError, prediction_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from functions import Utility


def shuffleDataFrame(filename, typeOfTrafficLight):
    global non_fraud_df, df_red, df_yellow, df_green
    dataFrame = pd.read_csv(filename)

    # Shuffle the Dataset.
    shuffled_df = dataFrame.sample(frac=1, random_state=4)

    # Put all the fraud class in a separate dataset.
    fraud_df = shuffled_df.loc[shuffled_df['Clasa'] == 1]

    if typeOfTrafficLight == "All":
        df_red = shuffled_df.loc[shuffled_df['Clasa'] == 1]
        df_green = shuffled_df.loc[shuffled_df['Clasa'] == 2]
        df_yellow = shuffled_df.loc[shuffled_df['Clasa'] == 3]

    # Randomly select 2000 observations from the non-fraud (majority class)
    if typeOfTrafficLight == "Green":
        non_fraud_df = shuffled_df.loc[shuffled_df['Clasa'] == 0].sample(n=3000, random_state=42)
    elif typeOfTrafficLight == "Red":
        non_fraud_df = shuffled_df.loc[shuffled_df['Clasa'] == 0].sample(n=8000, random_state=42)
    elif typeOfTrafficLight == "Yellow":
        non_fraud_df = shuffled_df.loc[shuffled_df['Clasa'] == 0].sample(n=1100, random_state=42)
    elif typeOfTrafficLight == "All":
        non_fraud_df = shuffled_df.loc[shuffled_df['Clasa'] == 0].sample(n=7000, random_state=42)

    # Concatenate both dataframes again
    normalized_df = pd.concat([fraud_df, non_fraud_df])

    # plot the dataset after the undersampling
    plt.figure(figsize=(8, 8))
    sns.countplot('Clasa', data=normalized_df)
    plt.title('Balanced Classes')
    plt.show()

    if typeOfTrafficLight == "All":
        normalized_df = pd.concat([df_red, df_green])
        normalized_df = pd.concat([normalized_df, df_yellow])
        normalized_df = pd.concat([normalized_df, non_fraud_df])

    if typeOfTrafficLight == "Green":
        normalized_df.to_csv('C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Green.csv', index=False)
    elif typeOfTrafficLight == "Red":
        normalized_df.to_csv('C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Red.csv', index=False)
    elif typeOfTrafficLight == "Yellow":
        normalized_df.to_csv('C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Yellow.csv', index=False)
    elif typeOfTrafficLight == "All":
        normalized_df.to_csv('C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_AllTrafficLights.csv',
                             index=False)


def shuffleLinesInDataFrame(filename, typeOfTrafficLight):
    listOfPatches = []
    with open(filename, 'r') as readObj:
        csvReader = reader(readObj)
        header = next(csvReader)
        if header is not None:
            for row in csvReader:
                listOfPatches.append(row)

    random.shuffle(listOfPatches)

    valueFromPatchCell = []
    row_list = [[]]

    valueFromPatchCell.append('Position')
    for i in range(0, 234):
        valueFromPatchCell.append(i)
    valueFromPatchCell.append('Clasa')
    row_list[0] = valueFromPatchCell

    for i in range(0, len(listOfPatches)):
        row_list.append(listOfPatches[i])

    if typeOfTrafficLight == "Green":
        with open(f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Green.csv", 'w',
                  newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)
    elif typeOfTrafficLight == "Red":
        with open(f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Red.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)
    elif typeOfTrafficLight == "Yellow":
        with open(f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Yellow.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)
    elif typeOfTrafficLight == "All":
        with open(f"C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_AllTrafficLights.csv", 'w',
                  newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)


def SIFTfeatures(imagePath):
    initialImage = cv2.imread(imagePath)
    image = cv2.cvtColor(initialImage, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image, None)

    print(len(keypoints_1))


def AdaboostAlgorithm():
    filename = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Green.csv'
    dataFrame = pd.read_csv(filename)

    y = dataFrame.Clasa
    x = dataFrame.drop('Clasa', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    svc = SVC(probability=True, kernel='linear')
    abc = AdaBoostClassifier()

    # params = {'n_estimators': np.arange(10, 300, 10), 'learning_rate': [1, 0.5, 0.25, 0.12, 0.4]}
    # grid_cv = GridSearchCV(AdaBoostClassifier(), param_grid=params, cv=5, n_jobs=-1)
    # grid_cv.fit(x_train, y_train)
    # print("Best params:", grid_cv.best_params_)

    model = abc.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # filename = "TrainModel_V1.pkl"
    # with open(filename, 'wb') as file:
    #    pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    # joblib.dump(model, 'TrainModel_V2.pkl', compress=9)


def RandomForestClassifierAlgorithm(filename):
    dataFrame = pd.read_csv(filename)

    y = dataFrame.Clasa
    x = dataFrame.drop('Clasa', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    model = clf.fit(x_train, y_train)

    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_split=1e-07, min_samples_leaf=1,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
                           verbose=0, warm_start=False)

    preds = model.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, preds))

    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    filename = "TrainModel_V2.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def classifierKNN(typeOfTrafficLight):
    global filename

    if typeOfTrafficLight == "Red":
        filename = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Red.csv'
    elif typeOfTrafficLight == "Green":
        filename = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Green.csv'
    elif typeOfTrafficLight == "Yellow":
        filename = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Yellow.csv'
    elif typeOfTrafficLight == "All":
        filename = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_AllTrafficLights.csv'
    elif typeOfTrafficLight == "TEST":
        filename = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\New folder\\DataFrameForTrain_V2.csv'

    dataFrame = pd.read_csv(filename)

    y = dataFrame.Clasa
    x = dataFrame.drop('Clasa', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Use the KNN classifier to fit data:
    # classifier = KNeighborsClassifier(n_neighbors=150)     red
    # classifier = KNeighborsClassifier(n_neighbors=100)
    """
    grid_params = {'n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], 'weights': ['uniform', 'distance'],
                   'metric': ['euclidean', 'manhattan']}
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=3, n_jobs=-1)
    gs.fit(x_train, y_train)
    print("Best params:", gs.best_params_)
    """
    classifier = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
    model = classifier.fit(x_train, y_train)

    # Predict y data with classifier:
    y_predict = classifier.predict(x_test)

    # print(confusion_matrix(y_test, y_predict))
    print(classification_report(y_test, y_predict))

    scores = cross_val_score(model, x_train, y_train, cv=10)
    print("Accuracy: {0:.2f}%".format(100 * scores.mean()))

    diff = y_test - y_predict
    diff.hist(bins=40)
    plt.title('Histogram of prediction errors')
    plt.xlabel('MPG prediction error')
    plt.ylabel('Frequency')

    sqrt(mean_squared_error(y_test, y_predict))

    _, ax = plt.subplots()
    ax.scatter(x=range(0, y_test.size), y=y_test, c='blue', label='Actual', alpha=0.3)
    ax.scatter(x=range(0, y_predict.size), y=y_predict, c='red', label='Predicted', alpha=0.3)

    plt.title('Actual and predicted values')
    plt.xlabel('Observations')
    plt.ylabel('mpg')
    plt.legend()
    plt.show()

    """
    if typeOfTrafficLight == "Red":
        filename = "TrainModel_Red.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    elif typeOfTrafficLight == "Green":
        filename = "TrainModel_Green_TEST.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    elif typeOfTrafficLight == "Yellow":
        filename = "TrainModel_Yellow.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    elif typeOfTrafficLight == "All":
        filename = "TrainModel_AllTL.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    """


def naiveBayesClassifier():
    filename = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainDataset_Green.csv'
    dataFrame = pd.read_csv(filename)

    y = dataFrame.Clasa
    x = dataFrame.drop('Clasa', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Initialize classifier:
    gnb = GaussianNB()

    # Train the classifier:
    model = gnb.fit(x_train, y_train)
    # Make predictions with the classifier:
    predictive_labels = gnb.predict(x_test)
    # print(predictive_labels)

    # Evaluate label (subsets) accuracy:
    print(confusion_matrix(y_test, predictive_labels))
    print(classification_report(y_test, predictive_labels))

    filename = "TrainModel_V3.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def PredictOnTestImage(filename, typeOfTrafficLight, checker):
    global modelPath

    if typeOfTrafficLight == "Red":
        modelPath = "C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainModel_Red.pkl"
    elif typeOfTrafficLight == "Green":
        modelPath = "C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainModel_Green.pkl"
    elif typeOfTrafficLight == "Yellow":
        modelPath = "C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainModel_Yellow.pkl"
    elif typeOfTrafficLight == "All":
        modelPath = "C:\\Users\\Marius\\Desktop\\LicentaProiect\\TrainModel_AllTL.pkl"

    dataFrame = pd.read_csv(filename)

    Y = dataFrame.Clasa
    X = dataFrame.drop('Clasa', axis=1)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.99,
                                                                        random_state=1)

    with open(modelPath, 'rb') as file:
        Pickled_LR_Model = pickle.load(file)

    Ypredict = Pickled_LR_Model.predict(x_test)

    if typeOfTrafficLight == "Red":
        if checker == 0:
            x_test[Ypredict == 1].to_csv('C:\\Users\\Marius\\Desktop\\LicentaProiect'
                                         '\\PredictedPointsForRedTrafficLights.csv', index=False)
        elif checker == 1:
            x_test[Ypredict == 1].to_csv('C:\\Users\\Marius\\Desktop\\LicentaProiect'
                                         '\\AllTL\\PredictedPointsForRedTrafficLights.csv', index=False)
    elif typeOfTrafficLight == "Green":
        if checker == 0:
            x_test[Ypredict == 1].to_csv(
                'C:\\Users\\Marius\\Desktop\\LicentaProiect\\PredictedPointsForGreenTrafficLights.csv', index=False)
        elif checker == 1:
            x_test[Ypredict == 1].to_csv('C:\\Users\\Marius\\Desktop\\LicentaProiect'
                                         '\\AllTL\\PredictedPointsForGreenTrafficLights.csv', index=False)
    elif typeOfTrafficLight == "Yellow":
        if checker == 0:
            x_test[Ypredict == 1].to_csv(
                'C:\\Users\\Marius\\Desktop\\LicentaProiect\\PredictedPointsForYellowTrafficLights.csv', index=False)
        elif checker == 1:
            x_test[Ypredict == 1].to_csv('C:\\Users\\Marius\\Desktop\\LicentaProiect'
                                         '\\AllTL\\PredictedPointsForYellowTrafficLights.csv', index=False)
    elif typeOfTrafficLight == "All":
        x_test[Ypredict == 1].to_csv(
            'C:\\Users\\Marius\\Desktop\\LicentaProiect\\PredictedPointsForRedTrafficLights_All.csv', index=False)
        x_test[Ypredict == 2].to_csv(
            'C:\\Users\\Marius\\Desktop\\LicentaProiect\\PredictedPointsForGreenTrafficLights_All.csv', index=False)
        x_test[Ypredict == 3].to_csv(
            'C:\\Users\\Marius\\Desktop\\LicentaProiect\\PredictedPointsForYellowTrafficLights_All.csv', index=False)

    score = Pickled_LR_Model.score(x_test, y_test)
    print("Scorul testului: {0:.2f}%".format(100 * score))

    roc = roc_auc_score(y_test, Ypredict)
    print('ROC: %.3f ' % roc)

    f1 = f1_score(y_test, Ypredict, average='weighted', labels=np.unique(Ypredict))
    print('Scorul F1: %.3f' % f1)

    precision = precision_score(y_test, Ypredict, average='weighted', labels=np.unique(Ypredict))
    print('Scorul preciziei: %.3f' % precision)

    sqrt(mean_squared_error(y_test, Ypredict))

    """
    _, ax = plt.subplots()
    ax.scatter(x=range(0, y_test.size), y=y_test, c='blue', label='Actuale', alpha=0.3)
    ax.scatter(x=range(0, Ypredict.size), y=Ypredict, c='red', label='Prezise', alpha=0.3)

    plt.title('Valorile actuale și prezise')
    plt.xlabel('Observații')
    plt.ylabel('mpg')
    plt.legend()
    plt.show()
    """
    return [score, roc, f1, precision]


def PredictOnTestImage_ForGlobalScore():
    global modelPath

    directoryForTestImages = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\'
    files = listdir(directoryForTestImages)
    images_list = [i for i in files if i.endswith('.png')]

    scoreTotalGreen = 0
    rocTotalGreen = 0
    f1TotalGreen = 0
    precisionTotalGreen = 0
    scoreTotalRed = 0
    rocTotalRed = 0
    f1TotalRed = 0
    precisionTotalRed = 0

    for idx, image in enumerate(images_list):
        pathImage = directoryForTestImages + image

        Utility.FinalMethod(pathImage)
        print("CSV-urile pt imaginea ---", image, "--- sunt gata!\n")

        CSVFilenameForPredictForGreen = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\FinalCSVForTestImage_Green.csv'
        CSVFilenameForPredictForRed = 'C:\\Users\\Marius\\Desktop\\LicentaProiect\\AllTL\\testImages\\TestCSVs\\FinalCSVForTestImage_Red.csv'

        scoreGreen, rocGreen, f1Green, precisionGreen = PredictOnTestImage(CSVFilenameForPredictForGreen, "Green", 1)
        scoreRed, rocRed, f1Red, precisionRed = PredictOnTestImage(CSVFilenameForPredictForRed, "Red", 1)

        scoreTotalGreen += scoreGreen
        rocTotalGreen += rocGreen
        f1TotalGreen += f1Green
        precisionTotalGreen += precisionGreen

        scoreTotalRed += scoreRed
        rocTotalRed += rocRed
        f1TotalRed += f1Red
        precisionTotalRed += precisionRed

        print("\nSe trece la urmatoare imagine!\n")

    print('Medie pe --SCOR-- (pentru semafoarele cu Verde): %.3f ' % (scoreTotalGreen / 19))
    print('Medie pe --ROC-- (pentru semafoarele cu Verde): %.3f ' % (rocTotalGreen / 19))
    print('Medie pe --F1-- (pentru semafoarele cu Verde): %.3f ' % (f1TotalGreen / 19))
    print('Medie pe --Precizie-- (pentru semafoarele cu Verde): %.3f' % (precisionTotalGreen / 19))
    print("\n\n")
    print('Medie pe --SCOR-- (pentru semafoarele cu Rosu): %.3f ' % (scoreTotalRed / 19))
    print('Medie pe --ROC-- (pentru semafoarele cu Rosu): %.3f ' % (rocTotalRed / 19))
    print('Medie pe --F1-- (pentru semafoarele cu Rosu): %.3f ' % (f1TotalRed / 19))
    print('Medie pe --Precizie-- (pentru semafoarele cu Rosu): %.3f ' % (precisionTotalRed / 19))
