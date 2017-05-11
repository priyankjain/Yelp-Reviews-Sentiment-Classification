# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 23:45:09 2017

@author: vedant
"""
import sys
import string, csv, re, operator
from collections import Counter

#Converting to Lower Case
def lowerCase(your_list):
    for i in range(len(your_list)):
        your_list[i][2] = your_list[i][2].lower()
    return your_list

#Remove Punctuation
def remove_Punctuation(your_list):
    exclude = set(string.punctuation)
    for i in range(len(your_list)):
        your_list[i][2] = ''.join(ch for ch in your_list[i][2] if ch not in exclude)
    return your_list

#Converting to List to Words
def stringToListofWords(your_list, wordList):
    for i in range(len(your_list)):
        listOfWords = your_list[i][2].split()
        wordListSet = set(listOfWords)
        newWordList = list(wordListSet)
        wordList.extend(newWordList)
    #print(wordList)
    return wordList

#Creating Feature Matrix
def createMatrix(featureList, numberOfFeature, your_list):
    matrix = []
    for i in range(len(your_list)):
        row = [0]*(numberOfFeature + 1)
        listOfWords = your_list[i][2].split()
        row[numberOfFeature] = int(your_list[i][1])                  # Storing Class label in the last column
        for j in range(len(listOfWords)):
            idx = featureList.index(listOfWords[j]) if listOfWords[j] in featureList else -1
            if idx is not -1:
                row[idx] = 1
        matrix.append(row)    
    return matrix

def printTop10Words(top10Words):
    for i in range(0,10):
        print("WORD%d" % (i+1),top10Words[i][0])
    return
    
#Printing top 10 frequency Words
def printTop10FrequencyWords(sortedFeatureList):
    count= 0
    prevCount = 0
    for featurecount in sortedFeatureList:
        if featurecount[1] != prevCount:
            count = count + 1
            if count == 11:
                break
            prevCount = featurecount[1]
        print("WORD%d" % count, featurecount[0])
    
#Calculating Probability for NBC
def calculateProbability(numberOfFeature, matrix):
    a = [0]*(numberOfFeature + 1)      #Count for calculating P(X=yes|Class=Yes)
    b = [0]*(numberOfFeature + 1)      #Count for calculating P(X=No|Class=Yes)
    c = [0]*(numberOfFeature + 1)      #Count for calculating P(X=yes|Class=No)
    d = [0]*(numberOfFeature + 1)      #Count for calculating P(X=No|Class=No)
    probList = []
    classLabelProb = [0]*2
    #print(len(matrix))
    for i in range(len(matrix)):
        for j in range(0,numberOfFeature+1):
            if (matrix[i][j] == 1 and matrix[i][numberOfFeature] == 1):
                a[j] = a[j] + 1
            elif (matrix[i][j] == 0 and matrix[i][numberOfFeature] == 1):
                b[j] = b[j] + 1
            elif (matrix[i][j] == 1 and matrix[i][numberOfFeature] == 0):
                c[j] = c[j] + 1
            elif (matrix[i][j] == 0 and matrix[i][numberOfFeature] == 0):
                d[j] = d[j] + 1
        
    for j in range(0,numberOfFeature+1):
        if (j == numberOfFeature):
            classLabelProb[0] = (a[j] + 1)/(a[j] + d[j] + 2)        #For Class Label  Yes
            classLabelProb[1] = (d[j] + 1)/(a[j] + d[j] + 2)        #For Class LAbel No
        else:
            probListForFeature = [0]*4
            probListForFeature[0] = (a[j] + 1)/(a[j] + b[j] + 2)
            probListForFeature[1] = (b[j] + 1)/(a[j] + b[j] + 2)
            probListForFeature[2] = (c[j] + 1)/(c[j] + d[j] + 2)
            probListForFeature[3] = (d[j] + 1)/(c[j] + d[j] + 2)
            #print(probListForFeature)
            probList.append(probListForFeature)
    return probList, classLabelProb

#Predicting Class for Test Data
def predictClass(matrix, numberOfFeature, probList, classLabelProb):
    probNoClass = 1
    probYesClass = 1
    predictedClassList = []
    #print(classLabelProb)
    for i in range(len(matrix)):
        probNoClass = 1
        probYesClass = 1
        for j in range(0,numberOfFeature):
            if(matrix[i][j] == 1):
                probYesClass = probYesClass * probList[j][0]
                probNoClass = probNoClass * probList[j][2]
            elif (matrix[i][j] == 0):
                probYesClass = probYesClass * probList[j][1]
                probNoClass = probNoClass * probList[j][3]
        probYesClass = probYesClass * classLabelProb[0]
        probNoClass = probNoClass * classLabelProb[1]
        if(probYesClass > probNoClass):
            predictedClassList.append(1)
        else:
            predictedClassList.append(0)
    return predictedClassList

#Zero-One Error Base 
def errorCalculation(predictedClassList, matrix, numberOfFeature):
    err = 0
    for i in range(len(matrix)):
        if (predictedClassList[i] != matrix[i][numberOfFeature]):
            err = err + 1
    return err

#Base Line Error Calculation
def baseLineClassPredicted(numberOfFeature, matrix):
    baseLineZero = 0
    baseLineOne = 0
    for i in range(0, len(matrix)):
        if(matrix[i][numberOfFeature] == 1):
            baseLineOne = baseLineOne + 1
        else:
            baseLineZero = baseLineZero + 1
    if(baseLineZero > baseLineOne):
        return 0                    # Return Class 0
    else:
        return 1                    # Return Class 1
        
def baseLineLoss(numberOfFeature, matrix, classPredicted):
    noOfErrorBaseLine = 0
    for i in range(0,len(matrix)):
        if(matrix[i][numberOfFeature] != classPredicted):
            noOfErrorBaseLine = noOfErrorBaseLine + 1
    return (noOfErrorBaseLine)/(len(matrix))

def nbctraining():
    train_file = open(sys.argv[1], 'r')
    
    reader = csv.reader(train_file, delimiter='\t')
    your_list = list(reader)
    your_list = lowerCase(your_list)
    your_list = remove_Punctuation(your_list)
    wordList = []
    wordList = stringToListofWords(your_list, wordList)
    sortedWordList = wordList     #sorted(wordList)

    numberOfFeature = 500

    featureListWithCount = set(Counter(sortedWordList).most_common(numberOfFeature)) - set(Counter(sortedWordList).most_common(100)) 
    sortedFeatureList = sorted(featureListWithCount, key=lambda tup: tup[1], reverse=True)
    
    #sortedFeatureList = sorted(featureListWithCount, key=operator.itemgetter(1,0), reverse=True)
    top10Words = sortedFeatureList[0:10]
    
    #printTop10Words(top10Words)
    printTop10FrequencyWords(sortedFeatureList)
    
    featureList = [i[0] for i in sortedFeatureList]
    matrix = []
    matrix = createMatrix(featureList, numberOfFeature, your_list)
    probList = [] 
    probList,classLabelProb = calculateProbability(numberOfFeature, matrix)
  
    #For Base Line Loss
    classPredictedBaseLine = baseLineClassPredicted(numberOfFeature, matrix)
    
    test_file = open(sys.argv[2], 'r')
    reader = csv.reader(test_file, delimiter='\t')
    test_list = list(reader)
    #print(your_list)
    test_list = lowerCase(test_list)
    test_list = remove_Punctuation(test_list)
    testMatrix = []
    testMatrix = createMatrix(featureList, numberOfFeature, test_list)
    predictedClassList = predictClass(testMatrix, numberOfFeature, probList, classLabelProb)
    noOfErrors = errorCalculation(predictedClassList, testMatrix, numberOfFeature)
    zeroOneError = (noOfErrors)/(len(testMatrix))
    print("ZERO-ONE-LOSS", zeroOneError)
    
    #For Base Line Loss
    #baseLineError = baseLineLoss(numberOfFeature, testMatrix, classPredictedBaseLine)
    #print("BASE-LINE-LOSS", baseLineError)
nbctraining()






#Code for Variation of Number of Feature 
'''def numberFeature():
    train_file = open(sys.argv[1], 'r')
    
    numberOfFeatureList = [10, 50, 250, 500, 1000, 4000]
    percentageTrainingData = [1, 5, 10, 20, 50, 90]
    totalLen = 2000
    reader = csv.reader(train_file, delimiter='\t')
    original_file = list(reader)
    meanZeroOneError = [0]*6
    meanBaseLineError = [0]*6
    stdDevZeroOneError = [0]*6 
    stdDevBaseLineError = [0]*6   
    #Keeping Percentage Data constant =50 and varying number of Feature
    for k in range(0,6):
        zeroOneError = []
        baseLineError = []
        numberOfFeature = numberOfFeatureList[k]
        print("Running Loop # ",k)
        for j in range(0,10):
            lengthTrainingData =int((totalLen * percentageTrainingData[4])/100)

            your_list = random.sample(original_file,lengthTrainingData)
            test_list = [x for x in original_file if x not in your_list]
            
            your_list = lowerCase(your_list)
            your_list = remove_Punctuation(your_list)
            wordList = []
            wordList = stringToListofWords(your_list, wordList)
            sortedWordList = wordList   #sorted(wordList)
        
            featureListWithCount = set(Counter(sortedWordList).most_common(numberOfFeature+100)) - set(Counter(sortedWordList).most_common(100)) 
            sortedFeatureList = sorted(featureListWithCount, key=lambda tup: tup[1], reverse=True)
            #sortedFeatureList = sorted(featureListWithCount, key=operator.itemgetter(1,0), reverse=True)
            top10Words = sortedFeatureList[0:10]
            featureList = [i[0] for i in sortedFeatureList]
            matrix = []
            matrix = createMatrix(featureList, numberOfFeature, your_list)
            probList = []
            probList,classLabelProb = calculateProbability(numberOfFeature, matrix)
            
            #For Base Line Loss
            classPredictedBaseLine = baseLineClassPredicted(numberOfFeature, matrix)
            
            test_list = lowerCase(test_list)
            test_list = remove_Punctuation(test_list)
            testMatrix = []
            testMatrix = createMatrix(featureList, numberOfFeature, test_list)
            predictedClassList = predictClass(testMatrix, numberOfFeature, probList, classLabelProb)
            noOfErrors = errorCalculation(predictedClassList, testMatrix, numberOfFeature)
            zeroOneError.append((noOfErrors)/(len(testMatrix)))
            #For Base Line Loss
            baseLineError.append(baseLineLoss(numberOfFeature, testMatrix, classPredictedBaseLine))
        print("ZERO-ONE-LOSS", zeroOneError)
        print("BASE-LINE-LOSS", baseLineError)
        
        meanZeroOneError[k] = numpy.mean(zeroOneError)
        meanBaseLineError[k] = numpy.mean(baseLineError)
        stdDevZeroOneError[k] = numpy.std(zeroOneError) 
        stdDevBaseLineError[k] = numpy.std(baseLineError)
    print("Feature List Variation ")
    print(numberOfFeatureList)
    print(meanZeroOneError)
    print(stdDevZeroOneError)
    print(meanBaseLineError)
    print(stdDevBaseLineError)
    plt.errorbar(numberOfFeatureList,meanZeroOneError,stdDevZeroOneError,marker=".",label="Naive Bayes Zero One Error")
    plt.errorbar(numberOfFeatureList,meanBaseLineError,stdDevBaseLineError,marker=".",label="BaseLine Error")
    plt.legend()
    plt.xlabel("Number of Features (W)")
    plt.ylabel("Mean Zero One Error")
    plt.title("Error Curve for [10, 50, 250, 500, 1000, 4000]")

    plt.savefig('NumberFeatureVariation.jpg') '''
    
               
               
#Code for Variation of Data Percentage
'''def data_percentage():
    train_file = open(sys.argv[1], 'r')
    
    numberOfFeatureList = [10, 50, 250, 500, 1000, 4000]
    percentageTrainingData = [1, 5, 10, 20, 50, 90]
    totalLen = 2000
    reader = csv.reader(train_file, delimiter='\t')
    original_file = list(reader)
    meanZeroOneError = [0]*6
    meanBaseLineError = [0]*6
    stdDevZeroOneError = [0]*6 
    stdDevBaseLineError = [0]*6  
    #Keeping Feature constant =500 and varying percentage training data
    for k in range(0,6):
        zeroOneError = []
        baseLineError = []
        print("Running Loop # ",k)
        for j in range(0,10):
            lengthTrainingData =int((totalLen * percentageTrainingData[k])/100)

            your_list = random.sample(original_file,lengthTrainingData)
            test_list = [x for x in original_file if x not in your_list]
                       
            your_list = lowerCase(your_list)
            your_list = remove_Punctuation(your_list)
            wordList = []
            wordList = stringToListofWords(your_list, wordList)
            sortedWordList = wordList         #sorted(wordList)
        
            numberOfFeature = numberOfFeatureList[3]
            featureListWithCount = set(Counter(sortedWordList).most_common(numberOfFeature+100)) - set(Counter(sortedWordList).most_common(100)) 
            sortedFeatureList = sorted(featureListWithCount, key=lambda tup: tup[1], reverse=True)
            #sortedFeatureList = sorted(featureListWithCount, key=operator.itemgetter(1,0), reverse=True)
            top10Words = sortedFeatureList[0:10]
            featureList = [i[0] for i in sortedFeatureList]
            matrix = []
            matrix = createMatrix(featureList, numberOfFeature, your_list)
            probList = []
            probList,classLabelProb = calculateProbability(numberOfFeature, matrix)
            
            #For Base Line Loss
            classPredictedBaseLine = baseLineClassPredicted(numberOfFeature, matrix)
            
            test_list = lowerCase(test_list)
            test_list = remove_Punctuation(test_list)
            testMatrix = []
            testMatrix = createMatrix(featureList, numberOfFeature, test_list)
            predictedClassList = predictClass(testMatrix, numberOfFeature, probList, classLabelProb)
            noOfErrors = errorCalculation(predictedClassList, testMatrix, numberOfFeature)
            zeroOneError.append((noOfErrors)/(len(testMatrix)))
            
            #For Base Line Loss
            baseLineError.append(baseLineLoss(numberOfFeature, testMatrix, classPredictedBaseLine))
        print("ZERO-ONE-LOSS", zeroOneError)
        print("BASE-LINE-LOSS", baseLineError)
        
        meanZeroOneError[k] = numpy.mean(zeroOneError)
        meanBaseLineError[k] = numpy.mean(baseLineError)
        stdDevZeroOneError[k] = numpy.std(zeroOneError)
        stdDevBaseLineError[k] = numpy.std(baseLineError)

    print("Percentage Vary ")
    print(percentageTrainingData)
    print(meanZeroOneError)
    print(stdDevZeroOneError)
    print(meanBaseLineError)
    print(stdDevBaseLineError)
    
    plt.errorbar(percentageTrainingData,meanZeroOneError,stdDevZeroOneError,marker=".",label="Naive Bayes Zero One Error")
    plt.errorbar(percentageTrainingData,meanBaseLineError,stdDevBaseLineError,marker=".",label="BaseLine Error")
    plt.legend()
    plt.xlabel("Percentage Training Data")
    plt.ylabel("Mean Zero One Error")
    plt.title("Error Curve for [1, 5, 10, 20, 50, 90]")
    plt.savefig('PercentageVary.jpg')'''


               
    
