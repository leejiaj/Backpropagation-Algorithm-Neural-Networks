# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:57:36 2017

@author: leejia
"""
import pandas as pd
import sys
import random
import math

#!/usr/bin/python 


if __name__ == '__main__':
    global inputDataset
    inputDataset = sys.argv[1:][0]
    global trainingPercent 
    trainingPercent = int(sys.argv[1:][1])
    global maximumIterations
    maximumIterations = int(sys.argv[1:][2])
    numberHiddenLayers = int(sys.argv[1:][3])
    hiddenLayerLenList = []
    for i in range(numberHiddenLayers):
        hiddenLayerLenList.append(int(sys.argv[1:][4+i]))
    
    # Reading input dataset 
    file=open(inputDataset)
    inputDataset_df=pd.read_csv(inputDataset, skip_blank_lines=True).dropna()
    
    totalNumTrainingDataPoints = int(inputDataset_df.shape[0]*trainingPercent/100)
    totalNumTestDataPoints = inputDataset_df.shape[0] - totalNumTrainingDataPoints
    
    #Splitting to training and test dataset
    trainingDataset_df = inputDataset_df[0:totalNumTrainingDataPoints]
    testDataset_df = inputDataset_df[totalNumTrainingDataPoints:]
    
    layerLenList = []
    layerLenList.append(inputDataset_df.shape[1]-1)
    for i in range(len(hiddenLayerLenList)):
        layerLenList.append(hiddenLayerLenList[i])
    layerLenList.append(1)
    l, inp, out = max(layerLenList)*max(layerLenList), max(layerLenList)*max(layerLenList), max(layerLenList)*max(layerLenList)
    weight = [[[0 for x in range(out)] for y in range(inp)] for z in range(l)]
    weightUpdate = [[[0 for x in range(out)] for y in range(inp)] for z in range(l)]
    
    #Initializing all network weights to small random numbers (between -0.05 and 0.05)
    for i in range(numberHiddenLayers+1):
        for j in range(layerLenList[i]+1):
            for k in range(1,layerLenList[i+1]+1):
                weight[i][j][k] = round(random.uniform(-0.05,0.05),2)
    
    error = 1
    node = [[0 for x in range(out)] for y in range(inp)]
    errorTerm = [[0 for x in range(out)] for y in range(inp)]
    attributes_list = list(trainingDataset_df)
    attributes_list.remove('Class')
    numTrainingDataPoints = totalNumTrainingDataPoints 
    # Until termination condition is met, do
    while(maximumIterations > 0 and error != 0):
        maximumIterations = maximumIterations - 1
        currentDataPoint = 0
        # For each data points, do   
        while(numTrainingDataPoints > 0):
            numTrainingDataPoints = numTrainingDataPoints - 1
            #Forward Pass
            for i in range(numberHiddenLayers+2):
                node[i][0] = 1
                for j in range(1,layerLenList[i]+1):
                    if (i==0):
                        node[i][j] = trainingDataset_df.iloc[currentDataPoint][attributes_list[j-1]]
                    else:
                        net = 0                        
                        for k in range(layerLenList[i-1]+1):
                            net = net + node[i-1][k] * weight[i-1][k][j]
                        node[i][j] = 1 / ((1 + math.exp(-net)))                        
            #Backward Pass
            for i in range(numberHiddenLayers+1,-1,-1):
                for j in range(0,layerLenList[i]+1):
                    if (i==numberHiddenLayers+1):
                        if j < layerLenList[i]:
                            errorTerm[i][j+1] = node[i][j+1]*(1-node[i][j+1])*(trainingDataset_df.iloc[currentDataPoint]['Class']-node[i][j+1])
                            error = errorTerm[i][j+1]
                            if error == 0:
                                break
                    else:
                        neterror = 0
                        learningFactor = 1
                        for k in range(1,layerLenList[i+1]+1):
                            neterror = neterror + weight[i][j][k]*errorTerm[i+1][k]
                            weightUpdate[i][j][k] = learningFactor*node[i][j]*errorTerm[i+1][k]
                        errorTerm[i][j] = node[i][j]*(1-node[i][j])*neterror
                if error == 0:
                    break
            #Updating weights
            if error != 0:                
                for i in range(numberHiddenLayers+1):
                    for j in range(layerLenList[i]+1):
                        for k in range(1,layerLenList[i+1]+1):
                            weight[i][j][k] = weight[i][j][k] + weightUpdate[i][j][k]
            currentDataPoint = currentDataPoint + 1
 
    #Printing all weights
    for i in range(numberHiddenLayers+1):
        print(" ")
        if (i == 0):
            print("Layer ",i, "(Input Layer):")
        elif (i == numberHiddenLayers):
            print("Layer ",i, "(Last hidden layer):")
        else:
            print("Layer ",i, "(",i,"st hidden layer):")  
        print("--------------------------------------")
        for j in range(layerLenList[i]+1):
            if(j == 0):
                print("Bias term weights: ")
            else:
                print("Neuron ",j," weights: ")
            for k in range(1,layerLenList[i+1]+1):
                print("W",i,j,k," : ",weight[i][j][k])
                
    #Calculating Training Error
    numTrainingDataPoints = totalNumTrainingDataPoints 
    currentDataPoint = 0
    squaredErrorAndSum = 0
    while(numTrainingDataPoints > 0):
        numTrainingDataPoints = numTrainingDataPoints - 1
        for i in range(numberHiddenLayers+2):
            node[i][0] = 1
            for j in range(1,layerLenList[i]+1):
                if (i==0):
                    node[i][j] = trainingDataset_df.iloc[currentDataPoint][attributes_list[j-1]]
                else:
                    net = 0                        
                    for k in range(layerLenList[i-1]+1):
                        net = net + node[i-1][k] * weight[i-1][k][j]
                    node[i][j] = 1 / ((1 + math.exp(-net)))
                output = node[i][j]
        squaredErrorAndSum = squaredErrorAndSum + math.pow((trainingDataset_df.iloc[currentDataPoint]['Class'] - output),2)
        currentDataPoint = currentDataPoint + 1
    meanSquareError = (1/totalNumTrainingDataPoints)*squaredErrorAndSum
    print(" ")
    print("Total training error = ",meanSquareError)
    
    
    #Calculating Test Error
    numTestDataPoints = totalNumTestDataPoints 
    currentDataPoint = 0
    squaredErrorAndSum = 0
    while(numTestDataPoints > 0):
        numTestDataPoints = numTestDataPoints - 1
        for i in range(numberHiddenLayers+2):
            node[i][0] = 1
            for j in range(1,layerLenList[i]+1):
                if (i==0):
                    node[i][j] = testDataset_df.iloc[currentDataPoint][attributes_list[j-1]]
                else:
                    net = 0                        
                    for k in range(layerLenList[i-1]+1):
                        net = net + node[i-1][k] * weight[i-1][k][j]
                    node[i][j] = 1 / ((1 + math.exp(-net)))
                output = node[i][j]
        squaredErrorAndSum = squaredErrorAndSum + math.pow((testDataset_df.iloc[currentDataPoint]['Class'] - output),2)
        currentDataPoint = currentDataPoint + 1
    meanSquareError = (1/totalNumTestDataPoints)*squaredErrorAndSum
    print(" ")
    print("Total test error = ",meanSquareError)
        
  