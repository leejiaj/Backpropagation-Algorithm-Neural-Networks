# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:57:36 2017

@author: leejia
"""
import pandas as pd
import sys

#!/usr/bin/python 


if __name__ == '__main__':
    global inputDataset
    inputDataset = sys.argv[1:][0]
    global outputDataset 
    outputDataset = sys.argv[1:][1]
    global inputDataset_df
    
    # Reading input dataset 
    file=open(inputDataset)
    inputDataset_df=pd.read_csv(inputDataset, skip_blank_lines=True).dropna()
    
    
    attributes = inputDataset_df.columns
    attributes_list = list(set(attributes))
    numeric_attributes = inputDataset_df._get_numeric_data().columns
    numeric_attributes_list = list(set(numeric_attributes))
    if 'Class' in numeric_attributes_list:
        numeric_attributes_list.remove('Class');
    non_numeric_attributes_list = list(set(attributes) - set(numeric_attributes))
    
    # Removing data points with missing or incomplete features
    for i in range(len(non_numeric_attributes_list)):
        inputDataset_df = inputDataset_df[inputDataset_df[non_numeric_attributes_list[i]] != "?"]

    # Standardizing numeric values -> ie subtracting the mean from each of the values 
    # and dividing by the standard deviation
    for i in range(len(numeric_attributes_list)):
        mean = inputDataset_df[numeric_attributes_list[i]].mean()
        standard_deviation = inputDataset_df[numeric_attributes_list[i]].std()
        inputDataset_df[numeric_attributes_list[i]] = (inputDataset_df[numeric_attributes_list[i]] - mean)/standard_deviation
        
    # Converting categorical or nominal values to numerical values
    for i in range(len(non_numeric_attributes_list)):
        inputDataset_df[non_numeric_attributes_list[i]] = inputDataset_df[non_numeric_attributes_list[i]].astype('category')
        inputDataset_df[non_numeric_attributes_list[i]] = inputDataset_df[non_numeric_attributes_list[i]].cat.codes
    
    # Saving pre-processed dataset to the path specified in second command line argument
    inputDataset_df.to_csv(outputDataset, index=False, sep=',', encoding='utf-8')
    print("Pre-processed data in file: ",outputDataset)
  