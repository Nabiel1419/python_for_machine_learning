# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 10:09:01 2022

@author: nabiel akber
@mobile:8082197357
@about file:this file helps us to read files and load dataset in python using pandas module

"""
import pandas as pd
#import pandas module it has some inbuilt function that help us in loading the dataset and also datacleaning 
# pd is alias name for pandas
# now we have various methods to read file 
# what function we are going to use depends upon the type of file that is to be read
# here we are taking only two examples 
# 1. reading csv file(using read_csv(args) )function
# 2. reading excel file(using read_excel(args) function)
# both above mentioned functions takes many arguments but one that is most important 
# is the path of file
# we have to specific path  of the file
#  copy path  of the file make sure you have remembered the extension of the file

dataset=pd.read_csv(r"C:\Users\nabiel\Downloads\student.csv")
#here r is used to remove unicode error speicifies read operation
    
print(dataset)

#similarly you can use function to read excel file
dataset2=pd.read_excel(r"path of the file")
print(dataset2)


