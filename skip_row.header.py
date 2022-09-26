# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 14:44:18 2022

@author: nabiel akber 
@mobile:8082197357

"""

import pandas as pd
dataset=pd.read_csv(r"N:\student.csv",
                    nrows=4#this parameter specifies how many rows you are going to display
                    ,skiprows=1#this parameter is used to skip a row from the top,it is used when you 
                    #want to drop a row which is at the top of the excel or csv file which is of no use
                   #the R.H.S side of the skiprows is a value ,it is just a number 
                   # that specifies how many rows you are going to skip
                   
                    )
print(dataset)
df=pd.read_csv(r"N:\student.csv",header=1 ,nrows=5)# Above skiprows argument can be replaced 
# by header argument in above line of code it specifies that my header column is row no 1
# r.h.s of header specifies which row is goining to be header column of  the dataset
print(df)
    
    