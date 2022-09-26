# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:20:18 2022


@author: nabiel
"""

import pandas as pd

#dataframe=pd.read_csv(r"N:\album.csv",header=1,nrows=4)
#print(dataframe)
# Now we don't have header column in our dataset we want to insert New Header col
# let us see How we can insert new header col
dataframe=pd.read_csv(r"N:\album.csv", header=None,names=['YEAR','RANKING','ALBUM','WORLD_WISE_SALES(Est)','TRACKS','CDs','ALBUM_LENGTH','GENRE']
    )
print(dataframe)