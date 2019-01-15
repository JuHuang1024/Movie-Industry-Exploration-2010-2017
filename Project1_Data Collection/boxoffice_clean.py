#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:51:19 2018

@author: Xinyue Liu, Ju Huang, Tianyi Yang, Zikai Zhu
"""

'''
Boxoffice data contains 11 attributes. These 11 attributes are in three types: Numeric, string, date
1. Fill in the missing values:
    Fill in the missing valule with -1
2. Convert the noise values:
    2.1 Numeric data:
    Remove the '$',',', then convert the data into a target data type
    2.2 String data:
    No noise values
    2.3 Date data:
    Convert the date data into date type
    

'''



import pandas as pd
import numpy as np

b = pd.read_csv('boxoffice.csv', sep=',', encoding='latin-1')
    


# FOR ALL cols: fill na with '-1' 
def fillNA(df):
    df=df.fillna(-1)
    return df

# numerical cols: clean '$',',', etc
def cleanNum(df):
    numeric_cols=[['id',int],['Year',int],['Rank',int],['Total Gross',float],['All Theaters',int],['Opening',int],['Opening Theaters',int]]
    for i in numeric_cols:
        name=i[0]
        dtype=i[1]
        try:
            df[name]=df[name].str.replace(',', '')
            df[name]=df[name].str.replace('$', '')
            df[name]=df[name].astype(dtype)
        except:
            pass
        
        # drop all rows which is still not numeric 
#        df[df.name.apply(lambda x: x.isnumeric())]
    return df

def cleanDate(df):
    df['Open']=df['Open'].astype(str)+'/'+ df['Year'].astype(str)
    df['Close']=df['Close'].astype(str)+'/'+ df['Year'].astype(str)
    return df


b=fillNA(b)
b=cleanNum(b)
b=cleanDate(b)
b.to_csv('boxoffice_cleaned.csv',sep=',',index = False)









