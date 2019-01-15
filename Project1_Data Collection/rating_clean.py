#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:28:05 2018

This program is to clean the ratings.csv
First, fill all the na columns with -1
delete bad columns such as redundant and irrelevant columns
Then, clean the numerical columns by deleting non-numeric characters and convert them to numbers
Last, clean the value in categorical attributes such as 'Response' and 'Rated'

@author: Zikai Zhu, Xinyue Liu, Ju Huang, Tianyi Yang
"""


import pandas as pd

r = pd.read_csv('ratings.csv', encoding='latin-1',sep=',')

# FOR ALL cols: fill na with '-1' 
def fillNA(df):
    df=df.fillna(-1)
    return df

# delete irrelevant and redundant columns
def deleteBadCols(df):
    bad_cols=['Internet Movie Database','Metacritic','Type']
    for i in bad_cols:
        del df[i]
    return df
    
# for numerical cols: clean 'min','$',',','%', etc
def cleanNum(df):
    numeric_cols=[['BoxOffice',float],['Metascore',int],['Rotten Tomatoes',int],['Runtime',int],['imdbRating',float],['imdbVotes',int],['totalSeasons',int]]
    for i in numeric_cols:
        name=i[0]
        dtype=i[1]
        try:
            df[name]=df[name].str.replace(',', '')
            df[name]=df[name].str.replace('$', '')
            df[name]=df[name].str.replace('%', '')
            df[name]=df[name].str.replace(' min', '')
            df[name]=df[name].str.replace('&pound;', '')
            df[name] = df[name].astype(dtype)
        except:
            pass
    return df

# clean the categorical columns
# for 'rated': change 'UNRATED' and 'NR' to 'NOT RATED'
# for 'response': change 'Movie not found!' and 'Year not match!' to '-1'
def cleanCategory(df):
    df['Rated']=df['Rated'].str.replace('UNRATED', 'NOT RATED')
    df['Rated']=df['Rated'].str.replace('NR', 'NOT RATED')
    df.Response[df.Response=='Movie not found!']=-1
    df.Response[df.Response=='Year not match!']=-1
    return df

r=deleteBadCols(r)
r=cleanNum(r)
r=cleanCategory(r)
r=fillNA(r)
r.to_csv('ratings_cleaned.csv',sep=',')