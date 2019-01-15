#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:51:19 2018

@author: Xinyue Liu, Ju Huang, Tianyi Yang, Zikai Zhu

"""
'''
Boxoffice data contains 11 attributes. These 11 attributes are in three types: Numeric, string, date

Score deduction contains two parts: missing value deduction and noise deduction, each part weighs 50%.
1. Missing value deduction: 
    for each attribute, missing_value_deduction = percentage of missing values * 1oo * 50%
2. Noise value deduction: 
    noise_value_deduction = percentage of noise values *100 *50%
    2.1 Numeric data: 
        For each attribute, try convert the data into a target data type. If fails, take the certain data as a noise
    2.2 String data:
        no noise value
    2.3 data type:
        For each attribute, try conert the data into date type. IF fails, take the certain data as a noise

Score = 100-missing_value_deduction-noise_value_deduction
Final_Score = Score/numbers_of_attributes

'''

import pandas as pd
import numpy as np
import statistics

# read in the raw data csv and clean data csv to make comparison
b = pd.read_csv('boxoffice.csv', encoding='latin-1',sep=',')
b_cleaned=pd.read_csv('boxoffice_cleaned.csv', encoding='latin-1',sep=',')


# Fraction of missing values for each attributes
def num_missing(df,col_name):
    num_missing=df[col_name].isna().sum()/len(df)*100
    return(num_missing)

#try convert the col with 'col_name' to datatype (input 'd_type'), count how many times it failed
def num_noise(df,col_name,d_type):
    counter_noise=0
    for i in np.array(df[col_name]):    
        if pd.notnull(i):        
            try:
                d_type(i)
            except Exception as e:
                counter_noise+=1
    rate_noise=counter_noise/len(df)*100
    return (rate_noise)

numeric_cols=[['id',int],['Year',int],['Rank',int],['Total Gross',float],['All Theaters',int],['Opening',int],['Opening Theaters',int]]
string_cols=['Name','Studio']
date_cols=['Open','Close']

## Some pre-checks
# year
# check if all years are in range(2010, 2017)
b["Year"].unique()
# noise 0
 

# Rank
# check if all ranks is in range(1,800)
np.array_equal(b["Rank"], b["Rank"].astype(int))
b["Rank"].min()
b["Rank"].max()
# noise 0



# Studio
studio = b['Studio'].value_counts().to_dict()
studio1 = {k:v for k,v in studio.items() if v == 1}
studio1
# The studios in studio1 are "dangerous" studio name since they occur only once
# However, since there are many studio names only occur once
# it seems that it is common that one studio only produce one movie
# So we don't do anything to studio column now.

# Movie Name is unique for each of the movie(each row) so I skip this one


#Main checking function 
def get_final_score(df):
    #For each col, the score starts at 100
    score_deduction=dict()
    for col_name in list(df):
        score_deduction[col_name]=0
    #get scores_deduction for all numeric cols  
    #for numeric cols, two checks: 1. missing values 2. if all rows can be converted to a certain datatype
    for i in numeric_cols:
        try:
            col_name=i[0]
            d_type=i[1]
            weight_missing=1/2
            deduct_missing= weight_missing*num_missing(df,col_name)
            weight_noise= 1/2
            deduct_noise= weight_noise*num_noise(df,col_name,d_type)
            score_deduction[col_name]=deduct_missing+deduct_noise
#            print(col_name,score_deduction[col_name])
        except:
            pass
    #get scores_deduction for all string cols
    for i in string_cols:
        try:
            col_name=str(i)
            score_deduction[col_name]=num_missing(df,col_name)
#            print(col_name,score_deduction[col_name])
        except:
            pass
    #looking at date cols 
    for i in date_cols:
        col_name=str(i)
        #Check 1/2: missing values
        weight_missing=1/2
        deduct_missing=num_missing(df,col_name)*weight_missing
        #check 2/2: check # of cols can't be converted to data format 
        weight_noise=1/2
        counter_noise=1
        if pd.notnull(i):        
            try:
                pd.to_datetime(df[col_name],infer_datetime_format=True)
            except Exception as e:
                counter_noise+=1
        deduct_noise=weight_noise*(counter_noise/len(df)*100)
        #get total
        deduct_total=deduct_missing+deduct_noise
        score_deduction[col_name]=deduct_total
#        print(col_name,score_deduction[col_name])
    avg_score_per_col=statistics.mean(score_deduction.values())
    #print(score_deduction)
    return (avg_score_per_col)


score_beforeclean=get_final_score(b)
print('The score before cleaning is ',100-score_beforeclean)


### Calculate the score for cleaned data 
numeric_cols=[['id',int],['Year',int],['Rank',int],['Total Gross',float],['All Theaters',int],['Opening',int],['Opening Theaters',int]]
string_cols=['Name','Studio']
date_cols=['Open','Close']

score_afterclean=get_final_score(b_cleaned)
print('The score after cleaning is ',100-score_afterclean)
