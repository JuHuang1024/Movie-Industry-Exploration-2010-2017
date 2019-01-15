#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 00:51:19 2018

@author: Ju Huang, Xinyue Liu, Tianyi Yang, Zikai Zhu

Description of file: 
    This script takes 2 files as input: original 'rating' document & the cleaned 'rating' document.
    It will calcualte the score of 'cleaniness' for each file as output. 
    
How we calculate the score for each col:
    The total score for each column is 100. 
    We judge a column by up to n criteria. Each criteria is given a weight of 1/n. 
    These criterias include: 
        - 1. % of noise: score=  100- % of wierd values *100  
        - 2. % of missing values: score = 100- % of missing values * 100
        - 3. If the column is irrelevant/ useless to our furture analysis: score=100-100=0 
    Finally, we calculate the avg. score of the dataframe 
"""

import pandas as pd
import numpy as np
import statistics
#from pprint import pprint

######################################## Reading in the two dataframe 
#the original, unclenaed df
r = pd.read_csv('ratings.csv', encoding='latin-1',sep=',')
#the new, cleaned df 
r_cleaned=pd.read_csv('ratings_cleaned.csv', encoding='latin-1',sep=',')


####################################### Write a program to calculate the score 
#define a function that calculate % missing value for a column 
def num_missing(df,col_name):
    num_missing=df[col_name].isna().sum()/len(df)*100
    return(num_missing)

#define a function to find noise: if a cell is of a col with 'col_name' can't be converted to the desired datatype (input 'd_type'), then add to the count 
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

#Main Program: get the total score deduction of a dataframe 
def get_final_deduction(df):
    #For each col, the score deduction starts at 0
    score_deduction=dict()
    for col_name in list(df):
        score_deduction[col_name]=0
        
    #get scores_deduction for all numeric cols  
    #for numeric cols, two checks: 1. missing values 2. noise
    for i in numeric_cols:
        try:
            col_name=i[0]
            d_type=i[1]
            weight_missing=1/2
            deduct_missing= weight_missing*num_missing(df,col_name)
            weight_noise= 1/2
            deduct_noise= weight_noise*num_noise(df,col_name,d_type)
            score_deduction[col_name]=deduct_missing+deduct_noise
        except:
            pass
        
    #get score_deduction for all irrelevant cols= 100
    for i in irrelevant_cols:
        try:
            col_name=str(i)
            score_deduction[col_name]=100
        except:
            pass
        
    #get scores_deduction for all string cols
    for i in string_cols:
        try:
            col_name=str(i)
            score_deduction[col_name]=num_missing(df,col_name)
        except:
            pass
        
    #for categorical cols, need to look at the numeric values and give scores manually (with the following 3 lines)
    #for i in categorical_cols:
    #    col_name=str(i)
    #    print(col_name,'\n',r[col_name].value_counts())
    
    #For categorical column 'Rated', there're duplicated values: 'NOT RATED' = "UNRATED'= 'NR'
    #count the # of UNRATED& NR as # of noise
    weight_noise=1/2
    deduct_dup=(len(df['Rated'][df['Rated']=='UNRATED'])+len(df['Rated'][df['Rated']=='UNRATED']))/len(df)*100
    weight_missing=1/2
    deduct_missing=weight_missing* num_missing(df,'Rated')
    score_deduction['Rated']=deduct_dup*weight_noise+deduct_missing*weight_missing
    
    #For categorical column  'Reponse', there're 2 noise values: 'Movie not found!' & 'Year not match!'
    #count the # of 'Movie not found!' & 'Year not match!' as noise 
    weight_noise=1/2
    deduct_noise=(len(df['Response'][df['Response']=='Movie not found!'])+len(df['Response'][df['Response']=='Year not match! ']))/len(df)*100
    weight_missing=1/2
    deduct_missing=weight_missing* num_missing(df,'Response')
    score_deduction['Response']=deduct_noise*weight_noise+deduct_missing*weight_missing
    
    #get scores_deduction for all date cols
    for i in date_cols:
        col_name=str(i)
        #Check 1/2: missing values
        weight_missing=1/2
        deduct_missing=num_missing(df,col_name)*weight_missing
        #check 2/2: check noise: # of cols can't be converted to data format 
        weight_noise=1/2
        counter_noise=1
        if pd.notnull(i):        
            try:
                pd.to_datetime(r[col_name],infer_datetime_format=True)
            except Exception as e:
                counter_noise+=1
        deduct_noise=weight_noise*(counter_noise/len(df)*100)
        deduct_total=deduct_missing+deduct_noise
        score_deduction[col_name]=deduct_total
    #pprint(score_deduction)
    
    #get the score_deduction for the df: avg. of all columns deduction
    avg_score_per_col=statistics.mean(score_deduction.values())
    #print(score_deduction)
    return (avg_score_per_col)


####################################### Calculate score for each dataframe 
   
#categorize all the cols in the orginal dataframe 
numeric_cols=[['BoxOffice',float],['Internet Movie Database',float],['Metacritic',int],['Metascore',int],['Rotten Tomatoes',int],['Runtime',int],['imdbRating',float],['imdbVotes',int],['totalSeasons',int]]
irrelevant_cols=['DVD','Type','Website','Poster'] 
categorical_cols=['Rated','Response']
string_cols=['Actors','Awards','Country','Director','Language','Plot','Title','Writer','imdbID','Genre','Production']
date_cols=['Released','Year']
#calculate the score for the originla dataframe 
score_beforeclean=get_final_deduction(r)
print('The score before cleaning is ',100-score_beforeclean)

#categorize all the cols in the cleaned, new dataframe 
numeric_cols=[['BoxOffice',float],['Metascore',int],['Rotten Tomatoes',int],['Runtime',int],['imdbRating',float],['imdbVotes',int],['totalSeasons',int]] 
categorical_cols=['Rated','Response']
irrelevant_cols=[]
string_cols=['Actors','Awards','Country','Director','Language','Plot','Title','Writer','imdbID','Genre','Production']
date_cols=['Released','Year']
score_afterclean=get_final_deduction(r_cleaned)
#calculate the score for the new, cleaned dataframe
print('The score after cleaning is ',100-score_afterclean)

