#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:00:34 2018

@author: Xinyue Liu, Tianyi Yang, Ju Huang, Zikai Zhu
"""
import pandas as pd
import datetime
import numpy as np
from scipy import stats
import scipy.special as scsp
import matplotlib.pyplot as plt 
################################################
# IGNORE the Warnings!
import warnings
warnings.filterwarnings("ignore")
################################################
################################################
# Data Cleaning Functions for columns
################################################
def convert_hours_to_mins(r):
    # Convert 'run & min' format data in Runtime column to 'min' format
    if 'h' in r:
        mins = float(str(r).split(' ')[0])*60+float(str(r).split(' ')[2])
        return int(mins)
    return int(r)

def calculate_timespan(opentime, closetime):
    # Calculate Timespan = Close time - open time
    if '-/' in closetime or '-/' in opentime:
        return np.NaN
    elif pd.to_datetime(closetime) >= pd.to_datetime(opentime):
        t = pd.to_datetime(closetime) - pd.to_datetime(opentime)
        if(t==0):
            return 1
        else:
            return pd.to_datetime(closetime) - pd.to_datetime(opentime)
    else:
        return pd.to_datetime(closetime) - pd.to_datetime(opentime)+datetime.timedelta(days=365)

def group_studios(b1):
    # Reduce number of categories for studio column through grouping
    # Indie:There are 411 studios that produced fewer than 10 movies. These 80% of the studios produced around 20% of the movies 
    # Midsized: These 56 studios that produced between 10 to 40 movies. These 10% of the studios produce another 20% of the movies
    studios= b1['Studio'].value_counts()
    indie=b1['Studio'].isin(studios.index[studios<10])
    midsize=b1['Studio'].isin(studios.index[(studios<40)&(studios>=10)])
    b1['Studio_new']= b1['Studio']
    b1.loc[indie,'Studio_new']='Indie'
    b1.loc[midsize,'Studio_new']='Mid_sized'
    b1['Studio_new'].value_counts()
    return b1

######################################## 
# Functions to Detect and Manage NA Values
########################################
def condfill(rankmin, rankmax, yr, col):
    # Fill NA values of any numeric col with mean grouped by movie ranking and year
    cond = (final['Rank'] >= rankmin) & (final['Rank'] <= rankmax) & (final['Year_x'] == yr)
    condmean = np.trunc(final.loc[cond, col].mean(skipna=True))
    final.loc[cond, col] = final.loc[cond, col].fillna(condmean)
    
def create_combined_rating(f):
    # Use Z-score to combine three rating cols
    f['Metascore'] = f['Metascore'].astype(float) / 10
    f['Rotten Tomatoes'] = f['Rotten Tomatoes'].astype(float) / 10
    f['imdbRating'] = f['imdbRating'].astype(float)
    f['M_z'] = (f['Metascore']-f['Metascore'].mean(skipna=True))/f['Metascore'].std(skipna=True)
    f['R_z'] = (f['Rotten Tomatoes']-f['Rotten Tomatoes'].mean(skipna=True))/f['Rotten Tomatoes'].std(skipna=True)
    f['I_z'] = (f['imdbRating']-f['imdbRating'].mean(skipna=True))/f['imdbRating'].std(skipna=True)
    
    f['combined_rating'] = scsp.ndtr(f[['M_z', 'R_z', 'I_z']].mean(axis = 1))
    return f

######################################## 
# Outlier Detection
########################################
def outliers_detect(final):
    # describe the data
    # select the numeric columns
    numeric_column = ['Rank','Total Gross', 'All Theaters','Opening', 
                      'Opening Theaters', 'Timespan', 'Metascore', 
                      'Rotten Tomatoes',
                      'imdbRating', 'imdbVotes','combined_rating']
    final_numeric = final[numeric_column]
    #describe the numeric columns
    final_numeric_describe = final_numeric.describe()
    print(final_numeric_describe)
    categorical_column =['Studio_new', 'Rated']
    final_categorical = final[categorical_column]
    final_catrgorical_mode=stats.mode(final_categorical)[0][0]
    print(final_catrgorical_mode)

    #create boxplots for every numeric columns to see the outliers
    final_numeric.plot(kind='box', subplots=True, layout=(4, 3), sharex=False, sharey=False, figsize=(9, 12))
    plt.show()
    # plt.savefig('boxplot.png')

    #see movie names of the 3 extreme large outliers of total gross, all theaters, opening, opening theaters, imdbVotes
    excellent_movie_performance = ['Total Gross', 'All Theaters', 'Opening', 'Opening Theaters', 'imdbVotes']
    for i in excellent_movie_performance:
        print(i,':','\n',final.nlargest(3,i)['Name'],'\n')

    #see movie names of the 3 extreme large and small outliers of Metascore and imdbRating
    movie_score = ['Metascore', 'imdbRating']
    #convert the scores into negative numbers in order to select the 3 smallest outliers of them
    final['Metascore_-1'] = final['Metascore']*(-1)
    final['imdbRating_-1'] = final['imdbRating']*(-1)
    movie_score_min = ['Metascore_-1', 'imdbRating_-1']
    #see the outliers of scores
    print('\nExcellent outliers')
    for i in movie_score:
        print(i,':','\n',final.nlargest(3,i)['Name'],'\n')
    print('\nPoor outliers')
    for i in movie_score_min:
        print(i,':','\n',final.nlargest(3,i)['Name'],'\n')    


    #see movie names of the 3 extreme small outliers of timespan
    #convert the timespan into negative numbers in order to select the 3 smallest outliers of them
    final['Timespan_-1'] = final['Timespan']*(-1)
    #see the top 3 extremely large outliers
    print('\nExtremely large outliers')
    print('Timespan:','\n',final.nlargest(3,'Timespan')['Name'],'\n')
    #see the top 3 extremely small outliers
    print('\nExtremely small outliers')
    print('Timespan_-1:','\n',final.nlargest(3,'Timespan_-1')['Name'],'\n')
    #select movies that timespan<=5, ranked as top 200. Try to figure out the reasons.
    timespan_outliers = final[(final['Timespan'].values<=5) & (final['Rank'].values <= 200)]
    print(timespan_outliers)
    #look into them and deal with them one by one
    #reasons will be explained specifically in report.
    final[final['Name'] == 'The Dark Knight (2012 re-release)']['Timespan']='7'
    final[final['Name'] == '20 Feet from Stardom']['Timespan']='370'
    final[final['Name'] == 'Batman: The Killing Joke']['Timespan']='366'
    final[final['Name'] == 'Newsies: The Broadway Musical']['Timespan']='5'
    return final

def binning(final): 
    ### Bin the Total Gross
    print('\n\nNumber of values in each bin:\n')
    names = [1,2,3,4,5,6]
    bins=[0, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9]
    final['Total Gross Bin'] = pd.cut(final['Total Gross'], bins, labels=names)
    #Look at new bins
    totalGrossBins = np.digitize(final['Total Gross'],bins)
    # Count the number of values in each bin
    totalGrossBinCounts = np.bincount(totalGrossBins)
    print("Total Gross Bin", totalGrossBinCounts)

    ### Bin the All Theaters
    names = [1,2,3,4]
    bins=[0, 10, 100, 1000, 10000]
    final['All Theaters Bin'] = pd.cut(final['All Theaters'], bins, labels=names)
    #Look at new bins
    allTheatersBins = np.digitize(final['All Theaters'],bins)
    # Count the number of values in each bin
    allTheatersBinCounts = np.bincount(allTheatersBins)
    print("All Theaters Bin", allTheatersBinCounts)

    ### Bin the Opening
    names = [1,2,3,4,5,6]
    bins=[0, 3*10**3, 3*10**4, 3*10**5, 3*10**6, 3*10**7, 3*10**8] #### opening nan need to be filled (eg. id-147) ###########################
    final['Opening Bin'] = pd.cut(final['Opening'], bins, labels=names)
    #Look at new bins
    OpeningBins = np.digitize(final['Opening'],bins)
    # Count the number of values in each bin
    OpeningBinCounts = np.bincount(OpeningBins)
    print("Opening Bin", OpeningBinCounts)
    
    ### Bin the Opening Theaters
    names = [1,2,3,4]
    bins=[0, 5, 50, 500, 5000] 
    final['Opening Theaters Bin'] = pd.cut(final['Opening Theaters'], bins, labels=names)
    #Look at new bins
    openingTheatersBins = np.digitize(final['Opening Theaters'],bins)
    # Count the number of values in each bin
    openingTheatersBinCounts = np.bincount(openingTheatersBins)
    print("Opening Theaters Bin", openingTheatersBinCounts)
    
    ### Bin the Timespan
    names = [1,2,3,4,5,6]
    bins=[0,7, 30, 60, 90, 120, 500] 
    final['Timespan Bin'] = pd.cut(final['Timespan'], bins, labels=names)
    #Look at new bins
    TimespanBins = np.digitize(final['Timespan'],bins)
    # Count the number of values in each bin
    TimespanBinCounts = np.bincount(TimespanBins)
    print("Timespan Bin", TimespanBinCounts)
    
    ### Bin the Metascore
    names = [1,2,3,4]
    bins=[0,3,6,8,10]                       ####contain missing values
    final['Metascore Bin'] = pd.cut(final['Metascore'], bins, labels=names)
    #Look at new bins
    MetascoreBins = np.digitize(final['Metascore'],bins)
    # Count the number of values in each bin
    MetascoreBinCounts = np.bincount(MetascoreBins)
    print("Metascore Bin", MetascoreBinCounts)
    ### Seems people in Metascore tend to rate at medium scores.
    
    ### Bin the Rotten Tomatoes
    names = [1,2,3,4]
    bins=[0,3,6,8,10]                          ####contain missing values
    final['Rotten Tomatoes Bin'] = pd.cut(final['Rotten Tomatoes'], bins, labels=names)
    #Look at new bins
    rottenTomatoesBins = np.digitize(final['Rotten Tomatoes'],bins)
    # Count the number of values in each bin
    rottenTomatoesBinCounts = np.bincount(rottenTomatoesBins)
    print("Rotten Tomatoes Bin", rottenTomatoesBinCounts)
    ### Seems people in Rotten Tomatoes tend to rate evenly in each score.
        
    ### Bin the combined_rating
    names = [1,2,3,4]
    bins=[0, 0.3, 0.6, 0.8, 1] 
    final['combined_rating Bin'] = pd.cut(final['combined_rating'], bins, labels=names)
    #Look at new bins
    combined_ratingBins = np.digitize(final['combined_rating'],bins)
    # Count the number of values in each bin
    combined_ratingBinCounts = np.bincount(combined_ratingBins)
    print("Combined Rating Bin", combined_ratingBinCounts)
        
    ### Bin the imdbVotes
    names = [1,2,3,4,5]
    bins=[0, 5*10**2, 5*10**3, 5*10**4, 5*10**5, 5*10**6]    ####  nan need to be filled  ###########################
    final['imdbVotes Bin'] = pd.cut(final['imdbVotes'], bins, labels=names)
    #Look at new bins
    imdbVotesBins = np.digitize(final['imdbVotes'],bins)
    # Count the number of values in each bin
    imdbVotesBinCounts = np.bincount(imdbVotesBins)
    print("imdbVotes Bin", imdbVotesBinCounts)
    
    return final
    
if __name__ == "__main__":
    # Read Data
    r = pd.read_csv('ratings_cleaned.csv', encoding='utf-8',sep=',').replace(-1, np.NaN)
    b = pd.read_csv('boxoffice_cleaned.csv', encoding='latin-1',sep=',').replace(-1, np.NaN)
    
    # Data Cleaning
    # Update runtime col so that all runtime values are in unit min
    r['Runtime'] = r['Runtime'].map(convert_hours_to_mins)
    # Create new col check_na to check if all at least one rating is exist for each movie
    r['check_na'] = r[['Metascore', 'Rotten Tomatoes', 'imdbRating']].mean(axis = 1)
    # Create new column Timespan based on movie opening time and close time
    b['Timespan'] = b.apply(lambda x: calculate_timespan(x['Open'], x['Close']), axis = 1)
    b['Timespan'] = b['Timespan'].dt.days
    # Aggregate small studios into one for studio column
    b = group_studios(b)
    
    
    # Manage NA values
    final = b.merge(r,left_index=True,right_index=True)
    # Replace -1 in Rated to 'Not Rated'
    final['Rated']=final['Rated'].replace('-1','NOT RATED')
    # Delete rows that all three ratings are NA
    final = final[np.isfinite(final['check_na'])]
    # Since there are still many missing values in 3 rating columns, we use z-score to combine three rating columns
    final = create_combined_rating(final)
    # Replace NA by group mean value for column All Theaters, Opening Theaters, Timespan and Runtime 
    # Group by Rank 0-200, 201-400, 401-600, 601-800 and each year
    for y in range(2010, 2018):
        for c in ['Timespan', 'All Theaters', 'Opening Theaters', 'Runtime']:
            condfill(0,100,y,c)
            condfill(101,200,y,c)
            condfill(201,300,y,c)
            condfill(301,400,y,c)
            condfill(401,500,y,c)
            condfill(501,600,y,c)
            condfill(601,700,y,c)
            condfill(701,800,y,c)
    
    # Outlier Detection
    final = outliers_detect(final)
    # Bin total gross, all theaters opening, opening theaters timespan, metascore, rotten tomatoes, combined ratings, imdbvotes 
    final = binning(final)
    
    final_tmp = final

    # Keep the columns we need for analysis
    cols_to_keep=['id', 'Year_x', 'Rank', 'Name', 'Total Gross', 'All Theaters',
       'Opening', 'Opening Theaters', 'Timespan',
       'Studio_new', 'Plot','Genre', 'Metascore', 'Rated', 'Rotten Tomatoes', 'Runtime',
       'imdbRating', 'imdbVotes', 'combined_rating', 
       'Total Gross Bin', 'All Theaters Bin', 'Opening Bin',
       'Opening Theaters Bin', 'Timespan Bin', 'Metascore Bin',
       'Rotten Tomatoes Bin', 'combined_rating Bin', 'imdbVotes Bin']
    #final = final[cols_to_keep].replace('-1','NA')
    final = final.replace(-1,'NA')
    
    # Output data to csv for further use
    final.to_csv("final.csv", index = False)