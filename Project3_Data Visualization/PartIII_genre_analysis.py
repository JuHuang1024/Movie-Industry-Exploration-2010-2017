#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 15:35:03 2018

@author: Xinyue Liu, Tianyi Yang, Zikai Zhu, Ju Huang
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#########################################
# Genre Analysis by year
#########################################
sns.set(font_scale=1)
sns.set_style("whitegrid")
def genre_df(f, yr):
    # Reorganize genre data of year yr into sparse dataset
    f = f[f["Year_x"]==yr]
    f1 = f[['id', 'Genre']]
    f2 = f[['id', 'Total Gross']]
    genredf = f1.apply(lambda x: pd.Series((x['Genre'].split(","))),axis=1)
    genredf['id'] = f1['id']
    genredf = genredf.set_index('id').stack().reset_index()
    del genredf["level_1"]
    genredf['genre'] = genredf[0].str.strip()
    del genredf[0]
    genredf2 = pd.merge(genredf, f2, on='id')
    #print(g)    
    return genredf2
  
def sum_NNA(l):
    # find the sum of not NA items in the list
    a = np.array(l)
    valid = a > -0.0001
    if valid.any():
        return a[valid].sum()
    else:
        return 0
    
def count(l):
    # count items in the list
    a = np.array(l)
    return len(a)

def topn_gbdatalist(f, n):
    datalist = []
    for y in range(2010, 2018):
        # From 2010 to 2018, for each year calculate movie count by genre and boxoffice by genre
        gdf = genre_df(final, y)
        gdf2 = gdf.groupby('genre')[['id', 'Total Gross']].agg(lambda x: x.tolist()).reset_index()
        gdf2["Total Gross"] = gdf2["Total Gross"].apply(lambda x: [int(i) for i in x])
        gdf2["sum"] = gdf2["Total Gross"].apply(sum_NNA)
        s = gdf2["sum"].sum()
        gdf2["Box Office by Genre (%)"] = gdf2["sum"]/s
        gdf2["cnt"] = gdf2["id"].apply(count)
        s2 = gdf2["cnt"].sum()
        gdf2["Number of Movies by Genre (%)"] = gdf2["cnt"]/s2
        # Sort by boxoffice percentage of each genre and select top n genre of that year
        g = gdf2.sort_values(["Box Office by Genre (%)"], ascending=[False])
        gtn = g[:n]
        if datalist == []:
            datalist = [gtn]
        else:
            datalist.append(gtn)
    return datalist

# FInal dataset
def topn_gbdata(f, n):
    # Aggregate top n genres' data and create dataframe
    data = pd.concat(topn_gbdatalist(f,n)).reset_index()
    yearlist = pd.DataFrame([2010]*n+[2011]*n+[2012]*n+[2013]*n+[2014]*n+[2015]*n+[2016]*n+[2017]*n)
    data["Year"] = yearlist
    return data

if __name__ == "__main__":
    # Read Data
    final = pd.read_csv('final.csv', encoding='utf-8',sep=',')
    
    # This dataframe has each year's top n movie genre, each genre's boxoffice and count
    # boxoffice is normalized by calculating % of boxoffice of each genre over total boxoffice
    # count is normalized by calculating % of movies in each genre over total # of movies
    genre_data = topn_gbdata(final,5)
    
    # Plot 1: Genre Count by Year
    fig, ax = plt.subplots(figsize=(12,8))
    ax = sns.barplot(x="Year", y="Number of Movies by Genre (%)", hue="genre", data=genre_data)
    ax.set_title("Top 5 Most Produced Genres by Year")
    ax.legend(bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.)
    plt.setp(ax.get_legend().get_texts(), fontsize='12')
    plt.savefig('genre_count.png')
    
    # Plot 2: Genre Boxoffice by Year
    fig2, ax2 = plt.subplots(figsize=(12,8))
    ax2 = sns.barplot(x="Year", y="Box Office by Genre (%)", hue="genre", data=genre_data)
    ax2.set_title("Top 5 Most Profitable Genres by Year")
    ax2.legend(bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.)
    plt.setp(ax2.get_legend().get_texts(), fontsize='12')
    plt.savefig('genre_boxoffice.png')
    
    
    

  
