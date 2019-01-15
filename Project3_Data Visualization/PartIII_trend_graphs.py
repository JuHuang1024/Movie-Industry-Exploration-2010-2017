#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 00:36:40 2018

@author: Xinyue Liu, Tianyi Yang, Zikai Zhu, Ju Huang
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate col by year by avg/cnt/sum:
def col_by_year(y_var, df, method, var):
    result = []
    for y in range(2010, 2018):
        f = df[df[y_var]==y]
        if method == "sum":
            result.append(f[var].sum())
        elif method == "avg":
            result.append(f[var].mean())
        elif method == "cnt":
            result.append(len(f[var]))
    return result

#########################################
# Trend of different Features by year
#########################################


if __name__ == "__main__":
    sns.set_style("darkgrid")
    # Read Data
    final = pd.read_csv('final.csv', encoding='utf-8',sep=',')
    final2 = pd.read_csv('boxoffice_cleaned.csv', encoding='utf-8',sep=',')
    
    df = pd.DataFrame({'Year': range(2010, 2018),
                       'Box Office': col_by_year("Year", final2, "sum", "Total Gross"),
                       'Number of Movies': col_by_year("Year", final2, "cnt", "Year"),
                       'Average Ratings': col_by_year("Year_x", final, "avg", "imdbRating")})
    
    # Plot 1: BoxOffice Year by Year
    fig, ax = plt.subplots()
    ax = sns.lineplot(x="Year", y="Box Office", data=df)
    ax.set(ylim=(9000000000, 12000000000))
    ax.set_title("Total Boxoffice by Year")
    plt.savefig('boxoffice_trend.png')
    
    # Plot 2: Number of Movies by Year
    fig2, ax2 = plt.subplots()
    ax2 = sns.lineplot(x="Year", y="Number of Movies", data=df)
    ax2.set(ylim=(500, 800))
    ax2.set_title("Number of Movies by Year")
    plt.savefig('n_movies_trend.png')
    
    # Plot 3: Average Rating of Movies by Year
    fig3, ax3 = plt.subplots()
    ax3 = sns.lineplot(x="Year", y="Average Ratings", data=df)
    ax3.set(ylim=(5, 7))
    ax3.set_title("Average Rating of Movies by Year")
    plt.savefig('rating_trend.png')

