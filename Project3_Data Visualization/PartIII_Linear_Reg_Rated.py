#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 21:02:29 2018

@author: Tianyi Yang, Xinyue Liu, Zikai Zhu, Ju Huang
"""
       
import seaborn as sns
import math 
import pandas as pd



def further_grouping(final):
    # Further groupings for categorical varaibles 'Rated'
    final_g= final.copy()
    ### Rates: transform the rated col further to create dummy variables   
    rates=final_g['Rated'].value_counts()
    other= list(rates[rates<50].index)
    final_g['Rated'][final_g['Rated'].isin(other)]= 'Other'
    return (final_g)    

def gross_by_rated(final):
	# Plot boxplot of Total Gross by rated 
    final_plot=further_grouping(final)
    final_plot['$ Total Gross(Log10)']= [math.log(x,10) for x in final_plot['Total Gross']]
	# Set the aesthetics of the graph
    sns.set(style='dark',font_scale=1.5)
    sns.boxplot(x="Rated", y="$ Total Gross(Log10)", data=final_plot,palette="Set3",saturation=0.6).set_title('Total Gross($) by Rated')
    sns.despine(top=False, right=False,left=False,bottom = False)
    
if __name__ == "__main__":  
    df = pd.read_csv('final.csv', encoding='utf-8',sep=',')
    gross_by_rated(df)   
