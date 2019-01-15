#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 19:05:30 2018

@author: Tianyi Yang, Xinyue Liu, Zikai Zhu, Ju Huang
"""
import pandas as pd
import matplotlib.pyplot as plt 
from tabulate import tabulate
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
################################################
# IGNORE the Warnings!
import warnings
warnings.filterwarnings("ignore")
################################################

################################################
# Histogram Function
################################################
def histogram(final):
    final['Total Gross Bin'].hist()
    plt.title('Total Gross')
    plt.xlabel('Total Gross Bin')
    plt.savefig('totalgross.png')
    plt.clf()
    final['combined_rating Bin'].hist()
    plt.title('Combined Rating')
    plt.xlabel('combined rating bin')
    plt.savefig('combinedrating.png')
    plt.clf()
    final['Opening Bin'].hist()
    plt.title('Opening')
    plt.xlabel('opening bin')
    plt.savefig('opening.png')
    plt.clf()

################################################
# Correlation Function
################################################
def correlation(final):
    final_corr = final.corr()
    print('\n\nCorrelation Table:\n')
    print (tabulate(final_corr, headers='keys',tablefmt="psql"))
    
    tg=final['Total Gross']
    op=final['Opening']
    cr=final['combined_rating']
    
    plt.scatter(tg,cr)
    plt.title('scatter for total gross and rating')
    plt.xlabel('total gross')
    plt.ylabel('rating')
    plt.savefig('totalgross_rating.png')
    plt.clf()
    plt.scatter(tg,op)
    plt.title('scatter for total gross and opening')
    plt.xlabel('total gross')
    plt.ylabel('opening')
    plt.savefig('totalgross_opening.png')
    plt.clf()
    plt.scatter(op,cr)
    plt.title('scatter for opening and rating')
    plt.xlabel('opening')
    plt.ylabel('rating')
    plt.savefig('opening_rating.png')
    plt.clf()

################################################
# Clustering Function
################################################    
def preprocess(final):
    myData=pd.concat([final['Total Gross'],final['Timespan'],final['All Theaters'],final['combined_rating']],
                     axis=1,keys=['Total Gross','Timespan','All Theaters','combined_rating'])
    x = myData.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    return normalizedDataFrame
    
def k_clustering(final,k):
    # preprocess data for clustering
    normalizedDataFrame=preprocess(final)
    
    # K-Mean clustering
    print('\n K clustering:\n')
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    # silhouette and calinski score for k-clustering
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    calinski_score=calinski_harabaz_score(normalizedDataFrame,cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    print("For n_clusters =", k, "The average calinski_score is :", calinski_score)
    # convert our high dimensional data to 2 dimensions
    pca2D = decomposition.PCA(2)
    # Turn the NY Times data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.savefig('k_cluster.png')
    # Clear plot
    plt.clf()
    return cluster_labels
    
    
def hie_clustering(final,k):
    # preprocess data for clustering
    normalizedDataFrame=preprocess(final)
    # Hierarchical clustering
    print('\n Hierarchical clustering:\n')
    hie=AgglomerativeClustering(n_clusters=k)
    h_labels=hie.fit_predict(normalizedDataFrame)
    silhouette_avg = silhouette_score(normalizedDataFrame, h_labels)
    calinski_score=calinski_harabaz_score(normalizedDataFrame,h_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
    print("For n_clusters =", k, "The average calinski_score is :", calinski_score)
    # convert our high dimensional data to 2 dimensions
    pca2D = decomposition.PCA(2)
    # Turn the NY Times data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=h_labels)
    plt.savefig('hie_cluster.png')
    # Clear plot
    plt.clf()
    return h_labels
    
    
def dbscan(final):
    # preprocess data for clustering
    normalizedDataFrame=preprocess(final)
    # DBSCAN
    print('\n DBSCAN clustering:\n')
    dbscan=DBSCAN(eps=0.07, min_samples=10)
    dbscan_labels=dbscan.fit_predict(normalizedDataFrame)
    silhouette_avg = silhouette_score(normalizedDataFrame, dbscan_labels)
    calinski_score=calinski_harabaz_score(normalizedDataFrame,dbscan_labels)
    print("For DBSCAN The average silhouette_score is :", silhouette_avg)
    print("For DBSCAN The average calinski_score is :", calinski_score)
    
    # convert our high dimensional data to 2 dimensions
    pca2D = decomposition.PCA(2)
    # Turn the NY Times data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=dbscan_labels)
    plt.savefig('dbscan_cluster.png')
    # Clear plot
    plt.clf()
    return dbscan_labels

def analyze_cluster(final,labels,k):
    final['cluster labels']=labels
    for i in range(k):
        cluster=final.loc[final['cluster labels'] == i]
        mean=cluster['combined_rating'].mean()
        plt.boxplot(cluster['combined_rating'])
        plt.title('cluster'+str(i))
        plt.savefig('c'+str(i)+'.png')
        plt.clf()
        print('for cluster ',i,', the combined_rating mean is ',mean)
    
#########################################
# Association Rules Function
#########################################
def genre_df(f):
    # Reorganize genre data into sparse dataset
    genredf = f[['id', 'Genre']]
    genredf2 = genredf.apply(lambda x: pd.Series((x['Genre'].split(","))),axis=1)
    genredf2['id']=genredf['id']
    genredf2 = genredf2.set_index('id').stack().reset_index()
    genredf2['level_1'] = genredf2['level_1'] + 1
    genredf2.columns = ['id', 'priority', 'genre']
    genredf2['genre'] = genredf2['genre'].str.strip()
    basket = (genredf2.groupby(['id', 'genre'])['priority'].sum().unstack().reset_index().fillna(0).set_index('id'))
    del basket["-1"]
    return genredf2, basket
  
def encode_units(x):
    # For Cleaned data, if movie has that genre, return 1; otherwise return 0
    if(x <= 0):
        return 0
    if(x >= 1):
        return 1
   
if __name__ == "__main__":
    # Read Data
    final = pd.read_csv('final.csv', encoding='utf-8',sep=',')
    # Histogram of Data
    histogram(final)
    # Correlation of Data
    correlation(final)
    # Clustering of Data
    # k-means
    k_label=k_clustering(final,3)
    analyze_cluster(final,k_label,3)
    # hierarchical
    h_label=hie_clustering(final,3)
    analyze_cluster(final,h_label,3)
    # dbscan
    d_label=dbscan(final)
    d_label=d_label+1
    analyze_cluster(final,d_label,4)
    # Create data subset for Apriori analysis
    gdf, bucket = genre_df(final)
    genre_sets = bucket.applymap(encode_units)
    # Get frequent itemsets with min_support = 0.03, 0.05, 0.07, 0.09)
    # Generate frequent rules
    for sup in (0.03, 0.05, 0.07, 0.09):
        freq_itemsets = apriori(genre_sets, min_support=sup, use_colnames=True)
        fi_filename = "freq_itemsets" + str(sup) + ".csv"
        freq_itemsets.to_csv(fi_filename)
        rules = association_rules(freq_itemsets, metric="lift", min_threshold=1)
        r_filename = "association_rules" + str(sup) + ".csv"
        rules.to_csv(r_filename)
        print("Support Level:", sup)
        print("Frequent Itemsets:", freq_itemsets)
        print("Frequent Rules", rules)