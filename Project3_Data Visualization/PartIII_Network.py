#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:51:10 2018

@author: Tianyi Yang, Xinyue Liu, Zikai Zhu, Ju Huang
"""

import pandas as pd
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate
import community
import math
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.lines as mlines
import re

# IGNORE the Warnings!
import warnings
warnings.filterwarnings("ignore")




def clean_df(dataframe):
    # Clean the 'Production' column since there are typos in the column 
    word_to_delete = [' pictures',' studios',' studio','!',' usa',' entertainment'
                   ,' film',' studios','walt ',' inc.',' co.',' productions'
                   ,' media',' releasing',' llc.',' distribution',' classics'
                   ,' home video',' animation',' motion',' home','. well go'
                   ,'the ',' screen gems',' international',' classic',' cinema'
                   ,' u.s.',' america',' limited',' worldwide americas',' visions'
                   ,'twentieth century ','20th century ','century ',' us',' bros.'
                   ,' bros',' searchlight',' features',' world',' group',' adventuras'
                   ,' ventures',' brothers',' laboratories',' films',' studio'
                   ,' television',' india',' premiere',' deluxe',' pictutures'
                   ,' picutres',' twc',' fi lms',' big',' a e indies',' midnight']
    word_to_replace = ['rlje',' attractions','screen gems','tribecca'
                       ,'weinsteinpany','weinsteinpanypany','weinsteinpanypanypany'
                       ,'weinsteinpanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypanypany'
                       ,'utvmunications','varience','new line','new linewarner'
                       ,' and ','radiustwc','millenium','milennium','maya sigel'
                       ,'liongate','lionsgatecodeblack','lionsgatepantelion'
                       ,'lionsgatesummit','lionsgateroadside attraction','summit'
                       ,'warnerwarner','warnerlegendary','warnerimax','weinsteindimension'
                       ,'sonysony','paramountdwa','lionsgates','dwa','variances'
                       ,'a24s','ifcs','bond/360','edreamworksrd r pressmanporation'
                       ,'produktion','miramaxs','sonys','tribecas']
    replaced_word = ['rlj',' attraction','sony','tribeca','weinstein','weinstein'
                     ,'weinstein','weinstein','utv','variance','warner','warner'
                     ,'/','radius','millennium','millennium','maya','lionsgate'
                     ,'lionsgate','lionsgate','lionsgate','lionsgate/roadside attraction'
                     ,'lionsgate','warner','warner','warner','weinstein','sony'
                     ,'paramount/dwa','lionsgate','dreamworks','variance','a24'
                     ,'ifc','bond360','dreamworks','production','miramax','sony'
                     ,'tribeca']
    for word in word_to_delete:
        dataframe['Production'] = dataframe['Production'].str.lower()
        dataframe['Production'] = dataframe['Production'].apply(lambda x:re.sub('[^A-Za-z0-9/]+', ' ', x))
        dataframe['Production'] = dataframe['Production'].str.replace(word,'')
    for i in range(len(word_to_replace)):
        dataframe['Production'] = dataframe['Production'].str.replace(word_to_replace[i],replaced_word[i])
    return (dataframe)


def get_nodes(dataframe):
    # Get the nodes of the network for each year 
    studios_list= dataframe['Production']
    studios_edges=list()
    for i in studios_list:
        studios= i.split('/')
        # Data cleaning: delete the space if there's any 
        studios = list(map(lambda x: x[1:] if x[0]==' ' else x,studios))
        studios = list(map(lambda x: x[:-1] if x[-1]==' ' else x,studios))
        # Generate the nodes
        comb= combinations(studios,1)
        for c in list(comb):
            studios_edges.append(c)    
        getweight=pd.Series((v for v in studios_edges))
        nodes= getweight.value_counts()
    return (nodes)

def get_studios_edges(dataframe):
    # Get the edges of network for each year    
    studios_list= dataframe['Production']
    studios_edges=list()
    for i in studios_list:
        studios= i.split('/')
        # Data cleaning: delete the space if there's any 
        studios = list(map(lambda x: x[1:] if x[0]==' ' else x,studios))
        studios = list(map(lambda x: x[:-1] if x[-1]==' ' else x,studios))
        # Generate the edge list 
        comb= combinations(studios,2)
        for c in list(comb):
            studios_edges.append(c)
        # Get weight: weight = # of cooperations between two studios
        getweight=pd.Series((v for v in studios_edges))
        edges_weight= getweight.value_counts()
    return (edges_weight)

def get_nw(edge_list,node_list,year,result):
    # For each edge list (for each year), generate the network      
    G=nx.Graph() 
    # Add nodes
    for i in range(len(node_list)):
        node= node_list.index[i][0]
        G.add_node(node)
    # Add edges
    for i in range(len(edge_list)):
        node1= edge_list.index[i][0]
        node2= edge_list.index[i][1] 
        weight=edge_list.values[i]
        G.add_edge(node1,node2,weight=weight) 
    # Record global metrics of the network into a dataframe     
    nbr_nodes = nx.number_of_nodes(G)
    nbr_edges = nx.number_of_edges(G)
    density = nx.density(G)
    triangle_list= nx.triangles(G)
    triangles= max(triangle_list.values())
    result.loc[year][0]=nbr_nodes
    result.loc[year][1]=nbr_edges
    result.loc[year][2]=round(density*1000,3)
    result.loc[year][3]=triangles
    # Record local metrics of the network 
    # get betweenness centralities 
    betweenness_centralities = nx.betweenness_centrality(G)
    metrics_node_in= range(len(betweenness_centralities))
    metrics_node = pd.DataFrame(index=metrics_node_in,  
                                columns=['Film','Btwns Cntr','Degree','Clstr Coef.'])
    for i in range(len(betweenness_centralities)):
        film= list(betweenness_centralities.keys())[i]
        bc = list(betweenness_centralities.values())[i]
        metrics_node.loc[i]['Film']= film
        metrics_node.loc[i]['Btwns Cntr']=round(bc,3)
    # get clustering coefficient  
    partition = community.best_partition(G)
    p = [partition.get(node) for node in G.nodes()]    
    clusters= max(p)+1    
    result.loc[year][4]=clusters
    for i in range(len(p)): 
        metrics_node.iloc[i]['Clstr Coef.'] = p[i]   
    # get degree for each node
    metrics_node_rein = metrics_node.set_index('Film')
    for i in G:
        d = G.degree(i)
        metrics_node_rein.loc[i]['Degree'] = d
    # Record the local metrics for the network into a txt file
    file_name = 'Network metrics for year 2010 - 2017.txt'
    with open (file_name,"a") as f:
        f.write('\n\n\nNetwork Metrics for year '+str(year)+'\n\n')
        f.write(tabulate(metrics_node_rein,headers='keys'))
        f.close()

def get_nw_trend(df):
    # For each year, get the stats about network of studio collaboration 
    nw_year= pd.DataFrame(index=[2010,2011,2012,2013,2014,2015,2016,2017],
                              columns=['# Nodes','# Edges','Density in 10^(-3)','# Triangles','# Clusters'])
    df_c = clean_df(df) 
    for y in [2010,2011,2012,2013,2014,2015,2016,2017]:
        df_year = df_c[df_c['Year_x']==y]
        edge_yr = get_studios_edges(df_year)
        node_yr= get_nodes(df_year)
        get_nw(edge_yr,node_yr,y,nw_year)
    print('\n\nPlease see the txt output for local network metrics for each year')
    print ('\n\nNetwork of studios 2010-2017\n')
    print (tabulate(nw_year, headers='keys',tablefmt="github"))

def get_node_color(G):
    # For each studio, calculate the total gross 
    nodes= [node for node in G]
    studio_boxoffice=dict()
    default_value = [0]*len(nodes)
    studio_boxoffice = dict(zip(nodes,default_value))
    df_c = clean_df(df)[['Production','Total Gross']]
    for node in nodes:
        for index, row in df_c.iterrows():
            prd = row['Production']
            grs = row['Total Gross']
            if str(node) in str(prd):
                studio_boxoffice[str(node)] += float(grs)
    # Give label to studios by level of box office
    ten_5=list()
    ten_6=list()
    ten_7=list()
    ten_8=list()
    ten_9=list()
    ten_10=list()
    ten_11=list()
    for k,v in studio_boxoffice.items():
        if v< 100000:
            ten_5.append(k)
        elif 100000<=v<1000000:
            ten_6.append(k)
        elif 1000000<=v<10000000:
            ten_7.append(k)
        elif 10000000<=v<100000000:
            ten_8.append(k)
        elif 100000000<=v<1000000000:
            ten_9.append(k)
        elif 1000000000<=v<10000000000:
            ten_10.append(k)
        else:
            ten_11.append(k)
    # Color the studios by the labels created above
    color_map=[]
    for node in G:
        if node in ten_5:
            color_map.append('black')
        elif node in ten_6:
            color_map.append('dimgray')
        elif node in ten_7:
            color_map.append('lightgrey')
        elif node in ten_8:
            color_map.append('pink')
        elif node in ten_9:
            color_map.append('salmon')
        elif node in ten_10:
            color_map.append('indianred')
        else:
            color_map.append('darkred')
    return color_map

def get_edge_color(G):
    # Code the edge color by its weight (# of collaborations)
    edge_color=[]
    for edge in G.edges(data=True):
        if edge[2]['weight']>1:
            edge_color.append('blue')
        else:
            edge_color.append('yellow')
    return edge_color

def get_nw_graph(df):
    # Draw the network graph for studio collaboration since 2010 
    df_c=clean_df(df)
    edge_7yr = get_studios_edges(df_c)
    G = nx.Graph()
    for i in range(len(edge_7yr)):
        node1= edge_7yr.index[i][0]
        node2= edge_7yr.index[i][1] 
        weight=edge_7yr.values[i]
        G.add_edge(node1,node2,weight=weight) 
    color_map = get_node_color(G)
    edge_color = get_edge_color(G)
    # Draw the graph
    nx.draw(G,pos=nx.spring_layout(G,k=0.23),node_color=color_map, node_size=30,edge_color=edge_color)
    # Create the lagends  & title 
    darkred = mpatches.Patch(color='darkred',label='10^11')
    indianred = mpatches.Patch(color='indianred',label='10^10')
    salmon = mpatches.Patch(color='salmon',label='10^9')
    pink = mpatches.Patch(color='pink',label='10^8')
    lightgrey = mpatches.Patch(color='lightgrey',label='10^7')
    dimgray = mpatches.Patch(color='dimgray',label='10^6')
    black = mpatches.Patch(color='black',label='10^5')
    legend_down=plt.legend(handles=[lightgrey,dimgray,black],fontsize=10,loc=3)
    legend_up= plt.legend(handles=[darkred,indianred,salmon,pink]
        ,title="Box Office($)",fontsize=10,loc=2)
    blue_line = mlines.Line2D([], [], color='blue', marker='_',
                          markersize=1, label='>1')
    orange_line = mlines.Line2D([], [], color='orange', marker='_',
                          markersize=1, label='=1')
    plt.legend(handles=[blue_line,orange_line],title='# Collaboration'
               ,fontsize=8,loc=1)
    plt.gca().add_artist(legend_down)
    plt.gca().add_artist(legend_up)
    plt.title('Network of studio collaboration 2010-2017',fontsize=18)
    plt.rcParams["figure.figsize"] = [8,8]
    plt.show()

if __name__ == "__main__":
    df= pd.read_csv('final.csv')
    get_nw_trend(df)
    get_nw_graph(df)
