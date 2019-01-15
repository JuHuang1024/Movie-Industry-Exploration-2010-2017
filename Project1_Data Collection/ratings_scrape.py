#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 11:49:00 2018

After we get the data from boxoffice, we need to get rating data based on movie names in boxoffice.csv
This script is to scraping movie rating data from the omdbapi.com

@author: Ju Huang, Tianyi Yang, ZiKai Zhu, Xinyue Liu
"""

import requests
import pandas as pd
import re

baseurl = 'http://www.omdbapi.com/'
query_params = {'apikey':"ecfbadf2",
                't': 'Silent House',
                }
requests.get(baseurl,query_params).json()




# import the csv file
df_input = pd.read_csv("boxoffice.csv", encoding="cp1252")
df_input= df_input.rename(index=str,columns={'Unnamed: 0':'Ranking'})

# initialize final data list, base url and a fakerow
df_final = []
baseurl = 'http://www.omdbapi.com/'

# if fail to get a movie, add it into dataframe
fakerow = {'Title': 'N/A', 'Year': 'N/A', 'Rated': 'N/A', 'Released': 'N/A', 
           'Runtime': 'N/A', 'Genre': 'N/A', 'Director': 'N/A', 'Writer': 'N/A', 
           'Actors': 'N/A', 'Plot': 'N/A', 'Language': 'N/A', 'Country': 'N/A', 
           'Awards': 'N/A', 'Poster': 'N/A', 'Ratings': [], 'Metascore': 'N/A', 
           'imdbRating': 'N/A', 'imdbVotes': 'N/A', 'imdbID': 'N/A', 'Type': 'N/A', 
           'DVD': 'N/A', 'BoxOffice': 'N/A', 'Production': 'N/A', 'Website': 'N/A', 
           'Response': 'False'}

# initialize error counts to 0
notFoundCount = 0
yearMatchCount = 0
ratingCount = 0

# cleanTitle cleans the movie title
def cleanTitle(title):
    #delete the parethesis and the contents
    title = title.split('(')[0]
    #delete the "3-D"
    title = re.sub(r'3\-D','',title)
    #delete the "3D"
    title = re.sub(r'3D','',title)
    #delete the "Tyler Perry's "
    title = re.sub(r"Tyler Perry's\s",'',title)
    return title

# After first clean, if still 
def cleanTitle2(title):
    #delete the colon
    title = title.split(':')[0]
    # replace '&' with 'and'
    title = re.sub(r'&','and',title)
    # replace ',' with ''
    return title
    
def requestMovie(base, params):
    requests.adapters.DEFAULT_RETRIES = 511
    file = requests.get(base, params)
    content = file.json()
    return content

# This function clean the 'Rating' list in the 'c' dictionary and add it into the dataframe
def cleanRating(mid, c):
    if(c['Ratings']==[]):
        c['Ratings'] = [{'Source': 'Internet Movie Database', 'Value': 'N/A'}, 
                       {'Source': 'Rotten Tomatoes', 'Value': 'N/A'}, 
                       {'Source': 'Metacritic', 'Value': 'N/A'}]
    rating = pd.DataFrame.from_dict(c['Ratings'])
    df_rating = rating.set_index('Source').T
    df_rating.index = range(1)     
    movie = pd.DataFrame.from_dict(c)
    del movie['Ratings']
    movie = movie.drop(movie.index[1:])
    movie = pd.concat([movie, df_rating], axis=1)
    movie['movie_id'] = str(mid)
    df_final.append(movie)
    return(df_final)

# if 'Ratings' is null, add 'N/A' value into the dataframe
def cleanFake(mid, f):
    if(f['Ratings']==[]):
        data = {'Internet Movie Database':['N/A'],'Rotten Tomatoes':['N/A'],'Metacritic':['N/A']}
        df_rating=pd.DataFrame(data)
    else:
        rating = pd.DataFrame.from_dict(f['Ratings'])
        df_rating = rating.set_index('Source').T
        df_rating.index = range(1)
    movie = pd.DataFrame.from_dict(f, orient='index').T
    del movie['Ratings']
    movie = movie.drop(movie.index[1:])
    movie = pd.concat([movie, df_rating], axis=1)
    movie['movie_id'] = str(mid)
    df_final.append(movie)
    return(df_final)
    
# scraping the content from omdbapi
for index, row in df_input.iterrows():
    id = row['id']
    year = row['Year']
    title = cleanTitle(row['Name'])
    query_params = {'apikey':"ecfbadf2",
                    't': title,
                    'y': year,
                    # close the connection after every request
                    'Connection': 'close'
                    }
    
    content = requestMovie(baseurl, query_params)
    if('Error' in content):
        title = cleanTitle2(title)
        query_params = {'apikey':"ecfbadf2",
                        't': title,
                        # close the connection after every request
                        'Connection': 'close'
                        }
        content = requestMovie(baseurl, query_params)
    if('Year' in content):
        if(int(content['Year'][:4]) not in range(year-3, year+3)):
            content = {'Error':'Year not match!'}
            yearMatchCount += 1
            for yr in range(year-2, year+2):
                query_params = {'apikey':"ecfbadf2",
                                't': title,
                                'y': yr,
                                'Connection': 'close'
                                }
                if('Error' not in requestMovie(baseurl, query_params)):
                    content = requestMovie(baseurl, query_params)
                    yearMatchCount -=1
                    break
    if('Error' in content):
        #print(id, title, "Error: ", content['Error'])
        fakerow['Title'] = title
        fakerow['Year'] = year
        fakerow['Response'] = content['Error']
        df_final = cleanFake(id, fakerow)
        notFoundCount += 1
    else:
        df_final = cleanRating(id, content)
        
#print("number of movies:", len(df_final)) 
#print("number of errors:", notFoundCount)
#print("number of year errors:", yearMatchCount)
#print("number of source errors:", ratingCount)

df = pd.concat(df_final)
df.to_csv('ratings.csv', index = False)