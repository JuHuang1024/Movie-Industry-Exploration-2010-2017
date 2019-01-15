#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:51:46 2018

This script is to scrape movie's boxoffice data from www.boxofficemojo.com from 2010 to 2018

@author: Xinyue Liu, Zikai Zhu, Ju Huang, Tianyi Yang
"""

import requests
import lxml.html as lh
import pandas as pd

# this function return the movie table elements
def movie_table_request(base_url, url_para):
    response = requests.get(base_url, url_para)
    doc = lh.fromstring(response.content)
    tr_elements = doc.xpath('//tr')
    return tr_elements

base_url='https://www.boxofficemojo.com/yearly/chart/?'
df_list = []

for yr in range(2010, 2018):
    # filename = "boxoffice" + str(yr) + ".csv"
    rows = []

    for p in range(0, 8):
        url_para = {'page': p+1,
                    'yr': yr}

        tr_elements = movie_table_request(base_url, url_para)

        for r in range(9, len(tr_elements)-8):
            rows.append(tr_elements[r])
            # print(len(rows))
    
    col = [('Year', [yr]*len(rows)),
           ('Rank', []),
           ('Name', []),
           ('Studio', []),
           ('Total Gross', []),
           ('All Theaters', []),
           ('Opening', []),
           ('Opening Theaters', []),
           ('Open', []),
           ('Close', [])]
    
    for r in range(0, len(rows)):
        i = 1
        for c in rows[r].iterchildren():
            data = c.text_content()
            col[i][1].append(data)
            i += 1

    d = {title: column for (title, column) in col}
    df_list.append(pd.DataFrame(d))

df = pd.concat(df_list)
df.insert(loc=0, column='id', value=range(1, len(df) + 1))

df.to_csv("boxoffice.csv", index=False)
    

    