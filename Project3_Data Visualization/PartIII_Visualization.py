#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 23:05:30 2018

@author: Ju Huang, Zikai Zhu, Tianyi Yang, Xinyue Liu
"""
import plotly
plotly.tools.set_credentials_file(username='juhuang', api_key='3ulNsW3VBPDO3KzEKYvn')

import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go
import math
from plotly import tools

############Preparation
def pre():
    # Read in the data set
    final = pd.read_csv('final.csv',encoding='latin-1',sep=',')
    
    # Prepare data for visualization of correlations with boxoffice
    final_cor = final.corr()
    # Select numeric variables that we need to see the correlations with boxoffice
    cols_to_cor = ['All Theaters', 'Opening', 'Opening Theaters', 'Timespan',
       'Metascore', 'Rotten Tomatoes', 'Runtime','imdbRating', 'imdbVotes', 'combined_rating']

    cor_box = final_cor[cols_to_cor][3:4].stack()
    # Rename column and index
    cor_box = pd.DataFrame(cor_box).set_index([cols_to_cor])
    cor_box = cor_box.rename(columns = {0: 'Total Gross'})

    return final, cols_to_cor, cor_box

#############Visualization1---Scatter Plot for correlation
def vis_corr(cols_to_cor, cor_box):
    # Create a trace for a scatter plot
    trace = go.Scatter(
            x = cols_to_cor,
            y = cor_box['Total Gross'],
            mode = 'markers')
    
    # Assign it to an iterable object named myData
    myData = [trace]
    
    # Add axes and title
    myLayout = go.Layout(
    	title = "Correlation between all numeric variables and boxoffice",
    	xaxis=dict(
    		title = 'Factors'
    	),
    	yaxis=dict(
    		title = 'Boxoffice',
            range=[-1,1]
    	)
    )
        
    # Setup figure
    myFigure = go.Figure(data=myData, layout=myLayout)
    # Create the scatterplot
    py.iplot(myFigure, filename='Correlation with boxoffice')



#############Visualization2---Boxplot for movie ratings
def vis_rating(df):
    # Create a trace for a boxplot
    trace0 = go.Box(y=df['M_z'], name = 'Metascore')
    trace1 = go.Box(y=df['R_z'], name = 'Rotten Tomato')
    trace2 = go.Box(y=df['I_z'], name = 'Imdb Rating')


    # Assign it to an iterable object named myData
    myData = [trace0, trace1,trace2]
    
    # Add axes and title
    myLayout = go.Layout(
    	title = "Distributions of movie ratings from three resources",
    	xaxis=dict(
    		title = 'Resourses'
    	),
    	yaxis=dict(
    		title = 'Movie Rating(z-score)'
    	)
    )
    
    # Setup figure
    myFigure = go.Figure(data=myData, layout=myLayout)
    # Create the scatterplot
    py.iplot(myFigure, filename='Movie Rating Distribution')


##############Visualization3---Bubble graph for rating vs boxoffice with year and studio size
# calculate the number of movies each studio produces, which represents the studio size
def numeric_studio(df):
    value_ct=pd.value_counts(df.Studio)
    studio_size=[]
    for index, row in df.iterrows():
        name=row['Studio']
        studio_size.append(value_ct.loc[name])
        
    df['Studio_size'] = studio_size

def vis_bubble(df):
	# add a column of studio size
    numeric_studio(df)
    slope = 2.666051223553066e-05
    hover_text = []
    bubble_size = []
    
	# text infomation shown for each bubble and apply bubble size
    for index, row in df.iterrows():
        hover_text.append(('Name: {Name}<br>'+
                           'Year: {Year_x}<br>'+
                           'Total Gross: {TotalGross}<br>'+
                           'Rating: {combined_rating}<br>'+
                           'All Theaters: {AllTheaters}').format(Name=row['Name'],
                                                Year_x=row['Year_x'],
                                                TotalGross=row['Total Gross'],
                                                combined_rating=row['combined_rating'],
                                                AllTheaters=row['All Theaters']))
        bubble_size.append(math.sqrt(row['Studio_size']*slope))
    
    df['text'] = hover_text
    df['size'] = bubble_size
    sizeref = 2.*max(df['size'])/(50.**2)
    
	# trace for 2010
    trace0 = go.Scatter(
        x=df['Total Gross'][df['Year_x'] == 2010],
        y=df['combined_rating'][df['Year_x'] == 2010],
        mode='markers',
        name='2010',
        text=df['text'][df['Year_x'] == 2010],
        marker=dict(
            symbol='circle',
            sizemode='area',
            sizeref=sizeref,
            size=df['size'][df['Year_x'] == 2010],
            line=dict(
                width=2
            ),
        )
    )
    
	# trace for 2011	
    trace1 = go.Scatter(
        x=df['Total Gross'][df['Year_x'] == 2011],
        y=df['combined_rating'][df['Year_x'] == 2011],
        mode='markers',
        name='2011',
        text=df['text'][df['Year_x'] == 2011],
        marker=dict(
            symbol='circle',
            sizemode='area',
            sizeref=sizeref,
            size=df['size'][df['Year_x'] == 2011],
            line=dict(
                width=2
            ),
        )
    )
    
	# trace for 2012
    trace2 = go.Scatter(
        x=df['Total Gross'][df['Year_x'] == 2012],
        y=df['combined_rating'][df['Year_x'] == 2012],
        mode='markers',
        name='2012',
        text=df['text'][df['Year_x'] == 2012],
        marker=dict(
            symbol='circle',
            sizemode='area',
            sizeref=sizeref,
            size=df['size'][df['Year_x'] == 2012],
            line=dict(
                width=2
            ),
        )
    )        
    
	# trace for 2013
    trace3 = go.Scatter(
        x=df['Total Gross'][df['Year_x'] == 2013],
        y=df['combined_rating'][df['Year_x'] == 2013],
        mode='markers',
        name='2013',
        text=df['text'][df['Year_x'] == 2013],
        marker=dict(
            symbol='circle',
            sizemode='area',
            sizeref=sizeref,
            size=df['size'][df['Year_x'] == 2013],
            line=dict(
                width=2
            ),
        )
    )        

	# trace for 2014
    trace4 = go.Scatter(
        x=df['Total Gross'][df['Year_x'] == 2014],
        y=df['combined_rating'][df['Year_x'] == 2014],
        mode='markers',
        name='2014',
        text=df['text'][df['Year_x'] == 2014],
        marker=dict(
            symbol='circle',
            sizemode='area',
            sizeref=sizeref,
            size=df['size'][df['Year_x'] == 2014],
            line=dict(
                width=2
            ),
        )
    )
    
	# trace for 2015
    trace5 = go.Scatter(
        x=df['Total Gross'][df['Year_x'] == 2015],
        y=df['combined_rating'][df['Year_x'] == 2015],
        mode='markers',
        name='2015',
        text=df['text'][df['Year_x'] == 2015],
        marker=dict(
            symbol='circle',
            sizemode='area',
            sizeref=sizeref,
            size=df['size'][df['Year_x'] == 2015],
            line=dict(
                width=2
            ),
        )
    )
    
	# trace for 2016
    trace6 = go.Scatter(
        x=df['Total Gross'][df['Year_x'] == 2016],
        y=df['combined_rating'][df['Year_x'] == 2016],
        mode='markers',
        name='2016',
        text=df['text'][df['Year_x'] == 2016],
        marker=dict(
            symbol='circle',
            sizemode='area',
            sizeref=sizeref,
            size=df['size'][df['Year_x'] == 2016],
            line=dict(
                width=2
            ),
        )
    )
    
	# trace for 2017
    trace7 = go.Scatter(
        x=df['Total Gross'][df['Year_x'] == 2017],
        y=df['combined_rating'][df['Year_x'] == 2017],
        mode='markers',
        name='2017',
        text=df['text'][df['Year_x'] == 2017],
        marker=dict(
            symbol='circle',
            sizemode='area',
            sizeref=sizeref,
            size=df['size'][df['Year_x'] == 2017],
            line=dict(
                width=2
            ),
        )
    )
            
    data = [trace0, trace1, trace2, trace3, trace4,trace5,trace6,trace7]
    layout = go.Layout(
        title='Rating v. Total Box Office, 2010-2017',
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='Rating v. Total Box Office')
    
    
#############Visualization4---Histograms for topic frequency in total
def vis_hist1():
    # Create a data set to store the frequency of topics
    topics = ["An Adventure Game","Families and Friends","Women's Struggling Lives",
              "Young People's Family Lives and Fights", "A Documentary of the World of American",
              "A Journey of Love", "Dreams of American Young People", "Crime"]
    frequency = [554, 674, 611, 634, 604, 663, 477, 529]
    
    # Create a trace for a histogram
    trace = go.Bar(
            x = topics,
            y = frequency,
            marker=dict(
                    color=['rgb(255,192,203)', 'rgb(255,248,220)',
                           'rgb(173,216,230)', 'rgb(123,104,238)',
                           'rgb(255,165,0)', 'rgb(75,0,130)', 
                           'rgb(255,228,181)', 'rgb(205,92,92)'])
           )
    
    # Assign it to an iterable object named myData
    myData = [trace]
    
    # Add axes and title
    myLayout = go.Layout(
    	title = "Topics Frequency",
    	xaxis=dict(
    		title = 'Topics'
    	),
    	yaxis=dict(
    		title = 'Frequency',
            range = [0,700]
    	),
    )
        
    # Setup figure
    myFigure = go.Figure(data=myData, layout=myLayout)
    myFigure['layout'].update(height=600, width=600)
    # Create the scatterplot
    py.iplot(myFigure, filename='Topic Frequency in total')    
    
vis_hist1()

#############Visualization5---Histograms for topic frequency by year
def vis_hist2():
    # Create a data set to store the frequency of topics
    topics = ["Adventure Game","Families and Friends","Women's Struggling Lives",
              "Young People's Family Lives and Fights", "Documentary of the World of American",
              "Journey of Love", "Dreams of American Young People", "Crime"]
    frequency2010 = [55, 61, 67, 69, 61, 68, 52, 52]
    frequency2011 = [62, 82, 68, 68, 73, 68, 52, 70]
    frequency2012 = [69, 73, 96, 85, 68, 84, 55, 63]
    frequency2013 = [69, 100, 64, 86, 76, 94, 83, 52]
    frequency2014 = [68, 76, 73, 93, 75, 101, 58, 78]
    frequency2015 = [73, 107, 85, 70, 77, 84, 55, 55]
    frequency2016 = [75, 99, 71, 84, 87, 86, 56, 80]
    frequency2017 = [83, 76, 87, 79, 87, 78, 66, 79]

   
    # Create traces for histograms
    trace2010 = go.Bar(
            x = topics,
            y = frequency2010,
            name = 2010
           )
    
    trace2011 = go.Bar(
            x = topics,
            y = frequency2011,
            name = 2011
           )

    trace2012 = go.Bar(
            x = topics,
            y = frequency2012,
            name = 2012
           )    
    
    trace2013 = go.Bar(
            x = topics,
            y = frequency2013,
            name = 2013
           )    
    
    trace2014 = go.Bar(
            x = topics,
            y = frequency2014,
            name = 2014
           )    
    
    trace2015 = go.Bar(
            x = topics,
            y = frequency2015,
            name = 2015
           )  
    
    trace2016 = go.Bar(
            x = topics,
            y = frequency2016,
            name = 2016
           )      

    trace2017 = go.Bar(
            x = topics,
            y = frequency2017,
            name = 2017
           )      
 
    # Add sub title
    myFigure = tools.make_subplots(rows=4, cols=2, subplot_titles=('2010', '2011', '2012', '2013',
                                                              '2014', '2015', '2016', '2017'))
    
    myFigure.append_trace(trace2010, 1, 1)
    myFigure.append_trace(trace2011, 1, 2)
    myFigure.append_trace(trace2012, 2, 1)
    myFigure.append_trace(trace2013, 2, 2)
    myFigure.append_trace(trace2014, 3, 1)
    myFigure.append_trace(trace2015, 3, 2)
    myFigure.append_trace(trace2016, 4, 1)
    myFigure.append_trace(trace2017, 4, 2)    
    
    myFigure['layout'].update(height=1400, width=600, title='Topic Frequency by Year')

    # Create the scatterplot
    py.iplot(myFigure, filename='Topic Frequency by Year')
vis_hist2()    
     
    
if __name__ == "__main__":
    final, cols_to_cor, cor_box = pre()
    vis_corr(cols_to_cor, cor_box)
    vis_rating(final)
    vis_bubble(final)
    vis_hist1()
    vis_hist2()
