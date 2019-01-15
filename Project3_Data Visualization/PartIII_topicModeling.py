#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 10:34:38 2018

@author: Zikai Zhu, Ju Huang, Xinyue Liu, Tianyi Yang
"""
import pandas as pd
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np
np.random.seed(2018)
import nltk
from pprint import pprint
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import warnings
warnings.filterwarnings("ignore")
#nltk.download('wordnet')
#nltk.download('stopwords')

############################ Preprocess the text
# Define a function of Stemming and lemmatization
def lemmatize_stemming(text):
    stemmer=PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Preprocess the text: select the stopwords and filter words that appear less than 3 times
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stopword and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def filter_out(processed_docs):
    # Create a dictionary from 'processed_docs' containing the number of 
    # times a word appears.
    dictionary = gensim.corpora.Dictionary(processed_docs)
    # Filter out tokens that appear in either less than 15 documents (absolute number) or
    # more than 0.5 documents (fraction of total corpus size, not absolute number)
    # After the above two steps, keep only the first 100000 most frequent tokens
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    # For each document, create a dictionary reporting how many words and how many times those words appear. 
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    return bow_corpus,dictionary


############################ Use LDA to do topic modeling      
def lda(bow_corpus,dictionary,num_topics,film,year):
    # Train our lda model using gensim.models.LdaMulticore and save it to ‘lda_model’
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, 
                                           id2word=dictionary, passes=2, workers=2)
    # For each topic, explore the words occuring in that topic and its relative weight
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
    
    # Create a table to store the Topic Modeling results
    score_table=pd.DataFrame(columns=['Name','Best Topic','0','1','2','3','4','5','6','7'])
    score_table['Name']=film
    
    # For each plot, store the distribution of topics
    for i in range(4746):
        for index, score in lda_model[bow_corpus[i]]:
            # print("\nScore: {}\t \nTopic: {}".format(score, index))
            score_table.loc[i][str(index)]=score
    
    # Change the data type into float        
    score_table[['0','1','2','3','4','5','6','7']]=score_table[['0','1','2','3','4','5','6','7']].astype('float64')
    
    # For each plot, determine its most possible topic
    btopic=[]
    for i in range(4746):
        temp=pd.DataFrame(score_table.loc[i][2:])
        temp[i]=temp[i].astype('float64')
        # Find out the topic that contains the largest score
        best_c = temp.loc[temp[i].idxmax()]
        best=pd.DataFrame(best_c).columns[0]
        btopic.append(best)
    score_table['Best Topic']=btopic
    
    # Put the column of years into score_table
    score_table['Year'] = year
        
    return score_table    


############################ Figure out the most prevalent topics
def word_count(score_table):
    word_counts = Counter(score_table['Best Topic'])
    # Get the number of times that each topic appears as the best topic
    frequency = word_counts.most_common(8)
    # Print the result
    print('\n\n','The frequency of topics:')
    print(frequency)

    # For each year, do the same thing
    for i in range(2010,2018):
        word_counts = Counter(score_table[score_table['Year']==i]['Best Topic'])
        # Get the number of times that each topic appears as the best topic
        frequency_byyear = word_counts.most_common(8)
        # Print the result
        print('\n','The frequency of topics in',i,':')
        print(frequency_byyear)
        
    return frequency


############################ Draw word clouds
def wordcloud(df,stopword):
    # Join the plots into one piece of text
    text = ' '.join(i for i in df['Plot'])
    print ('\n\nThere are {} words in the combination of all plots.'.format(len(text)))
    # Set max_font_size, change the maximum number of word and lighten the background
    # Set the stopwords
    wordcloud = WordCloud(max_font_size=100, max_words=200, width=800, height=400,
                          background_color="white", stopwords=stopword).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    # Save the graphs
    wordcloud.to_file('wordcloud.png')
    
    # For each year, do the same thing
    print('\n\nWord clouds for each year:')
    for j in range(2010,2018):
        text = ' '.join(i for i in df[df['Year_x']==j]['Plot'])
        print ('\n\nThere are {} words in the combination of all plots in the year'.format(len(text)),j,'.')
        # Set max_font_size, change the maximum number of word and lighten the background
        # Set the stopwords
        wordcloud = WordCloud(max_font_size=100, max_words=200, width=800, height=400,
                              background_color="white", stopwords=stopword).generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        # Save the graphs
        wordcloud.to_file('wordcloud'+str(j)+'.png') 
    
    
    
if __name__ == "__main__":
    __spec__ = None
    # Read in the data file
    df=pd.read_csv('final.csv')
    plot=df['Plot']
    film=df['Name']
    year=df['Year_x']
    # Prepare the stopwords
    stopwords_given = stopwords.words('english')
    stopwords_selected = ['one','would','could','said','may','might',
             'shall','first','also','must','should','instead','year','make',
             'years','take','takes','two','three','without','turn','turns','lives',
             'life','live','story','stories','film','find','finds','set','sets',
             'get','gets']
    stopword = stopwords_given + stopwords_selected
    
    # Preprocess the text
    processed_docs = plot.map(preprocess)
    # Create a dictionary to store the words and their times of appearance
    bow_corpus,dictionary=filter_out(processed_docs)
    # Use lda model to select a list of topics and create a table to store the result
    # (We choose the numbers of topics as 8.)
    score_table=lda(bow_corpus,dictionary,8,film,year)
    # Count the frequency of each topic
    frequency=word_count(score_table)
    # Draw the word clouds
    wordcloud(df,stopword)

