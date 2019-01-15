Project Website (Story Document Link):
http://ivoryue.georgetown.domains/501/home.html

_________________________________________________________________________________________________________________


Overall Trend:



File: PartIII_trend_graphs.py

Input: final.csv



Program: 
This script cleans final.csv and plot 3 graphs: trend of number of movies by year, trend of box office by year and trend of average ratings by year



Output:

file (graph) output:

1. boxoffice_trend.png 

This graph shows the overall trend of box office by year from 2010 to 2017
2. 
n_movies_trend.png

This graph shows the overall trend of number of movies by year from 2010 to 2017

3. rating_trend.png

This graph shows the overall trend of average ratings of movies by year from 2010 to 2017








________________________________________________________________________________________________________________


Genre Analysis: 


File: PartIII_genre_analysis.py



Input: final.csv, boxoffice_cleaned.csv



Program: (needs Seaborn package version 0.9.0) This script transforms the dataset into a sparse dataset. Each row represents one of top n genres of that year (2010-2017). Columns including number of movies of each genre for that year and box office of movies of each genre for that year. Using the dataset genreated above, plot the bar graphs related to genre trend.



Output:

file (graph) output: 

1. genre_count.png 

This graph shows proportion of number of movies for each genre of each year from 2010-2017
 
2. genre_boxoffice.png

This graph shows proportion of boxoffice for each genre of each year from 2010-2017


_________________________________________________________________________________________________________________


Topic Modeling:

File: PartIII_topicModeling.py

Input: final.csv

Program: This script uses LDA model to extract 8 topics from all movie plots and shows a list of words for each topic. It also gives accuracy rate of each topic for each plot and shows the best topic for each plot.
This script then calculate the frequency of each topic in total and by year.
The script also creates a wordcloud for all plots and 8 wordclouds for each year.

Output:
1. Printed output: The words of each topic
2. Printed output: The frequency of each topic
3. File output:
wordcloud.png
wordcloud2010.png
wordcloud2011.png
wordcloud2012.png
wordcloud2013.png
wordcloud2014.png
wordcloud2015.png
wordcloud2016.png
wordcloud2017.png

________________________________________________________________________________________________________________


Network Analysis：

File: PartIII_Network.py

Input: final.csv



Program: 
This script will clean the data frame, analyze the 'production' column, which shows one or more studios that produce the film, and generate a network of collaboration between studios. 

Output: 

1. printed output: table "Network of studios 2010-2017'

This output summarize each year's global network metrics of the studio collaboration network from 2010 to 2017. 


2. file output: Network metrics for year 2010 - 2017.txt

This file contains the local network metrics for each year's network. 


3. printed output: network graph 2010-2017 

This output is a visualization of the network 2010-2017. The nodes (studios) are colored by level of box-office, and the edges are colored by the weight (how many collaboration). 





_________________________________________________________________________________________________________________


Linear Regression:

File: PartIII_Linear_Reg_Rated.py


Input: final.csv



Program: (needs Seaborn package) 
This script tries to generate visualization to help explain the linear regression result from Part II. From the linear regression model, we found that the categorical variable, 'Rated', has an impact on box office. This script uses the seaborn package to create a box plot for each category of 'Rated' to visualize how different rating lead to different level of box office. 



Printed Output: "Total Gross by Rated"
Described above. 



__________________________________________________________________________________________________________________


Visualization:

File: PartIII_Visualization.py

Input: final.csv

Program: This script uses plotly to do visualization.

Output:
1. Scatter plot of the correlation between all other numerical attributes and box office.
https://plot.ly/~juhuang/14

2. Boxplot of the movie ratings of three different websites.
https://plot.ly/~juhuang/12

3. Bubble graph for rating vs boxoffice with year and studio size
https://plot.ly/~juhuang/16

4. Histograms for topic frequency in total
https://plot.ly/~juhuang/18

5. Histograms for topic frequency by year
https://plot.ly/~juhuang/20


