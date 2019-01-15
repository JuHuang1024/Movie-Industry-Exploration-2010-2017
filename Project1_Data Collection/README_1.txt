PLEASE RUN PROGRAMS IN THE FOLLOWING ORDER:1. boxoffice_scrape.py2. ratings_scrape.py (It might get an error because of the API Key daily requests limitation.’ratings.csv’is the result we get from this program)3. boxoffice_clean.py4. boxoffice_score.py5. ratings_clean.py6. ratings_score.pyDATA:1. boxoffice.csv	This is the data we get after running boxoffice_scrape.py
2. boxoffice_cleaned.csv	This is the cleaned data we get after running boxoffice_clean.py
3. ratings.csv	This is the data we get after running ratings_scrape.py
4. ratings_cleaned.csv	This is the cleaned data we get after running ratings_clean.py
PROGRAMS:1. boxoffice_scrape.py	This program gets the movie boxoffice data from www.boxofficemojo.com from 2010 to  2017.
2. boxoffice_score.py:	This program takes the boxoffice csv as input (‘boxoffice.csv’ and 'boxoffice_cleaned.csv’) and will generate a score for the two csv file. 3. boxoffice_clean.py:	This program takes the boxoffice data (‘boxoffice.csv’) as input, and generates the cleaned version of boxoffice (‘boxoffice_cleaned.csv’) as output. 
4. ratings_scrape.py:	This program is to scrape movie rating data from omdbapi based on the movie name from boxoffice.csv.
5. ratings_score.py:	This program takes the movie rating csv as input (‘ratings.csv’ and 'ratings_cleaned.csv’) and will generate a score for the two csv file. 
6. ratings_clean.py:	This program takes the movie rating data (‘ratings.csv’) as input, and generates the cleaned version of movie rating (‘ratings_cleaned.csv’) as output.