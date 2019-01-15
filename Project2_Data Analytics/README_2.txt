=============================
Python Code Files:
Code Files includes all detailed analysis results of the project.
Run the scripts in the order of: data_cleaning.py -> exploratory_analysis.py -> predictive_analysis.py


data_cleaning.py: Run this script first, this python script does the following things:

1. Merge original dataset (boxoffice_cleaned.csv and ratings_cleaned.csv)

2. Manage NA values (fill NA values or delete rows or columns)

3. Calculate data statistics and print 

4. Detect outliers and print information about outliers

5. Create bins of many columns, print

6. Generate final dataset (final.csv) for further analysis.


exploratory_analysis.py: This python script does the following things:

1. Plot histograms of some attributes

2. Generate correlation table and make scatterplot of some attributes 

3. K-means, Hierarchical, DBscan Clustering

4. Association Rule (Apriori)


predictive_analysis.py: This python script does the following things:

1. 3 Hypothesis Test using ANOVA and T-test

Accuracy (k-fold cross validation), ROC, Confusion Matrix using below algorithms:
2. Linear Regression

3. Decision Tree

4. KNN

5. Naive Bayes

6. SVM

7. Random Forest

=============================
Project Report Document: 
ANLY501 Project2 Report.pdf
The report addresses and discusses all noted requirements and elements.

=============================
Datasets:
Input datasets:
boxoffice_cleaned.csv
ratings_cleaned.csv

Output dataset:
final.csv

=============================
Extra Credits:
1. We used both T-test and ANOVA test for hypothesis testing

2. We constructed 3 types of Naive Bayes models: Gaussian, Mulinomial and Bernoulli.

3. We constructed 3 types of SVM models: Linear SVC, SVC with linear kernel, SVC with RBF kernel and SVC with Polynomial Kernal.

4. We used different ways to select features for classification models:
a. correlation
b. RFE
c. chi square test





