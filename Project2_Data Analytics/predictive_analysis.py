#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 19:05:30 2018

@author: Tianyi Yang, Xinyue Liu, Zikai Zhu, Ju Huang
"""
import pandas as pd
import numpy as np
from scipy import stats
########################################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold  
########################################
# IGNORE the Warnings!
import warnings
warnings.filterwarnings("ignore")
########################################

def further_grouping(final):
    # Further groupings for categorical varaibles (studios & rat with many categories
    final_g= final.copy()
    ### Studio: transform the studio col further to create dummy variables
    studios_lr = final_g['Studio_new'].value_counts()
    super_studio=final_g['Studio_new'].isin(studios_lr.index[(studios_lr<237)&(studios_lr>=90)])
    large_studio=final_g['Studio_new'].isin(studios_lr.index[studios_lr<90])
    final_g.loc[super_studio, 'Studio_new'] = 'Super_studios'
    final_g.loc[large_studio,'Studio_new']= 'Large_studios'
    ### Rates: transform the rated col further to create dummy variables   
    rates=final_g['Rated'].value_counts()
    other= list(rates[rates<50].index)
    final_g['Rated'][final_g['Rated'].isin(other)]= 'Other'
    return (final_g)    

######################################## 
# Indepedent T-test
########################################
def t_test(f):
    # Is there a difference between Rotten Tomato & Imdb rating?
    ratings=f[['Rotten Tomatoes','imdbRating']]
    ratings_c=ratings.dropna(axis=0,how='any')
    t,p=stats.ttest_ind(ratings_c['Rotten Tomatoes'],ratings_c['imdbRating'])
    print('\nIndepedent T-test on Rotten Tomatoes vs. imdb')
    print('t: '+str(t))
    print('p: '+str(p))

######################################## 
# Anova Analysis
########################################         
#for diffrent year, is there a significant change in boxoffice
#final.boxplot('Total Gross', by='Year_x', figsize=(12, 8))
def anova(f):
    bo_by_year={y:f['Total Gross'][f.Year_x==y] for y in f['Year_x']}
    F, p_a = stats.f_oneway(bo_by_year[2010], 
                            bo_by_year[2011],
                            bo_by_year[2012],
                            bo_by_year[2013],
                            bo_by_year[2014],
                            bo_by_year[2015],
                            bo_by_year[2016],
                            bo_by_year[2017])
    print('\nAvona Test on Boxoffice across years')
    print('F: '+str(F))
    print('p_year: '+str(p_a))
    
    #for different-sized studios, is tehre a significant difference in boxoffice?
    final_c=further_grouping(f)
    #final_c['Studio_new'].value_counts()
    bo_by_s={s:final_c['Total Gross'][final_c.Studio_new==s] for s in final_c['Studio_new']}
    F, p_s = stats.f_oneway(bo_by_s['Super_studios'], 
                            bo_by_s['Large_studios'],
                            bo_by_s['Mid_sized'],
                            bo_by_s['Indie'])
    print('\nAvona Test on Boxoffice across studios')
    print('F: '+str(F))
    print('p_studio: '+str(p_s))
    
    #for different-rated movies, is tehre a significant difference in boxoffice?
    final_c=further_grouping(f)
    #final_c['Studio_new'].value_counts()
    bo_by_r={r:final_c['Total Gross'][final_c.Rated==r] for r in final_c['Rated']}
    bo_by_r.keys()
    F, p_r = stats.f_oneway(bo_by_r['Other'], 
                            bo_by_r['PG'],
                            bo_by_r['PG-13'],
                            bo_by_r['R'],
                            bo_by_r['NOT RATED'])
    print('\nAvona Test on Boxoffice across rate')
    print('F: '+str(F))
    print('p_rated: '+str(p_r))
    
######################################## 
# Linear Regression 
######################################## 
def get_df_for_lr(f):
    # for linear regression, we want to transform & filter certain features as described below 
    ### Categorical Variables: get dummies for studios & rated 
    final_lr=further_grouping(f)
    dummy_lr=pd.get_dummies(final_lr[['Studio_new','Rated']], prefix=['Studios', 'Rated'])
    ### Numeric Variables: 1. find cols with corr. at least 0.2 with Total Gross 2. avoid highly corr. independent varaibles
    #final[['Total Gross','All Theaters','Opening Theaters','Timespan','Metascore','Rotten Tomatoes','imdbRating','imdbVotes','combined_rating']].corr()
    final_lr=final_lr[['Total Gross','All Theaters','Timespan','imdbVotes','Opening Theaters']]
    final_lr= final_lr.merge(dummy_lr,left_index=True,right_index=True)
    return final_lr

def linear_reg_setup(f,target,features):
    # Set up cross validation through splitting the df into train & test 
    final_lr=get_df_for_lr(f)
    final_lr=final_lr.dropna(how='any')
    x_train, x_test, y_train, y_test= train_test_split(final_lr[features],final_lr[target],test_size=0.3,random_state=1)    
    return (x_train, x_test, y_train, y_test)

def linear_reg(f,target,features):
    # Write the function to run linear regression & get evaluation score   
    x_train, x_test, y_train, y_test= linear_reg_setup(f,target,features)
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    variables= list(x_train)
    coef= lr.coef_[0].tolist()
    result=pd.DataFrame({'Variable':variables,'Coef.':coef})
    print('\nLinear Regression Coefficient:')
    print (tabulate(result, headers='keys',tablefmt="github"))
    train_prediction= lr.predict(x_train)
    #get result for train
    train_mse=mean_squared_error(train_prediction,y_train)
    train_rmse= np.sqrt(train_mse)
    print('\ntrain_rmse: '+str(round(train_rmse,2)))
    print('train_R²: ',round(lr.score(x_train,y_train),2))
    #predict the test set
    test_prediction= lr.predict(x_test)
    test_mse= mean_squared_error(test_prediction,y_test)
    test_rmse=np.sqrt(test_mse)
    print('test_rmse: '+str(round(test_rmse,2)))
    print('test_R²: ',round(lr.score(x_test,y_test),2))

def lr_get_feature(f):
    # Select features with rfe& crete feature combinations 
    target=['Total Gross']
    numeric=['All Theaters','Timespan','imdbVotes','Opening Theaters']
    cat1=['Studios_Large_studios','Studios_Mid_sized', 'Studios_Super_studios']
    cat2=['Rated_Other', 'Rated_PG', 'Rated_PG-13', 'Rated_R']
    # Use rfe feature selection to get rank for numeric variables 
    x_train, x_test, y_train, y_test= linear_reg_setup(f,target,numeric)
    # Run regression
    lr=LinearRegression()
    rfe = RFE(lr, 1)
    rfe = rfe.fit(x_train, y_train)  
    feature_ranking=pd.DataFrame({'Variable':numeric,'Ranking':(rfe.ranking_).tolist()}) 
    print('\n\nFeature Selection Numeric Varaible Ranking for Linear Regression:\n')
    print (tabulate(feature_ranking, headers='keys',tablefmt="github"))
    feature1= ['All Theaters','Timespan']+cat1+cat2
    feature2= ['All Theaters','Timespan']+cat1
    feature3= ['All Theaters','Timespan']+cat2
    feature4= ['All Theaters','Timespan']
    feature5= ['All Theaters','Timespan','imdbVotes']+cat1+cat2
    feature6= ['All Theaters','Timespan','imdbVotes']+cat1
    feature7= ['All Theaters','Timespan','imdbVotes']+cat2
    feature8= ['All Theaters','Timespan','imdbVotes']
    return([feature1, feature2,feature3,feature4,feature5,feature6,feature7, feature8])

def linear_regression(f):
    # Run regression with 8 combination of features 
    features= lr_get_feature(f)
    for i in range(8):
        target=['Total Gross']
        print('\n\n\nRESULT FOR LINEAR REGRESSION MODEL',i+1)
        linear_reg(f,target,features[i])

###############################
# Methods
###############################
def eval_classification(df,best_feature,target,classifier):
    # Print confusion matrix 
    x_train, x_test, y_train, y_test = train_test_split(df[best_feature], df[target], test_size=0.3, random_state=1)
    model= classifier
    model.fit(x_train,y_train)
    test_prediction = model.predict(x_test)
    print('\nConfusion Matrix for the model with highest accuracy:\n')
    print(confusion_matrix(y_test, test_prediction))
    # Generate ROC FP & TP values
    y_score= model.predict_proba(x_test)
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    y_test=label_binarize(y_test, classes=[1,2,3,4,5,6])
    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    print('\n\nROC Curve for the model with highest accuracy:')
    # Plot ROC curve 
    fig= plt.figure(figsize=(8, 8))
    for i in range(1,7):
        ax = fig.add_subplot(3,2,i)
        ax.plot(fpr[i-1], tpr[i-1], label='ROC curve' % roc_auc[i-1])
        ax.plot([0, 1], [0, 1], 'k--')
    fig.text(0.5, 0.04, 'False Positive Rate', ha='center')
    fig.text(0.04, 0.5, 'True Positive Rate', va='center', rotation='vertical')
    plt.show()
    #plt.savefig('ROC Curve_'+str(classifier)+'.png')
    
##############################
# KNN
##############################
def get_df_knn(f, c_knn):
    final_knn=f[c_knn].dropna()
    transform_cols=['Rated','Studio_new','Year_x']
    lb_make = LabelEncoder()
    for col in transform_cols:
        final_knn[col] = lb_make.fit_transform(final_knn[col]) 
    return (final_knn)

def get_feature_knn(f, c_knn, f_knn, t_knn):
    final_knn= get_df_knn(f, c_knn)
    x_train, x_test, y_train, y_test = train_test_split(final_knn[f_knn], final_knn[t_knn], test_size=0.3, random_state=1)
    knn=KNeighborsClassifier()
    knn.fit(x_train, y_train)
    # see the correlation of each attribute
    #final_knn_value = final_knn.values
    #final_knn_df= pd.DataFrame(final_knn_value).astype(int)
    #final_knn_corr = final_knn_df.corr()
    #print('\n\nCorrelation Table:\n')
    #print (tabulate(final_knn_corr, headers='keys',tablefmt="psql"))
    #create feature combination to test based on the correlations (selection strategy specified is report)
    feature1=['All Theaters Bin','Timespan Bin']
    feature2=['All Theaters Bin','Timespan Bin','imdbVotes Bin']
    feature3=['All Theaters Bin','Timespan Bin','imdbVotes Bin','Rated']
    feature4=['All Theaters Bin','Timespan Bin','imdbVotes Bin','Rated','Studio_new']
    return [feature1, feature2,feature3,feature4]

def knn(f, c_knn, f_knn, t_knn):
    final_knn= get_df_knn(f, c_knn)
    features= get_feature_knn(f, c_knn, f_knn, t_knn)
    accuracy=list()
    model= KNeighborsClassifier()
    num_folds = 10
    seed = 0
    scoring = 'accuracy'
    for i in range(4):
        x_train, x_test, y_train, y_test = train_test_split(final_knn[features[i]], final_knn[t_knn], test_size=0.3, random_state=1)
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        accuracy.append(cv_results.mean())
    feature=[', '.join(i) for i in features]
    result=pd.DataFrame({'Accuracy':accuracy,'Features':feature})
    best_knn=features[result['Accuracy'].idxmax()]
    target=t_knn
    print('\n\nK-nearest Neighbors Model Evaluation:\n')
    print (tabulate(result, headers='keys',tablefmt="psql"))
    eval_classification(final_knn,best_knn,target,KNeighborsClassifier())

#######################################
# Decision Tree
#######################################
    
def get_feature_dt():
    #create feature combination to test based on the correlations (selection strategy specified is report)
    feature1=['All Theaters Bin','Timespan Bin']
    feature2=['All Theaters Bin','Timespan Bin','imdbVotes Bin']
    feature3=['All Theaters Bin','Timespan Bin','imdbVotes Bin','Rated']
    feature4=['All Theaters Bin','Timespan Bin','imdbVotes Bin','Rated','Studio_new']
    return [feature1, feature2,feature3,feature4]

def dt(f, c_dt, f_dt, t_dt):
    final_dt=get_df_knn(f, c_dt)
    features=get_feature_dt()
    accuracy=list()
    model= DecisionTreeClassifier()
    num_folds = 10
    seed = 7
    scoring = 'accuracy'
    for i in range(4):
        x_train, x_test, y_train, y_test = train_test_split(final_dt[f_dt], final_dt[t_dt], test_size=0.3, random_state=1)
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        accuracy.append(cv_results.mean())
    feature=[', '.join(i) for i in features]
    result=pd.DataFrame({'Accuracy':accuracy,'Features':feature})
    best_dt=features[result['Accuracy'].idxmax()]
    target=t_dt
    print('\n\nDecision Tree Model Evaluation:\n')
    print (tabulate(result, headers='keys',tablefmt="psql"))
    eval_classification(final_dt,best_dt,target,DecisionTreeClassifier())

######################################## 
# Random Forest 
########################################
def get_df_rf(f, c_rf):
    # Transform the dataframe to get a dataframe for random forest 
    final_rf=f[c_rf].dropna()
    transform_cols=['Rated','Studio_new','Year_x']
    lb_make = LabelEncoder()
    for col in transform_cols:
        final_rf[col] = lb_make.fit_transform(final_rf[col]) 
    return (final_rf)

def get_feature_rf(f, c_rf, f_rf, t_rf):
    # Select features for random forest 
    final_rf= get_df_rf(f, c_rf)
    x_train, x_test, y_train, y_test = train_test_split(final_rf[f_rf], final_rf[t_rf], test_size=0.3, random_state=1)
    rf=RandomForestClassifier()
    rf.fit(x_train, y_train)
    feature_ranking=pd.DataFrame({'Variable':f_rf,'Importance':(rf.feature_importances_).tolist()}) 
    print('\n\nFeature Selection for Random Forest:\n')
    print (tabulate(feature_ranking, headers='keys',tablefmt="github"))
    #create feature combination to test based on selection score
    feature1=['All Theaters Bin','Timespan Bin']   
    feature2=['All Theaters Bin','Timespan Bin','Studio_new']
    feature3=['All Theaters Bin','Timespan Bin','Studio_new','Year_x']
    feature4=['All Theaters Bin','Timespan Bin','Studio_new','Year_x','imdbVotes Bin']
    #also run another model including opening boxoffice
    feature5=['Opening Bin','All Theaters Bin','Timespan Bin']
    feature6=['Opening Bin','All Theaters Bin','Timespan Bin','Studio_new']
    feature7=['Opening Bin','All Theaters Bin','Timespan Bin','Studio_new','Year_x']
    return [feature1, feature2,feature3,feature4,feature5,feature6,feature7]

def rf(f, c_rf, f_rf, t_rf):
    # Run random forests 
    final_rf= get_df_rf(f, c_rf)
    features= get_feature_rf(f, c_rf, f_rf, t_rf)
    accuracy=list()
    model= RandomForestClassifier()
    num_folds = 10
    seed = 7
    scoring = 'accuracy'   
    for i in range(7):
        x_train, x_test, y_train, y_test = train_test_split(final_rf[features[i]], final_rf[t_rf], test_size=0.3, random_state=1)
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        accuracy.append(cv_results.mean())
    feature=[', '.join(i) for i in features]
    result=pd.DataFrame({'Accuracy':accuracy,'Features':feature})
    print('\n\nRandom Forest Model Evaluation:\n')
    print (tabulate(result, headers='keys',tablefmt="psql"))
    # Figure out what is the best feature through the accuracy score 
    best_f_no_opening=features[result['Accuracy'][:4].idxmax()]
    best_f_opening=features[result['Accuracy'][4:].idxmax()]
    target=t_rf
    # Plot the confusion matrix for the best models 
    print('\n\nBest RF Model without opening boxoffice variable:')
    eval_classification(final_rf,best_f_no_opening,target,RandomForestClassifier())
    print('\n\nBest RF Model with opening boxoffice varaible:')
    eval_classification(final_rf,best_f_opening,target,RandomForestClassifier())

######################################## 
# Naive Bayes 
########################################
def get_df_nb(f, c_nb):
    final_nb=f[c_nb].dropna()
    transform_cols=['Rated','Studio_new','Year_x']
    lb_make = LabelEncoder()
    for col in transform_cols:
        final_nb[col] = lb_make.fit_transform(final_nb[col]) 
    return (final_nb)

def get_feature_nb(f, c_nb, f_nb, t_nb):
    final_nb= get_df_nb(f, c_nb)
    feature = []
    # Feature selection using Chi-Square test
    for k in range(2, 6):
        selector = SelectKBest(score_func=chi2, k=k)
        s_fit = selector.fit(final_nb[f_nb], final_nb[t_nb])
        f_name = final_nb[f_nb].columns[s_fit.get_support()]
        feature += [list(f_name)]
    return feature

def nb(f, c_nb, f_nb, t_nb, model_name, model):     
    final_nb= get_df_nb(f, c_nb)
    features= get_feature_nb(f, c_nb, f_nb, t_nb)
    accuracy=list()
    model= model
    num_folds = 10
    seed = 7
    scoring = 'accuracy'   
    for i in range(4):
        x_train, x_test, y_train, y_test = train_test_split(final_nb[features[i]], final_nb[t_nb], test_size=0.3, random_state=1)
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        accuracy.append(cv_results.mean())
    feature=[', '.join(i) for i in features]
    result=pd.DataFrame({'Accuracy':accuracy,'Features':feature})
    best_nb=features[result['Accuracy'].idxmax()]
    target=t_nb
    print('\n\n',model_name, 'Naive Bayes Model Evaluation:\n')
    print (tabulate(result, headers='keys',tablefmt="psql"))
    eval_classification(final_nb,best_nb,target,model)
    
######################################## 
# SVM
########################################
def get_feature_svm(f, c_svm, f_svm, t_svm):
    cols_nb=[['Total Gross Bin','Studio_new','All Theaters Bin', 'Timespan Bin',
              'combined_rating Bin']]
    return cols_nb

def svm(f, c_svm, f_svm, t_svm, model_name, model):
    final_svm= get_df_nb(f, c_svm)
    features= get_feature_nb(f, c_svm, f_svm, t_svm)
    accuracy=list()
    model= model
    num_folds = 10
    seed = 0
    scoring = 'accuracy'       
    for i in range(4):
        x_train, x_test, y_train, y_test = train_test_split(final_svm[features[i]], final_svm[t_svm], test_size=0.3, random_state=1)
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        accuracy.append(cv_results.mean())
    feature=[', '.join(i) for i in features]
    result=pd.DataFrame({'Accuracy':accuracy,'Features':feature})
    best_svm=features[result['Accuracy'].idxmax()]
    target=t_svm
    print('\n\n',model_name, 'SVM Model Evaluation:\n')
    print (tabulate(result, headers='keys',tablefmt="psql"))
    # Confusion Matrix for the 
    x_train, x_test, y_train, y_test = train_test_split(final_svm[best_svm], final_svm[target], test_size=0.3, random_state=1)
    model= model
    model.fit(x_train,y_train)
    test_prediction = model.predict(x_test)
    print('\nConfusion Matrix for the model with highest accuracy:\n')
    print(confusion_matrix(y_test, test_prediction))
    # Generate ROC FP & TP values
    y_score= model.decision_function(x_test)
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    y_test=label_binarize(y_test, classes=[1,2,3,4,5,6])
    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    print('\n\nROC Curve:')
    # Plot ROC curve 
    fig= plt.figure(figsize=(8, 8))
    for i in range(1,7):
        ax = fig.add_subplot(3,2,i)
        ax.plot(fpr[i-1], tpr[i-1], label='ROC curve' % roc_auc[i-1])
        ax.plot([0, 1], [0, 1], 'k--')
    fig.text(0.5, 0.04, 'False Positive Rate', ha='center')
    fig.text(0.04, 0.5, 'True Positive Rate', va='center', rotation='vertical')
    plt.show()
    
if __name__ == "__main__":
    # Read Data
    final = pd.read_csv('final.csv', encoding='utf-8',sep=',')
    
    ## Predictive Analytsis
    ## Parametric Statistical Tests
    # Run T-test
    t_test(final)
    # Run ANOVA Test
    anova(final)
    # Linear Regression
    linear_regression(final)
    
    ## Methods
    cols=['Total Gross Bin','Rated','Studio_new','Year_x','All Theaters Bin','Opening Bin', 'Opening Theaters Bin', 'Timespan Bin', 'Metascore Bin',
                    'Rotten Tomatoes Bin', 'combined_rating Bin', 'imdbVotes Bin']
    features= cols[1:]
    target= cols[0]
    # KNN  
    knn(final, cols, features, target)
    # Decision Tree
    dt(final, cols, features, target)
    # Random Forest
    rf(final, cols, features, target)
    
    # Naive Bayes
    cols_nb=['Total Gross Bin','Rated','Studio_new','Year_x','All Theaters Bin', 'Timespan Bin', 'Metascore Bin',
                    'Rotten Tomatoes Bin', 'combined_rating Bin', 'imdbVotes Bin']
    features_nb= cols_nb[1:]
    target_nb= cols_nb[0]
    nb_dict = {"Gaussian": GaussianNB(),
               "Multinomial": MultinomialNB(),
               "Bernoulli": BernoulliNB()}
    get_feature_nb(final, cols_nb, features_nb, target_nb)
    for k, v in nb_dict.items():
        nb(final, cols_nb, features_nb, target_nb, k, v)
        
    # SVM
    svm_dict = {"Linear Kernel": SVC(kernel='linear', C=1),
                "Linear": LinearSVC(C=1),
                "RBF(Gaussian) Kernel": SVC(kernel='rbf', gamma=0.7, C=1)}
                #"Polynomial Kernel (Degree=3)": SVC(kernel='poly', degree=3, C=1)}
    for k, v in svm_dict.items():
        svm(final, cols_nb, features_nb, target_nb, k, v)