import subprocess 
import os
import numpy as np
import pandas as pd
import numpy.ma as ma

import env
import scipy
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt

import matplotlib.pyplot as plt
#%matplotlib inline #this is giving incorrect syntaxt when imported in ac, so hopefully keeping it in Presentation will work
import seaborn as sns
from matplotlib import rcParams
import matplotlib

# ignore warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from scipy import stats

from env import get_db_url

from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#IMPORTS above pull in all of the libraries and stats things that we'll need to
#crunch these numbers

####################################################################
#We begin acquiring data below \/ \/ \/

def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df.
    '''
    sql_query = """
                select bedroomcnt,
                bathroomcnt,
                calculatedfinishedsquarefeet,
                taxvaluedollarcnt,
                yearbuilt,
                taxamount,
                fips
                from zillow.properties_2017
                
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    
    return df

#-------------------------------------------------------------------

def get_zillow_data():
    '''
    This function reads in zil data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df

#6037 = LA County, 6059 = Orange County , 6111 = Ventura County
##############PREPARE PREPARE PREPARE##########################beprepaaaared######

#----------------------ZILLOW DATA PREP----------------------

# Replace white space values with NaN values.
def prep_zillow_data(df):
    df = df.replace(r'^\s*$', np.nan, regex=True)
    #drop rows with NaNs
    df = df.dropna()
    #convert all columns to int32 data types to speed up processing as suggested by other contestants
    df = df.astype(np.float32)
    df = df[df.bedroomcnt != 0]
    df = df[df.bathroomcnt != 0]
    df = df[df.bedroomcnt < 11]
    df = df[df.bathroomcnt < 16]
    df = df[df.calculatedfinishedsquarefeet <= 20000]
    df = df[df.yearbuilt > 1874]
    df = df[df.taxvaluedollarcnt < 2000000]
    df['fips'].replace([6037, 6059, 6111],['LA County', 'Orange County', 'Ventura County'], inplace=True)
    
#6037 = LA County, 6059 = Orange County , 6111 = Ventura County    

    return df

#-----TVT split------
def split_zillow_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test                         


#------------- ZILLOW XYSPLIT------------------


def zsplit(train, validate, test):
    x_train = train.drop(columns=['taxvaluedollarcnt'])
    y_train = train.taxvaluedollarcnt
    x_validate = validate.drop(columns=['taxvaluedollarcnt'])
    y_validate = validate.taxvaluedollarcnt
    x_test = test.drop(columns=['taxvaluedollarcnt'])
    y_test = test.taxvaluedollarcnt
    return x_train, y_train, x_validate, y_validate, x_test, y_test


#-----scalers-----
def seath(x_train, x_validate, x_test):   
    
    Robscaler = sklearn.preprocessing.RobustScaler()
# Note that we only call .fit with the training data,
# but we use .transform to apply the scaling to all the data splits.
    Robscaler.fit(x_train)
    x_train_Robscaled = Robscaler.transform(x_train)
    x_validate_Robscaled = Robscaler.transform(x_validate)
    x_test_Robscaled = Robscaler.transform(x_test)
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(x_train, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(x_train_Robscaled, ec='black')
    plt.title('Scaled')
    
    Standscaler = sklearn.preprocessing.StandardScaler()
# Note that we only call .fit with the training data,
# but we use .transform to apply the scaling to all the data splits.
    Standscaler.fit(x_train)
    x_train_Standscaled = Standscaler.transform(x_train)
    x_validate_Standscaled = Standscaler.transform(x_validate)
    x_test_Standscaled = Standscaler.transform(x_test)
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(x_train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(x_train_Standscaled, bins=25, ec='black')
    plt.title('Scaled')
    
    
    return x_train_Robscaled, x_validate_Robscaled, x_test_Robscaled, x_train_Standscaled, x_validate_Standscaled, x_test_Standscaled


                         
######################## ZILLOW WRANGLER ####################

def wrangle_zillow_data():
    df = get_zillow_data()
    df = prep_zillow_data(df)
    train, validate, test = split_zillow_data(df)
    x_train, y_train, x_validate, y_validate, x_test, y_test = ZXYsplit(train, validate, test)
    return x_train, y_train, x_validate, x_test, y_test
    return df



    
    
    
###########    
    
    
def get_reg_test(x_train, x_test, y_train, y_test):
    '''get logistic regression accuracy on train and validate data'''

    # create model object and fit it to the training data
    logit = LogisticRegression(solver='liblinear')
    logit.fit(x_train, y_train)

    # print result
    print(f"Accuracy of Logistic Regression on test is {logit.score(x_test, y_test)}")
    
    
###########

def get_knn_test(x_train, x_test, y_train, y_test):
    '''get KNN accuracy on train and validate data'''

    # create model object and fit it to the training data
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(x_train, y_train)

    # print results
  #  print(f"Accuracy of KNN on train is {knn.score(x_train, y_train)}")
    print(f"Accuracy of KNN on test is {knn.score(x_test, y_test)}")




########

    
def getpredicts(x_train, y_train, x_validate, y_validate, x_test, y_test):    
    rf = RandomForestClassifier(max_depth=3, random_state=123)    
    rf.fit(x_train, y_train)    
    y_pred = rf.predict(x_train)
    y_pred_proba = rf.predict_proba(x_train)
    print("accuracy of random forest calssifier on training set{:.2f}"
     .format(rf.score(x_train, y_train)))
    print(confusion_matrix(y_train, y_pred))
    print(classification_report(y_train, y_pred))
    print('Accuracy of random forest classifier on test set: {:.2f}'
     .format(rf.score(x_validate, y_validate)))
    #pr.get_reg(x_train, x_validate, y_train, y_validate)
    
    
    arrXT = x_train
    arrYT = y_train
    arrXV = x_validate
    arrYV = y_validate
    arrTX = x_test
    arrTY = y_test
    # convert to tuple
    tupXT = tuple(arrXT)
    tupYT = tuple(arrYT)
    tupXV = tuple(arrXV)
    tupYV = tuple(arrYV)
    tupTX = tuple(arrTX)
    tupTY = tuple(arrTY)

    # set tuple as key
    #dict = {tup: 'value'}
    #print(dict)
    
    rmse_val = []
    K = 5
    kayenen = neighbors.KNeighborsClassifier(n_neighbors = 5)
    kayenen.fit(x_train, y_train)  #fit the model
    kpred = kayenen.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(tupTY,kpred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    cm = confusion_matrix(tupTY, kpred)
    print('RMSE value for k= ' , K , 'is:', error)
    predit = kayenen.predict(x_test)
    resul = ((904+180)/(904+180+129+194))
    print('Accuracy of KNN on test set: {}'.format(resul))


    #SUBPLOT GOES HERE



#--------------------PLOTS----------
#this code gets us a scatter plot showing calculatedtotalsquarefeet and taxvaluedollarcnt

def getfirstplot(train):
    rcParams['figure.figsize']=10,10
    sns.scatterplot(train.calculatedfinishedsquarefeet, train.taxvaluedollarcnt) 
    #rcParams['figure.figsize']=10,10  #may need to run it a few times to get the correct figsize
    

    
    
    



###########    
    
    #this code gets us a  plot showing location and values value
def getsecondplot(train):
    rcParams['figure.figsize']=4,8
    tempplotdf = train[['fips','taxvaluedollarcnt']].copy()
    lowerrangedf = train.taxvaluedollarcnt.where(train.taxvaluedollarcnt < 600000, inplace=False)
    ventmean = tempplotdf.where(tempplotdf.fips=='Ventura County') 
    ventmean = ventmean.dropna()
    VMS = np.float32(ventmean.taxvaluedollarcnt.sum()/len(ventmean.index))

    orangemean = tempplotdf.where(tempplotdf.fips=='Orange County')
    orangemean = orangemean.dropna()
    OMS = np.float32(orangemean.taxvaluedollarcnt.sum()/len(orangemean.index))

    lamean = tempplotdf.where(tempplotdf.fips=='LA County')
    lamean = lamean.dropna()
    LAMS = np.float32(lamean.taxvaluedollarcnt.sum()/len(lamean.index))
    #THE AVERAGE PRICE FOR A HOME IN EACH COUNTY /\ /\ /\

    #rcParams['figure.figsize']=4,8  
    
    
    PTT = sns.scatterplot(x=train.fips, y=train.taxvaluedollarcnt)
    
    
    
    #PTT.set_yscale("log")
    PTT.axhline(y=LAMS*1)
    PTT.axhline(y=OMS*1)
    PTT.axhline(y=VMS*1)
    tempplotdf
    PTT.annotate('Orange', xy=(2, OMS*1), xytext=(3, OMS*1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
    PTT.annotate('Ventura', xy=(2, VMS*1), xytext=(3, VMS*1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
    PTT.annotate('LA', xy=(2, LAMS*1), xytext=(3, LAMS*.9),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
    #MASH = {
    #    'County':["Orange County","Ventura County","LA County"],
    #    'Average Home Value':[OMS,VMS,LAMS]
   # fig, axes = plt.subplots(PTT, PTT2)
   
    #print('Ventura Average =', VMS, ', Orange Average =', OMS,', LA Average =', LAMS)
    
     
        
     
        #hue=train.fips, palette='bright'
    
    #PTT.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1,labels=['No Svc', 'FiberOp','DSL'])
    #PTT.set(xlabel='Mailed Check - 0           Electronic Check - 1           Bank Draft (auto) - 2           Credit Card (auto) - 3')
        

def cuffedplot(train):
    tempplotdf = train[['fips','taxvaluedollarcnt']].copy()
    lowerrangedf = train.where(train.taxvaluedollarcnt < 600000, inplace=False)
    
    ventmean = tempplotdf.where(tempplotdf.fips=='Ventura County') 
    ventmean = ventmean.dropna()
    VMS = np.float32(ventmean.taxvaluedollarcnt.sum()/len(ventmean.index))

    orangemean = tempplotdf.where(tempplotdf.fips=='Orange County')
    orangemean = orangemean.dropna()
    OMS = np.float32(orangemean.taxvaluedollarcnt.sum()/len(orangemean.index))

    lamean = tempplotdf.where(tempplotdf.fips=='LA County')
    lamean = lamean.dropna()
    LAMS = np.float32(lamean.taxvaluedollarcnt.sum()/len(lamean.index))
    #THE AVERAGE PRICE FOR A HOME IN EACH COUNTY /\ /\ /\
    
    
    CUFF = sns.scatterplot(x=lowerrangedf.fips, y=lowerrangedf.taxvaluedollarcnt)   
    CUFF.ylim=(0,650000) 
    
    CUFF.axhline(y=LAMS*1)
    CUFF.axhline(y=OMS*1)
    CUFF.axhline(y=VMS*1)
    
    CUFF.annotate('Orange', xy=(2, OMS*1), xytext=(3, OMS*1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
    CUFF.annotate('Ventura', xy=(2, VMS*1), xytext=(3, VMS*1.1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
    CUFF.annotate('LA', xy=(2, LAMS*1), xytext=(3, LAMS*1),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )   
        
        
        
###########        
        
#this code gets us a scatter plot with bedrooms x value, and bathrooms x value     
def getthirdplot(train):
    #CHANGE TO HIST PLOT AND THE SUM OF COLUMNS
    TD = sns.relplot(x=train.bedroomcnt, y=train.taxvaluedollarcnt, height=4, aspect=3)
    #TD.set(ylabel='Monthly   |   One-Year   |   Two-Year')
    TD2 = sns.relplot(x=train.bathroomcnt, y=train.taxvaluedollarcnt, height=4, aspect=3)
    
    
    
    
    
########### 
    
    
    
    
#this code gets us a scatter plot showing year built vs value
def getfourthplot(train):
    #rcParams['figure.figsize']=30,30
    INTC = sns.relplot(data=train, x=train.yearbuilt, y=train.taxvaluedollarcnt, col=train.fips, hue=train.fips, height = 10, aspect = 1)  
   # INTC.set_titles('')
   # INTC.fig.suptitle("0-None                                                                    1-FiberOptic                                                                    2-DSL")
    #INTC.fig.subplots_adjust(top=1)
 
    
    
    
    
    
###########        






#this code gets us a scatter plot showing monthly charge, total, and churn status      
def getfifthplot(train):
    rcParams['figure.figsize']=8,8
    sns.scatterplot(data=train, x=train.monthly_charges, y=train.total_charges, hue=train.churn)
    
    
    
    
    
    
    
############################## CLASSIFIERS ###########################

#This creates a decision tree classifier and runs the math for our training data

def get_tree(x_train, x_validate, y_train, y_validate):
    '''get decision tree accuracy on train and validate data'''
    x_train = x_train.drop(labels=fips)
    # create classifier object
    clf = DecisionTreeClassifier(max_depth=5, random_state=123)

    #fit model on training data
    clf = clf.fit(x_train, y_train)

    # print result
    print(f"Accuracy of Decision Tree on train data is {clf.score(x_train, y_train)}")
    print(f"Accuracy of Decision Tree on validate data is {clf.score(x_validate, y_validate)}")
    

#This creates a random forest classifier and runs the math for our training data

def get_forest(train_X, validate_X, train_y, validate_y):
    '''get random forest accuracy on train and validate data'''

    # create model object and fit it to training data
    rf = RandomForestClassifier(max_depth=4, random_state=123)
    rf.fit(train_X,train_y)

    # print result
    print(f"Accuracy of Random Forest on train is {rf.score(train_X, train_y)}")
    print(f"Accuracy of Random Forest on validate is {rf.score(validate_X, validate_y)}")

#This creates a  KNN classifier and runs the math for our training data

def get_knn(x_train, x_validate, y_train, y_validate):
    '''get KNN accuracy on train and validate data'''

    # create model object and fit it to the training data
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(x_train, y_train)

    # print results
    print(f"Accuracy of KNN on train is {knn.score(x_train, y_train)}")
    print(f"Accuracy of KNN on validate is {knn.score(x_validate, y_validate)}")

#This creates a linear regression classifier and runs the math for our training data
    
def get_reg(x_train, x_validate, y_train, y_validate):
    '''get logistic regression accuracy on train and validate data'''

    # create model object and fit it to the training data
    logit = LogisticRegression(solver='liblinear')
    logit.fit(x_train, y_train)

    # print result
    print(f"Accuracy of Logistic Regression on train is {logit.score(x_train, y_train)}")
    print(f"Accuracy of Logistic Regression on validate is {logit.score(x_validate, y_validate)}")


    
#-------    
    

# 
#def getpredictions(x_train, y_train, x_test, y_test):
#     indy = x_train.index
#     nrows = len(y_train)
#     total_sample_size = 1e4
#     x_train.groupby('gender').\
   
#     #x_train = x_train.values.flatten()
#     y_train = y_train.values.flatten()
#     #x_test = x_test.values.flatten()
#     #y_test = y_test.values.flatten()
#     logreg = LogisticRegression()
#     logreg.fit(x_train.values, y_train)


# def getfit(x_train, y_train):
#     fitted = tree.fit(x_train, y_train)
#     return fitted
    


##################### GET SOME ANALYSIS #############################

def getfirsttest(x_train, y_train):
    geh = np.corrcoef(x_train_, y_train)
    print(geh)


#def getchifirst(train):
 #   '''get results of chi-square for total sq ft and value'''
#
#    observed = pd.crosstab(train.taxvaluedollarcnt, train.calculatedfinishedsquarefeet)
#
#    chi2, p, degf, expected = stats.chi2_contingency(observed)
#
#    print(f'chi^2 = {chi2:.4f}')
#    print(f'p     = {p:.4f}')   
    
    
#--------    
    
    
#def getchisecond(train):
#    '''get rusults of chi-square for payment type and services'''

#    observed = pd.crosstab(train.payment_type, train.internet_service_type)

#    chi2, p, degf, expected = stats.chi2_contingency(observed)

#    print(f'chi^2 = {chi2:.4f}')
#    print(f'p     = {p:.4f}')   
                     
                 
#--------                 
    
    
#def getchithird(train):
#    '''get rusults of chi-square for contract type and churn status'''

#    observed = pd.crosstab(train.contract_type, train.churn)

#    chi2, p, degf, expected = stats.chi2_contingency(observed)

#    print(f'chi^2 = {chi2:.4f}')
#    print(f'p     = {p:.4f}')   

#--------
    
#def getchifourth(train):
#    '''get rusults of chi-square for internet service type and churn status'''

#    observed = pd.crosstab(train.contract_type, train.churn)

#    chi2, p, degf, expected = stats.chi2_contingency(observed)

#    print(f'chi^2 = {chi2:.4f}')
#    print(f'p     = {p:.4f}')  
#--------

#def getchififth(train):
#    '''get rusults of chi-square for monthly charges and churn status'''

#    observed = pd.crosstab(train.monthly_charges, train.churn)

#    chi2, p, degf, expected = stats.chi2_contingency(observed)

#    print(f'chi^2 = {chi2:.4f}')
#    print(f'p     = {p:.4f}')  
    
    
#---------------------OLD TELCO STUFF-------------------
#---------------------new dataframe----------------------------------------
    

def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a df.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('telco_churn'))
    
    return df


#----------------------get data if its local---------------------------------
def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('telco.csv')
        
        
    return df

#-------------------------------------------------------------------
                        
# ------------------- TELCO DATA -------------------
def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test
#-------------------TELCO-------------------------
def xysplit(train, validate, test):
    x_train = train.drop(columns=['churn'])
    y_train = train.churn

    x_validate = validate.drop(columns=['churn'])
    y_validate = validate.churn

    x_test = test.drop(columns=['churn'])
    y_test = test.churn
    
    my_var = 0 if my_var is None else my_var
    
#---------------------TELCO-----------------------------------------                        

def prep_telco_data(df):
     # Drop duplicate columns
    df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
       
     # Drop null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Convert to correct datatype
    df['total_charges'] = df.total_charges.astype(float)

    #trying to drop nulls if any in monthly charges
    df['monthly_charges'] = df['monthly_charges'].str.strip()
    df = df[df.total_charges != '']
    
    #converted to float
    df['monthly_charges'] = df.total_charges.astype(float)
    
    
#I was getting something out of place so I split them out differently
#Each line replaces the values which aren't numbers, with numbers so they can be
#processed/analyzed

    df['gender'].replace(['Female', 'Male'],[1,0], inplace=True)
    df['contract_type'].replace(['Two year', 'One year', 'Month-to-month'],[2,1,0], inplace=True)
    df['partner'].replace(['Yes', 'No'],[1,0], inplace=True)
    df['dependents'].replace(['Yes', 'No'],[1,0], inplace=True)
    df['paperless_billing'].replace(['Yes', 'No'],[1,0], inplace=True)
    df['phone_service'].replace(['Yes', 'No'],[1,0], inplace=True)
    df['multiple_lines'].replace(['Yes', 'No', 'No phone service'],[1,0,00], inplace=True)
    df['churn'].replace(['Yes', 'No'],[1,0], inplace=True)
    df['online_security'].replace(['Yes', 'No', 'No internet service'],[1,0,00], inplace=True)
    df['online_backup'].replace(['Yes', 'No','No internet service'],[1,0,00], inplace=True)	

    df['device_protection'].replace(['Yes', 'No','No internet service'],[1,0,00], inplace=True)	
    df['tech_support'].replace(['Yes', 'No','No internet service'],[1,0,00], inplace=True)	
    df['streaming_tv'].replace(['Yes', 'No','No internet service'],[1,0,00], inplace=True)	
    df['streaming_movies'].replace(['Yes', 'No','No internet service'],[1,0,00], inplace=True)	

  # df['monthly_charges'].replace(['Yes', 'No'],[1,0], inplace=True)	
    df['internet_service_type'].replace(['DSL', 'Fiber optic','None'],[2,1,0], inplace=True)	
    df['payment_type'].replace(['Mailed check', 'Electronic check', "Bank transfer (automatic)", "Credit card (automatic)" ],[0,1,2,3], inplace=True)	

    
    
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[[#'multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=True)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    
    df = df.T.drop_duplicates().T
    
    
    return df 
    

   
                            
                         
    