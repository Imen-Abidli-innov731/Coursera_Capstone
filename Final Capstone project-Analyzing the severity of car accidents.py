#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install seaborn')
import pandas as pd
import numpy as np 
import sklearn as skl 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[2]:


#Import Data 
URL= 'https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv'
df=pd.read_csv(URL,low_memory=False)


# In[3]:


#delete colomns with 'not enough information' or not important information
df.drop(['SEVERITYCODE.1','REPORTNO','EXCEPTRSNCODE','EXCEPTRSNDESC','INATTENTIONIND','PEDROWNOTGRNT','SPEEDING','OBJECTID','INCKEY','COLDETKEY','STATUS','SDOTCOLNUM','SEGLANEKEY','CROSSWALKKEY'], inplace=True, axis=1, errors='ignore')


# In[4]:


#to count a category histogramm of each column 
ax = df.apply(lambda x: pd.factorize(x)[0]).hist(bins=25,figsize = (15,20))


# In[5]:


#to detect and count missing data
missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())


# In[6]:


#calculate a corroletion coefficient of original dataframe (before cleaning)
df_original_facto = df.apply(lambda x: pd.factorize(x)[0])

sns.heatmap(abs(df_original_facto.corr()))


# In[7]:


#Let's deal with missing values and cleaning data !


# In[8]:


#to remove duplicated rows
df.drop_duplicates( keep='first', inplace=True)


# In[9]:


# replace "?" to NaN
df.replace("?", np.nan, inplace = True)


# In[10]:


#Select columns with type "int" or "float"
num= df.select_dtypes(include=['float','int']).copy()


# In[11]:


missing_data = num.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())


# In[12]:


##Replace missing data by mean
L = ["Y","X","INTKEY"]
for i in L:
    avg_i = num[i].astype("float").mean(axis=0)
    print("Average of " + i + ":", avg_i)
    num[i].replace(np.nan, avg_i, inplace=True)


# In[13]:


#Select columns with type 'object'
obj= df.select_dtypes(include=['object']).copy()


# In[14]:


#We can also use the “.idxmax()” method to calculate for us the most common type automatically
#lambda : apply to both rows and columns of a dataframe. 
obj.apply(lambda x: pd.value_counts(x).idxmax(x)[0])


# In[15]:


for column in obj.columns:
     print("\n" + column)
     print(obj[column].value_counts())


# In[16]:


##Replace missing data by frequency
dictionnaire = {'ADDRTYPE': 'Block', "LOCATION": 'BATTERY ST TUNNEL NB BETWEEN ALASKAN WY VI NB AND AURORA AVE N', "SEVERITYDESC": 'Property Damage Only Collision', 'COLLISIONTYPE': 'Parked Car', "INCDATE": '2006/11/02 00:00:00+00', 'INCDTTM': '11/2/2006',  'JUNCTIONTYPE': "Mid-Block (not related to intersection)","SDOT_COLDESC": 'MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END AT ANGLE', 'UNDERINFL': 'N', "WEATHER": "Clear", "ROADCOND" : "Dry", 'ST_COLCODE': '32', "ST_COLDESC": 'One parked--one moving', 'HITPARKEDCAR':'N',"LIGHTCOND" :"Daylight"}

for k, v in dictionnaire.items():
  
    obj[k].replace(np.nan, v, inplace=True)


# In[17]:


#to convert object columns to numérical 
df_obj_facto = obj.apply(lambda x: pd.factorize(x)[0])


# In[18]:


#use concat function to combine two dataframe
df_cleaned= pd.concat([num, df_obj_facto],axis=1)


# In[19]:


#to save dataframe to CSV
file_name='clean_df.csv'
df_cleaned.to_csv(file_name)


# In[20]:


#to check if there are missing values
df_cleaned.isna().sum()


# In[21]:


#In explanatory data analysing step, we analyzed how different features related to the severity of an accident
#by calculating the correlation coefficient.


# In[22]:


sns.heatmap(abs(df_cleaned.corr()))


# In[23]:


#Train/Test Split
#We will use 30% of our data for testing and 70% for training.
x = df_cleaned
y = df_cleaned['SEVERITYCODE']


# In[24]:


x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3,random_state=1100)
print ('Train set:', x_train.shape, y_train.shape)
print ('Test set:', x_test.shape, y_test.shape)


# In[25]:


#Building the decision tree model 
tree= DTC(criterion='entropy',max_depth=1).fit(x_train,y_train)
yhatdtc=tree.predict(x_test)
print( 'The predicted result is : ' , yhatdtc)


# In[26]:


#Building the KNN model
score={}
for k in range(1,50):
    neigh = KNN(n_neighbors = k).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    train_score= skl.metrics.accuracy_score(y_train,neigh.predict(x_train))
    test_score= skl.metrics.accuracy_score(y_test,yhat)
    score[k]= test_score
    print(k, 'train : ', train_score , ' test : ', test_score)
best_score=max(score.values())
#finding the k : 
for i,j in score.items():
    if j == best_score: 
        best_k=i
        break
print('the best k is', best_k)


# In[27]:


#Building the LR model 
LR1 = LR(C=0.01, solver='liblinear').fit(x_train,y_train)
LR1


# In[28]:


yhat1 = LR1.predict(x_test)
yhat1


# In[29]:


#Building SVM 
##Let’s just use the default, RBF (Radial Basis
clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)


# In[30]:


yhat2 = clf.predict(x_test)
yhat2


# In[31]:


#Evaluation 
#Now we will check the accuracy of our models
#Let's start with Decision tree
#jaccard_similarity_score and F1-score


# In[32]:


Jaccard_tree=jaccard_score(y_test, yhatdtc)
f1=f1_score(y_test, yhatdtc, average='binary')
print('Jaccard:',Jaccard_tree)
print('F1-Score: ' ,f1)


# In[33]:


#Model is most accurate with a max depth of 1


# In[34]:


#we will evaluate KNN model 
#Jaccard_index and F1-score
Jaccard_KNN=jaccard_score(y_test, yhat)
f1=f1_score(y_test, yhat, average='binary')
print('Jaccard:',Jaccard_KNN)
print('F1-Score: ' ,f1)


# In[35]:


#Model is most accurate when k is 1


# In[36]:


#we will evaluate Logistic Regression model
#Jaccard_index, F1-score and logloss
Jaccard_Regression=jaccard_score(y_test, yhat1)
f1=f1_score(y_test, yhat1, average='binary')
print('Jaccard:',Jaccard_Regression)
print('F1-Score: ' ,f1)


# In[37]:


yhat_prob = LR1.predict_proba(x_test)
yhat_prob
logloss_Regeression=log_loss(y_test, yhat_prob)
print('log_loss: ' ,logloss_Regeression)


# In[38]:


#Model is most accurate when the inverse of regularization strength parameter C=0.01


# In[39]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, x_test, y_test)  
plt.show() 


# In[48]:


#Lets try jaccard index for accuracy:
j2=jaccard_score(y_test, yhat2)
f2=f1_score(y_test, yhat2, average='weighted')
print('Jaccard:', f2)
print('F1-Score: ' ,j2)


# In[ ]:




