#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Exam 2020
# Title: Credit Card Fraud Detection Using a Machine Learning Approach <br>
# Student numbers: 133348, 116853, 133816, 133802
# 
# 

# ---

# ## EDA
# 

# #### Prerequisites

# Installing relevant packages

# In[1]:


# package to plot a heatmap of the correlation features with improves visability
#!pip install heatmapz


# Importing all relevant packages

# In[2]:


# General imports
import pandas as pd
import numpy as np
import time
get_ipython().run_line_magic('matplotlib', 'inline')
seed = 42
np.random.seed(seed)

# EDA imports
import seaborn as sns
from heatmap import heatmap, corrplot
import matplotlib.pyplot as plt

# Setting the style to seaborn
sns.set()
sns.set_style({'axes.facecolor': "#f1f1f1"})

# Preprocessing imports
import math
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline, make_pipeline

# Classification Imports
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


import sys
sys.path.append('./')
from evaluation import evaluate_classifier
import simple_ann


# Reading in the csv-file

# In[3]:


df = pd.read_csv("./creditcard.csv")


# #### Data Exploration

# In[4]:


df.head(5)


# In[5]:


df.describe()


# $\rightarrow$ At least one transactions has an transaction amount of zero

# Counting the zero amount transactions

# In[6]:


len(df[df["Amount"] == 0.0])


# Number of zero amount in fraudulent transactions

# In[7]:


len(df[(df["Amount"] == 0) & (df["Class"] == 1)])


# Drop the fraudulent transactions with a transaction value of 0.0 (27 values). 

# In[8]:


df.drop(list(df[(df["Amount"] == 0) & (df["Class"] == 1)].index), axis=0, inplace=True)


# In[9]:


len(df[df["Amount"] == 0.0])


# In[10]:


len(df[(df["Amount"] == 0) & (df["Class"] == 1)])


# Number of transactions with amount below zero

# In[11]:


len(df[df["Amount"] < 0])


# Counting the fraudulent and non fraudulent data and calculating their share in %

# In[12]:


df["Class"].value_counts()


# In[13]:


df["Class"].value_counts()/len(df["Class"])*100


# Total hours of data collection

# In[14]:


round(max(df["Time"])/(60*60), 1)


# Creating the correlation plot

# In[15]:


plt.figure(figsize=(14,14))
corrplot(df.corr())
plt.grid(False)


# Looking at the distribution only at the non-fraudulent payment amounts

# In[16]:


fig, ax = plt.subplots(1, 2, figsize=(14,4))
sns.distplot(df.Amount[df["Class"] == 0], bins=100, kde=False,
             hist_kws={"color": "#3f8094","linewidth": 0.4, "alpha": 1}, 
             ax=ax[0])
sns.distplot(df.Amount[df["Amount"] <= 250], bins=100, kde=False, 
             hist_kws={"color": "#3f8094", "linewidth": 0.4, "alpha": 1},
             ax=ax[1])


# Looking at the distribution only at the fraudulent payment amounts

# In[17]:


fig, ax = plt.subplots(1, 2, figsize=(14,4))
sns.distplot(df.Amount[df["Class"] == 1], bins=100, kde=False,
             hist_kws={"color": "#3f8094","linewidth": 0.4, "alpha": 1}, 
             ax=ax[0])
sns.distplot(df.Amount[(df["Amount"] <= 250) & (df["Class"] == 1)], bins=100, kde=False, 
             hist_kws={"color": "#3f8094", "linewidth": 0.4, "alpha": 1},
             ax=ax[1])


# Getting the maximum value of a fraudulent transaction

# In[84]:


max(df.Amount[df["Class"] == 1])


# Total value of fraudulent transactions

# In[85]:


sum(df.Amount[df["Class"] == 1])


# Average value of fraudulent transactions

# In[86]:


df.Amount[df["Class"] == 1].mean()


# Median value of fraudulent transactions

# In[87]:


df.Amount[df["Class"] == 1].median()


# Total value of non-fraudulent transactions

# In[88]:


sum(df.Amount[df["Class"] == 0])


# Median value of non-fraudulent transactions

# In[89]:


df.Amount[df["Class"] == 0].median()


# Average amount of non-fraudulent transactions

# In[90]:


df.Amount[df["Class"] == 0].mean()


# Looking at the distribution of the number of transactions over time

# In[42]:


plt.figure()
sns.distplot(df.Time, bins=48,
             hist_kws={"color": "#3f8094","linewidth": 0.4, "alpha": 1}, 
             kde_kws={"color": "#c4563a"})
plt.show()


# In[7]:


df_ts = pd.DataFrame(df.groupby(["Time"]).size(), columns=["Count"])


# In[8]:


df_ts.head()


# Splitting the variables in batches for boxplots

# In[45]:


df_v_vars1 = df.iloc[:,1:8]
df_v_vars1["Class"] = df["Class"]
df_v_vars1 = pd.melt(df_v_vars1, "Class", var_name="variables", value_name="values")

df_v_vars2 = df.iloc[:,8:15]
df_v_vars2["Class"] = df["Class"]
df_v_vars2 = pd.melt(df_v_vars2, "Class", var_name="variables", value_name="values")

df_v_vars3 = df.iloc[:,15:22]
df_v_vars3["Class"] = df["Class"]
df_v_vars3 = pd.melt(df_v_vars3, "Class", var_name="variables", value_name="values")

df_v_vars4 = df.iloc[:,22:29]
df_v_vars4["Class"] = df["Class"]
df_v_vars4 = pd.melt(df_v_vars4, "Class", var_name="variables", value_name="values")


# In[46]:


df_v_vars1.head()


# In[47]:


colors = {0: "#3f8094", 1: "#c4563a"}
fig, ax = plt.subplots(4, 1, figsize=(16,16))
sns.boxplot(x="variables", y="values",hue="Class", data=df_v_vars1, palette=colors, ax=ax[0])
ax[0].set_facecolor("#f1f1f1")
ax[0].legend(title="Class", frameon=False)
sns.boxplot(x="variables", y="values",hue="Class", data=df_v_vars2, palette=colors, ax=ax[1])
ax[1].set_facecolor("#f1f1f1")
ax[1].legend_.remove()
sns.boxplot(x="variables", y="values",hue="Class", data=df_v_vars3, palette=colors, ax=ax[2])
ax[2].set_facecolor("#f1f1f1")
ax[2].legend_.remove()
sns.boxplot(x="variables", y="values",hue="Class", data=df_v_vars4, palette=colors, ax=ax[3])
ax[3].set_facecolor("#f1f1f1")
ax[3].legend_.remove()


# In[91]:


df_small_share = df.iloc[:,25:]
df_small_share["Class"] = df["Class"]


# In[92]:


df_small_share.head()


# In[50]:


sns.pairplot(df_small_share, hue="Class",palette=colors)
plt.show()


# ---

# ## Preprocessing

# Log on amount column and afterwards apply normalization

# In[9]:


df['Amount'] = df['Amount'].apply(lambda x: math.log(x+1))


# Drop Time Column

# In[10]:


df = df.drop("Time", axis = 1)


# In[11]:


X = df.drop("Class", axis = 1)


# In[12]:


y = df["Class"]


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state=seed, train_size = 0.75)


# In[14]:


std_sc = StandardScaler()

std_sc.fit(X_train["Amount"].values.reshape(-1, 1))
X_train["Amount"] = std_sc.transform(X_train["Amount"].values.reshape(-1, 1))
X_test["Amount"] = std_sc.transform(X_test["Amount"].values.reshape(-1, 1))


# In[15]:


X_test.shape


# In[16]:


y_test.shape


# ##### Handling unbalanced data

# Oversample minority class in train data and undersample majority class

# In[17]:


# initialize undersampler to undersample majority class
random_undersampler = RandomUnderSampler(sampling_strategy=1, random_state=seed)


# SMOTE

# In[18]:


# oversample the minority class
smote = SMOTE(sampling_strategy = 0.5, random_state=seed) #,k_neighbors=5, )

# Initialize steps of pipeline
steps_smote = [('SMOTE', smote),
               ('Rand_Undersampler', random_undersampler)]

# Initialize Pipeline
pipe_smote = Pipeline(steps_smote)

# Re-sample the data
X_train_smote, y_train_smote = pipe_smote.fit_resample(X_train, y_train)


# BorderlineSMOTE

# In[19]:


# oversample the minority class
borderline_smote = BorderlineSMOTE(sampling_strategy=0.5,k_neighbors=5,random_state=seed, kind="borderline-1")

# Initialize steps of pipeline
steps_borderline = [('SMOTE', borderline_smote),
                    ('Rand_Undersampler', random_undersampler)]

# Initialize Pipeline
pipe_borderline = Pipeline(steps=steps_borderline)

# Re-sample the data
X_train_borderline, y_train_borderline = pipe_borderline.fit_resample(X_train, y_train)


# ADASYN
# 
# * Generate minority data examples based on easse of learning

# In[20]:


# oversample the minority class
adasyn = ADASYN(sampling_strategy=0.5,random_state=seed)

# Initialize steps of pipeline
steps_adasyn = [('ADASYN', adasyn),
                ('Rand_Undersampler', random_undersampler)]

# Initialize Pipeline
pipe_adasyn = Pipeline(steps=steps_adasyn)

# Re-sample the data
X_train_adasyn, y_train_adasyn = pipe_adasyn.fit_resample(X_train, y_train)


# # Classification 

# In[21]:


def train_classifier(train, test, model, params, oversampling):
    """
    
    """
    if oversampling == "SMOTE":
        steps = steps_smote.copy()
        steps.append(("model",model))
    
    elif oversampling == "Borderline":
        steps = steps_borderline.copy()
        steps.append(("model",model))
    
    elif oversampling == "ADASYN":
        steps = steps_adasyn.copy()
        steps.append(("model",model))
        
    pipe = Pipeline(steps=steps)
    
    clf = GridSearchCV(pipe,
                            params,
                            cv = 3,
                            scoring = "roc_auc",
                            n_jobs = -1)
    clf.fit(train,test)

    print("---------------Best parameters--------------\n")
    print(clf.best_params_)
    print("\n")

    return clf


# In[106]:


C = [0.001, 0.01, 0.1, 1.0, 10]


# ## Logistic Regression

# In[107]:


params_lr = {"model__solver":['saga'],
            "model__penalty":['l1', 'l2', 'elasticnet'],
            "model__C":C,
            "model__l1_ratio":[0, 0.25, 0.5, 0.75, 1]}

log_reg = LogisticRegression(random_state=seed)


# In[108]:


def lr():
    lr_smote = train_classifier(X_train, y_train, log_reg, params_lr,"SMOTE")
    lr_borderline = train_classifier(X_train,y_train,log_reg,params_lr,"Borderline")
    lr_adasyn = train_classifier(X_train,y_train,log_reg,params_lr,"ADASYN")
    
    return {'lr_smote':lr_smote, 
           'lr_borderline':lr_borderline,
           'lr_adasyn':lr_adasyn}


# ##### Printing best classifiers found in GridSearchCV performed on server: 
# 
# ```python 
# >>> for key, model in lr_dict.items():
# ...     print(key, model.best_params_)
# ...
# lr_smote {'model_C': 0.001, 
#           'modell1_ratio': 0, 
#           'modelpenalty': 'l1', 
#           'model_solver': 'saga'}
# 
# lr_borderline {'model_C': 0.001, 
#                'modell1_ratio': 0, 
#                'modelpenalty': 'l1', 
#                'model_solver': 'saga'}
# 
# lr_adasyn {'model_C': 0.001, 
#            'modell1_ratio': 0, 
#            'modelpenalty': 'l1', 
#            'model_solver': 'saga'}
# ```

# In[109]:


lr_smote_params = {'model__C': [0.001], 
          'model__l1_ratio': [0], 
          'model__penalty': ['l1'], 
          'model__solver': ['saga']}

lr_borderline_params = {'model__C': [0.001], 
               'model__l1_ratio': [0], 
               'model__penalty': ['l1'], 
               'model__solver': ['saga']}

lr_adasyn_params = {'model__C': [0.001], 
           'model__l1_ratio': [0], 
           'model__penalty': ['l1'], 
           'model__solver': ['saga']}

lr_smote = train_classifier(X_train, y_train, log_reg, lr_smote_params ,"SMOTE")
lr_borderline = train_classifier(X_train, y_train, log_reg, lr_borderline_params ,"Borderline")
lr_adasyn = train_classifier(X_train, y_train, log_reg, lr_adasyn_params ,"ADASYN")


# In[110]:


y_test = np.array(y_test).reshape(1, -1)


# #####  Evaluation Logistic Regression + SMOTE

# In[111]:


y_lr_smote_pred = lr_smote.predict(X_test).reshape(1, -1)
evaluate_classifier(y_test, y_lr_smote_pred)


# #####  Evaluation Logistic Regression + Borderline

# In[112]:


y_lr_borderline_pred = lr_borderline.predict(X_test).reshape(1, -1)
evaluate_classifier(y_test, y_lr_borderline_pred)


# #####  Evaluation Logistic Regression + ADASYN

# In[113]:


y_lr_adasyn_pred = lr_adasyn.predict(X_test).reshape(1, -1)
evaluate_classifier(y_test, y_lr_adasyn_pred)


# ## SVM

# In[114]:


params_poly = {"model__degree":[1, 2, 3, 5],
               "model__C": C}

params_rbf = {"model__C": C,
              "model__gamma":[0.1,1,10,100]}


# In[115]:


def svm():

    start = time.time()

    svc_poly = SVC(random_state=seed, kernel="poly")
    svc_rbf = SVC(random_state =seed, kernel ="rbf")

    # Train SVMs with polynomial kernel
    svc_smote_poly = train_classifier(X_train,y_train,svc_poly,params_poly,"SMOTE")
    svc_borderline_poly = train_classifier(X_train,y_train,svc_poly,params_poly,"Borderline")
    svc_adasyn_poly = train_classifier(X_train,y_train,svc_poly,params_poly,"ADASYN")

    # Train SVMs with gaussian kernel
    svc_smote_rbf = train_classifier(X_train,y_train,svc_rbf,params_rbf,"SMOTE")
    svc_borderline_rbf = train_classifier(X_train,y_train,svc_rbf,params_rbf,"Borderline")
    svc_adasyn_rbf = train_classifier(X_train,y_train,svc_rbf,params_rbf,"ADASYN")

    end = time.time()

    duration = start - end
    print("Elapsed Time:",duration)
    
    return {'svc_smote_poly':svc_smote_poly, 
           'svc_borderline_poly':svc_borderline_poly,
           'svc_adasyn_poly':svc_adasyn_poly,
           'svc_smote_rbf':svc_smote_rbf,
           'svc_borderline_rbf':svc_borderline_rbf,
           'svc_adasyn_rbf':svc_adasyn_rbf}


# ##### Printing best classifiers found in GridSearchCV performed on server: 
# 
# ```python
# svm_smote_poly {'model__C': 0.001, 'model__degree': 5}
# 
# 
# svm_borderline_poly {'model__C': 0.001, 'model__degree': 1}
# 
# 
# svm_adasyn_poly {'model__C': 0.001, 'model__degree': 2}
# 
# 
# svm_smote_rbf {'model__C': 0.01, 'model__gamma': 0.1}
# 
# 
# svm_borderline_rbf {'model__C': 0.1, 'model__gamma': 1}
# 
# 
# svm_adasyn_rbf {'model__C': 10, 'model__gamma': 1}
# ```
# 

# In[120]:


svm_poly = SVC(random_state=seed, kernel="poly")

svm_smote_poly_params = {'model__C': [0.001], 'model__degree': [5]}


svm_borderline_poly_params = {'model__C': [0.001], 'model__degree': [1]}


svm_adasyn_poly_params = {'model__C': [0.001], 'model__degree': [2]}





svm_poly_smote = train_classifier(X_train, y_train, svm_poly, svm_smote_poly_params ,"SMOTE")
svm_poly_borderline = train_classifier(X_train, y_train, svm_poly, svm_borderline_poly_params ,"Borderline")
svm_poly_adasyn = train_classifier(X_train, y_train, svm_poly, svm_adasyn_poly_params ,"ADASYN")


# In[ ]:


print(svm_poly_smote.cv_score_)
print(svm_poly_borderline.cv_score_)
print(svm_poly_adasyn.cv_score_)


# #####  Evaluation Support Vector Machine with Polynomial Kernel + SMOTE

# In[122]:


y_svm_poly_smote_pred = svm_poly_smote.predict(X_test).reshape(1, -1)
evaluate_classifier(y_test, y_svm_poly_smote_pred)


# #####  Evaluation Support Vector Machine with Polynomial Kernel + Borderline

# In[124]:


y_svm_poly_borderline_pred = svm_poly_borderline.predict(X_test).reshape(1, -1)
evaluate_classifier(y_test, y_svm_poly_borderline_pred)


# #####  Evaluation Support Vector Machine with Polynomial Kernel + ADASYN

# In[126]:


y_svm_poly_adasyn_pred = svm_poly_adasyn.predict(X_test).reshape(1, -1)
evaluate_classifier(y_test, y_svm_poly_adasyn_pred)


# In[ ]:


svm_rbf = SVC(random_state =seed, kernel ="rbf")


svm_smote_rbf_params = {'model__C': [0.01], 'model__gamma': [0.1]}


svm_borderline_rbf_params = {'model__C': [0.1], 'model__gamma': [1]}


svm_adasyn_rbf_params = {'model__C': [10], 'model__gamma': [1]}

svm_rbf_smote = train_classifier(X_train, y_train, svm_rbf, svm_smote_rbf_params ,"SMOTE")
svm_rbf_borderline = train_classifier(X_train, y_train, svm_rbf, svm_borderline_rbf_params ,"Borderline")
svm_rbf_adasyn = train_classifier(X_train, y_train, svm_rbf, svm_adasyn_rbf_params ,"ADASYN")


# In[ ]:


print(svm_rbf_smote.cv_score_)
print(svm_rbf_borderline.cv_score_)
print(svm_rbf_adasyn.cv_score_)


# #####  Evaluation Support Vector Machine with RBF Kernel + SMOTE

# In[ ]:


y_svm_rbf_smote_pred = svm_rbf_smote.predict(X_test).reshape(1, -1)
evaluate_classifier(y_test, y_svm_rbf_smote_pred)


# #####  Evaluation Support Vector Machine with RBF Kernel + Borderline

# In[ ]:


y_svm_rbf_borderline_pred = svm_rbf_borderline.predict(X_test).reshape(1, -1)
evaluate_classifier(y_test, y_svm_rbf_borderline_pred)


# #####  Evaluation Support Vector Machine with RBF Kernel + ADASYN

# In[ ]:


y_svm_rbf_adasyn_pred = svm_rbf_adasyn.predict(X_test).reshape(1, -1)
evaluate_classifier(y_test, y_svm_rbf_adasyn_pred)


# ## Random Forest

# In[116]:


rf_params = {"model__max_depth" : [2, 5,10,15], 
             "model__n_estimators" : [50,100,200]}

dt_params = {"model__max_depth":[2,5,10, 15]}

rf = RandomForestClassifier(random_state=seed)
dt = DecisionTreeClassifier(random_state=seed)

def rf():

    rf_smote = train_classifier(X_train,y_train,rf,rf_params,"SMOTE")
    rf_borderline = train_classifier(X_train,y_train,rf,rf_params,"Borderline")
    rf_adasyn = train_classifier(X_train,y_train,rf,rf_params,"ADASYN")

    dt_smote = train_classifier(X_train,y_train,dt,dt_params,"SMOTE")
    dt_borderline = train_classifier(X_train,y_train,dt,dt_params,"Borderline")
    dt_adasyn = train_classifier(X_train,y_train,dt,dt_params,"ADASYN")

    return {'rf_smote':rf_smote, 
           'rf_borderline':rf_borderline,
           'rf_adasyn':rf_adasyn,
           'rf_smote':rf_smote,
           'rf_borderline':rf_borderline,
           'rf_adasyn':rf_adasyn}


# ##### Printing best classifiers found in GridSearchCV performed on server: 
# 
# ```python 
# >>> for key, model in rf_dict.items():
# ...     print(key, model.best_params_)
# ...
# dt_smote {'model__max_depth': 5}
# 
# dt_borderline {'model__max_depth': 5}
# 
# dt_adasyn {'model__max_depth': 5}
# 
# rf_smote {'model_max_depth': 5, 
#           'model_n_estimators': 200}
# 
# rf_borderline {'model_max_depth': 10, 
#                'model_n_estimators': 200}
# 
# rf_adasyn {'model_max_depth': 5, 
#            'model_n_estimators': 200}
# ```
# 
# As the RandomForest always outperforms the respective decision tree we do not print the decision tree evaluation metrics here.
# They were analysed on the server though and can be reproduced using the parameters provided above.

# In[66]:


rf = RandomForestClassifier(random_state=seed)

rf_smote_params = {'model__max_depth': [5], 'model__n_estimators': [200]}

rf_borderline_params = {'model__max_depth': [10], 'model__n_estimators': [200]}

rf_adasyn_params = {'model__max_depth': [5], 'model__n_estimators': [200]}

rf_smote = train_classifier(X_train, y_train, rf, rf_smote_params ,"SMOTE")
rf_borderline = train_classifier(X_train, y_train, rf, rf_borderline_params ,"Borderline")
rf_adasyn = train_classifier(X_train, y_train, rf, rf_adasyn_params ,"ADASYN")


# #####  Evaluation Random Forest + SMOTE

# In[67]:


y_rf_smote_pred = rf_smote.predict(X_test).reshape(1, -1)
evaluate_classifier(y_test, y_rf_smote_pred)


# #####  Evaluation Random Forest + Borderline

# In[68]:


y_rf_borderline_pred = rf_borderline.predict(X_test).reshape(1, -1)
evaluate_classifier(y_test, y_rf_borderline_pred)


# #####  Evaluation Random Forest + ADASYN

# In[69]:


y_rf_adasyn_pred = rf_adasyn.predict(X_test).reshape(1, -1)
evaluate_classifier(y_test, y_rf_adasyn_pred)


# ## ANN

# In[183]:


"""
testing different 4 layer architectures on borderline samples using grid search

parameter values to be tested:
l1:


""" 
def run_arc_test(X_train, y_train, X_test, y_test, params, algo_name):
    X_train_nn = np.array(X_train).T
    y_train_nn = np.array(y_train).reshape(-1,1).T
    X_test_nn = np.array(X_test).T
    y_test_nn = np.array(y_test).reshape(-1,1).T
    
    for layer_dims in params['arcs']:

        layer_dims = list(layer_dims)
        
        y_score, param_dict, costs = simple_ann.train_network(X_train_nn, y_train_nn, X_test_nn, y_test_nn, layer_dims, num_iterations=params['num_iter'], num_checkpoints=params['num_iter']/params['checkpoints'], c_plot=False, learning_rate=params['learning_rate'], learn_adjust = params['learn_adjust'], weights = params['weights'])

        _, y_score_p = simple_ann.forward_propagation(X_test_nn, param_dict[params['num_iter']], layer_dims)
        y_score = (y_score_p > 0.5).astype(float)
        
        print('\tPredictions == 1: ', y_score.sum())
        text = '%s_test_%s' %(algo_name, str(layer_dims))

        print('-------------------------------')
        print('\t\t%s\t\t'%text)
        print('-------------------------------')
        evaluate_classifier(np.array(y_test_nn).reshape(1,-1), y_score.reshape(1,-1))
        print('\n\n')


# In[195]:


ann_test_params = {'weights' : {'w_min':1, 'w_maj':1},
                    'learning_rate' : 0.01,
                    'learn_adjust' : [1000], # no effect on testing
                    'arcs' : np.array(np.meshgrid([10, 29], [10, 15], [5, 20], [1])).T.reshape(-1,4).tolist(),
                    'num_iter' : 300,
                    'checkpoints':300}

# creating dev and train set out of unsampled training data
X_train_nn, X_dev_nn, y_train_nn, y_dev_nn = train_test_split(X_train, y_train, stratify = y_train, random_state=seed, train_size = 0.8)

# sampling training data of 1 fold cv
X_train_nn_smote, y_train_nn_smote = pipe_smote.fit_resample(X_train_nn, y_train_nn)
X_train_nn_borderline, y_train_nn_borderline = pipe_borderline.fit_resample(X_train_nn, y_train_nn)
X_train_nn_adasyn, y_train_nn_adasyn = pipe_adasyn.fit_resample(X_train_nn, y_train_nn)


# In[196]:


run_arc_test(X_train_nn_borderline, y_train_nn_borderline, X_dev_nn, y_dev_nn, ann_test_params, algo_name = 'Borderline')


# In[197]:


run_arc_test(X_train_nn_adasyn, y_train_nn_adasyn, X_dev_nn, y_dev_nn, ann_test_params, algo_name = 'ADASYN')


# In[198]:


run_arc_test(X_train_nn_smote, y_train_nn_smote, X_dev_nn, y_dev_nn, ann_test_params, algo_name = 'SMOTE')


# In[200]:



ann_params_borderline = {'weights' : {'w_min':1, 'w_maj':1},
                        'learning_rate' : 0.01,
                        'learn_adjust' : [600],
                        'arcs' : np.array([29,15,20,1]).reshape(1,-1).tolist(),
                        'num_iter' : 800,
                        'checkpoints':100}

ann_params_adasyn = {'weights' : {'w_min':1, 'w_maj':1},
                    'learning_rate' : 0.01,
                    'learn_adjust' : [2200],
                    'arcs' : np.array([29,15,20,1]).reshape(1,-1).tolist(),
                    'num_iter' : 2200,
                    'checkpoints':100}

ann_params_smote = {'weights' : {'w_min':1, 'w_maj':1},
                    'learning_rate' : 0.01,
                    'learn_adjust' : [4200],
                    'arcs' : np.array([29,15,20,1]).reshape(1,-1).tolist(),
                    'num_iter' : 4200,
                    'checkpoints':100}


# #####  Evaluation ANN + Borderline

# In[25]:


run_arc_test(X_train_borderline, y_train_borderline, X_test_nn, y_test_nn, ann_params_borderline, algo_name = 'Borderline')


# #####  Evaluation ANN + ADASYN

# In[26]:


run_arc_test(X_train_adasyn, y_train_adasyn, X_test_nn, y_test_nn, ann_params_adasyn, algo_name = 'ADASYN')


# #####  Evaluation ANN + SMOTE

# In[206]:


run_arc_test(X_train_smote, y_train_smote, X_test, y_test, ann_params_smote, algo_name = 'SMOTE')


# In[ ]:




