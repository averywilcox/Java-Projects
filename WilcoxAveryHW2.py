#!/usr/bin/env python
# coding: utf-8

# In this question, you are given the **training data** and **test data**, which are derived from the 768 observations. 
# 
# * Training data: https://github.com/binbenliu/Teaching/blob/main/IntroAI/data/diabetes_train.csv
# 
# * Test data: https://github.com/binbenliu/Teaching/blob/main/IntroAI/data/diabetes_test.csv
# 
# 
# You are asked to train and test following models:
# * item Logistic regression https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# * Support vector machine (SVM) https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# 
# After you train each  model on the **training data**, you should report following classification metrics on the **test data**: (1) accuracy, (2) precision, (3) recall, and (4) F1\_score.
# 

# # (1) Read the data 
# 
# Training data: https://raw.githubusercontent.com/binbenliu/Teaching/main/IntroAI/data/diabetes_train.csv
# 
# Testing data: https://raw.githubusercontent.com/binbenliu/Teaching/main/IntroAI/data/diabetes_test.csv

# In[4]:


import pandas as pd

# read training data
train_file = "https://raw.githubusercontent.com/binbenliu/Teaching/main/IntroAI/data/diabetes_train.csv"
train_df = pd.read_csv(train_file, header='infer')

print(f'num train records: {len(train_df)}')
train_df.head()


# In[5]:


# read test data
test_file = "https://raw.githubusercontent.com/binbenliu/Teaching/main/IntroAI/data/diabetes_test.csv"
test_df = pd.read_csv(test_file, header='infer')

print(f'num test records: {len(test_df)}')
test_df.head()


# In[6]:


cols = train_df.columns
cols


# In[7]:


# train data
X_train = train_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']].values
y_train = train_df['Outcome'].values

# test data
X_test = test_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']].values
y_test = test_df['Outcome'].values


# In[8]:


X_train


# In[9]:


y_train


# # (3) Logistic regression model
# 
# ## LR: Specify and train your logistic regression model
# 
# Logistic regression with regularziation
# 
# \begin{equation}
# L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n l\left(\sigma\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b\right),  y^{(i)}\right) + \frac{\alpha}{2} \|\mathbf{w}\|^2.
# \end{equation}
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 
# **sklearn.linear_model.LogisticRegression**(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

# In[10]:


import sklearn.linear_model
clf_lr = sklearn.linear_model.LogisticRegression(penalty='l2',  C=5, fit_intercept=True)
clf_lr.fit(X_train, y_train)


# ## LR:  predict and evaluate
# 
# Make predictions and get following classfication metrics: 
# 
# * accuracy
# * precision 
# * recall
# * f1_score
#  

# In[11]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

y_pred_lr = clf_lr.predict(X_test)


print('Accuracy on test data is %.2f' % (accuracy_score(y_test, y_pred_lr)))
print('Precision on test data is %.2f' % precision_score(y_test, y_pred_lr) )
print('Recall on test data is %.2f' % recall_score(y_test, y_pred_lr) )
print('F1_score on test data is %.2f' % f1_score(y_test, y_pred_lr) )


# In[12]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf_lr, X_test, y_test, normalize='all')  


# # (3) Specify and train the SVM model
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# 
# class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
# 

# In[13]:


from sklearn.svm import SVC

clf_svm = SVC(C=0.5,kernel='linear')
clf_svm.fit(X_train, y_train)


# In[14]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

y_pred_svm = clf_svm.predict(X_test)


print('Accuracy on test data is %.2f' % (accuracy_score(y_test, y_pred_svm)))
print('Precision on test data is %.2f' % precision_score(y_test, y_pred_svm) )
print('Recall on test data is %.2f' % recall_score(y_test, y_pred_svm) )
print('F1_score on test data is %.2f' % f1_score(y_test, y_pred_svm) )


# In[15]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf_svm, X_test, y_test, normalize='all')  

