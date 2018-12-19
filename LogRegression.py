# -*- coding: utf-8 -*-
"""
@author: Max
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

folder_path = r"C:\Users\Max\Documents\R\NUS"
data =  open(os.path.join(folder_path, "bank-additional.csv"), "r")

### dropping non customer information and macroeconomic var
df = pd.read_csv(r'C:\Users\Max\Documents\R\NUS\bank-additional.csv', encoding="utf8", sep=",")
df.head(3)
df.columns
df1 = df.drop(['month','day_of_week','duration','campaign','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'], axis=1)
df1.dropna()
print(df1.shape)
print(list(df1.columns))
df1.head(3)

### data exploration

#combining education for better analysis
df1['education'].unique()
df1['education']=np.where(df1['education'] =='basic.9y', 'Basic', df1['education'])
df1['education']=np.where(df1['education'] =='basic.6y', 'Basic', df1['education'])
df1['education']=np.where(df1['education'] =='basic.4y', 'Basic', df1['education'])

# y dependent variable comparison
df1['y'].value_counts()
sns.countplot(x='y', data=df1, palette='hls')
plt.show() #imbalanced data

#looking at purchase group comparison
df1.groupby('y').mean()
#to lower pdays for no purchase, to improve recall
#call should be more for no purchase but not the case
df1.groupby('education').mean() 
#people with basic education are contacted least but may need the info more
df1.groupby('job').mean()
#blue collar worker contacted least
df1.groupby('marital').mean()
#single contact most but divorced least (older)

##visualization
#by education
table=pd.crosstab(df1.education,df1.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.show()

#by job title
pd.crosstab(df1.job,df1.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.show()

#by marital status
table=pd.crosstab(df1.marital,df1.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.show()

#convert categorical to continous var
cat_var = ['job','marital','education','default','housing','loan','contact','poutcome']
for var in cat_var:
    cat_list = 'dum' + '_' + var
    cat_list = pd.get_dummies(df1[var], prefix=var)
    df2 = df1.join(cat_list)
    df1 = df2
    
print(list(df1.columns))
df2 = df1.drop(['job','job_admin.','marital','education','default','housing','loan','contact','poutcome'], axis=1)
print(list(df2.columns))
df2.dropna()

# SMOTE for imbalanced data - y
# from imblearn.over_sampling import SMOTE

x = df2.loc[:,df2.columns != 'y']
y = df2.loc[:,df2.columns == 'y']

# iterative approach to drop variable which has high p-value, backward elimination
import statsmodels.api as sm
logit_model = sm.Logit(y,x)
result = logit_model.fit()
print(result.summary2())

print(list(x.columns))
x1 = x.drop(['job_housemaid','job_self-employed','job_unemployed','job_unknown','job_management','default_no','default_unknown','default_yes','default_no','poutcome_nonexistent','poutcome_success'], axis=1)
logit_model = sm.Logit(y,x1)
result = logit_model.fit()
print(result.summary2())

x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2)
model = LogisticRegression()
model.fit(x_train, y_train)

# predictive value
y_pred = model.predict(x_test)
print('Accuracy of logistic regression classifier on test set: ' + str(model.score(x_test, y_test)))

# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# f-score - weighting between precision and recall(sensitivity)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) #near 1 is better, f-score weight precision more than recall 

# ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, model.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()
