from matplotlib import axis
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
df=pd.read_csv('WA_Fn-UseC_-Accounts-Receivable.csv',usecols=['countryCode','customerID','PaperlessDate','invoiceNumber','InvoiceDate','DueDate','InvoiceAmount','Disputed','SettledDate','PaperlessBill','DaysToSettle','DaysLate'])
print(df.head())
print(df.info)
print(df.isnull().sum())
print(df.describe())
#from sklearn.preprocessing labelencording
PaperlessBill_le=LabelEncoder()
Disputed_le=LabelEncoder()
customerID_le=LabelEncoder()
PaperlessDate_le=LabelEncoder()
InvoiceDate_le=LabelEncoder()
DueDate_le=LabelEncoder()
SettledDate_le=LabelEncoder()
df['PaperlessBill']=PaperlessBill_le.fit_transform(df['PaperlessBill'].astype(str))
df['Disputed']=Disputed_le.fit_transform(df['Disputed'].astype(str))
df['customerID']=customerID_le.fit_transform(df['customerID'].astype(str))
df['PaperlessDate']=PaperlessDate_le.fit_transform(df['PaperlessDate'].astype(str))
df['InvoiceDate']=InvoiceDate_le.fit_transform(df['InvoiceDate'].astype(str))
df['DueDate']=DueDate_le.fit_transform(df['DueDate'].astype(str))
df['SettledDate']=SettledDate_le.fit_transform(df['SettledDate'].astype(str))
print(df['PaperlessBill'])
print(df['Disputed'])
print(df['customerID'])
print(df['PaperlessDate'])
print(df['InvoiceDate'])
print(df['DueDate'])
#from X and y colum define
X=df[['countryCode','customerID','PaperlessDate','invoiceNumber','InvoiceDate','DueDate','InvoiceAmount','Disputed','SettledDate','PaperlessBill','DaysToSettle']]
y=df['DaysLate']
#from sklearn standard scaler
scaler=StandardScaler()
x_scaler=scaler.fit_transform(df)
print(x_scaler)
#from sklearn train test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(y_train.shape)

#from sklearn randomforest
model8=RandomForestClassifier(n_estimators=100)
model8.fit(X_train,y_train)
y_pred=model8.predict(X_test)
acc=accuracy_score(y_pred,y_test)
cmm=confusion_matrix(y_pred,y_test)
print(acc)
print(cmm)
with open('soring_pkl','wb')as file:
    pickle.dump(model8,file)
print("this file as save")


