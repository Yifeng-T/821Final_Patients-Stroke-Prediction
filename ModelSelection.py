'''Import Packages'''
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import style




'''Read Data'''
raw_data = pd.read_csv("healthcare-dataset-stroke-data.csv")




'''Clean Data'''
#add headers
headers = ["ID", "Gender", "Age", "Hypertension", "Heart Disease", "Ever Married", "Work Type", "Residence Type", "Avg. Glucose Level", "BMI", "Smoking Status", "Stroke"]
raw_data.columns = headers

#drop NA
raw_data.dropna(axis=0, inplace=True)

#Reset Index
raw_data.reset_index(drop=True, inplace=True)

#change 0,1 with No,Yes
raw_data["Hypertension"].replace([0,1], ["No","Yes"], inplace=True)
raw_data["Heart Disease"].replace([0,1], ["No","Yes"], inplace=True)

#drop ID variable
data2 = raw_data[["Gender","Age","Hypertension","Heart Disease","Ever Married","Work Type","Residence Type","Avg. Glucose Level","BMI","Smoking Status","Stroke"]]

#create dummy tables
gender = pd.get_dummies(data2["Gender"], drop_first=True)
hypertension = pd.get_dummies(data2["Hypertension"], drop_first=True, prefix="HT")
heartdisease = pd.get_dummies(data2["Heart Disease"], drop_first=True, prefix="HD")
evermarried = pd.get_dummies(data2["Ever Married"], drop_first=True, prefix="EM")
worktype = pd.get_dummies(data2["Work Type"], drop_first=True)
residencetype = pd.get_dummies(data2["Residence Type"], drop_first=True, prefix="RT")
smokingstatus = pd.get_dummies(data2["Smoking Status"], drop_first=True, prefix="SS")

#concatenate tables
data3 = pd.concat([data2,gender,hypertension,heartdisease,evermarried,worktype,residencetype,smokingstatus], axis=1, join='outer', ignore_index=False)

#drop some variables
data3.drop(["Gender","Hypertension","Heart Disease","Ever Married","Work Type","Residence Type","Smoking Status"], axis=1, inplace=True)

#select some varibales
data4 = data3.reindex(labels=["Age","Male","HT_Yes","HD_Yes","EM_Yes","Never_worked","Private","Self-employed","children","BMI","Avg. Glucose Level","RT_Urban","SS_formerly smoked","SS_never smoked","SS_smokes","Stroke"], axis=1)





'''Data Preprocessing'''
#balance the data
print("Not Stroke: ", len(data4[data4["Stroke"]==0]))
print("Stroke: ", len(data4[data4["Stroke"]==1]))
#As we can see, the data of stroke and not stroke are highly unbalanced.
#Models will be skewed to the "No Stroke". 
#Therefore, to get an accurate prediction model, we choose to decrease the sample size to a balanced one
balance1 = data4[data4["Stroke"]==1].sample(n=209, replace=False)
balance1.reset_index(drop=True, inplace=True)
balance0 = data4[data4["Stroke"]==0].sample(n=209, replace=False)
balance0.reset_index(drop=True, inplace=True)

data5 = pd.concat([balance1, balance0], axis=0, join='outer')

print("Balanced: Not stroke = ", len(data5[data5["Stroke"]==0]))
print("Balanced: Stroke = ", len(data5[data5["Stroke"]==1]))

#Create independent and dependent varibales
X = data5[["Age","Male","HT_Yes","HD_Yes","EM_Yes","Never_worked","Private","Self-employed","children","BMI","Avg. Glucose Level","RT_Urban","SS_formerly smoked","SS_never smoked","SS_smokes"]]
y = data5["Stroke"]


#split to train data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



'''Model Selection'''
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#create a model diagonose function
def model_diagnosis(y_test, y_pred):
    print("Accuracy = ", accuracy_score(y_test, y_pred))
    print("Precision = ", precision_score(y_test, y_pred))
    print("Recall = ", recall_score(y_test, y_pred))
    print("F1 = ", f1_score(y_test, y_pred))
    pass


#Model 1. Logistic Regression
Model1 = LogisticRegression()
Model1 = Model1.fit(X_train,y_train)
y_pred = Model1.predict(X_test)
print("The report of logistic regression is")
model_diagnosis(y_test, y_pred)

#Model 2. Random Forest
Model2 = RandomForestClassifier()
Model2 = Model2.fit(X_train,y_train)
y_pred = Model2.predict(X_test)
print("The report of logistic regression is")
model_diagnosis(y_test, y_pred)

#Model 3. Decision Tree
Model3 = DecisionTreeClassifier()
Model3 = Model3.fit(X_train,y_train)
y_pred = Model3.predict(X_test)
print("The report of logistic regression is")
model_diagnosis(y_test, y_pred)

#Logistic Regression have the highest accuracy score and F1 score, so we choose to use Logistic Regression model


'''Model Saving'''
import joblib
joblib.dump(Model1, "Model1.pkl")