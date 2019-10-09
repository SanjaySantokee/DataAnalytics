import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier


filename = 'wisconsin.csv'
df = pd.read_csv(filename)  #getting dataframe (panda specific data structure that looks like a table format)

print(df.head())

df = df.drop(['Unnamed: 32'], axis = 1).copy()  #this is needed since sometimes it may create a blank column and this is to remove it

print(df.head())

print(df['diagnosis'].value_counts())   #this does a count of unique values

print(df.dtypes)    #prints types of data in D

#printing all labels of columns
#for d in df.columns:
#    print(d)

#from sklearn.preprocessing import LabelEncoder
diagnosis_encoder = LabelEncoder()
diagnosis_encoder.fit(df['diagnosis'])  #pass in column into fit function to label the unique items (B is assigned to 0, M is assigned to 1), essientially finding the numbers
df['diagnosis'] = diagnosis_encoder.transform(df['diagnosis'])  #changes the column to numbers(changes B and M to 0 and 1 respectfully)
#print(df.head())
print(df['diagnosis'].value_counts())


target = df['diagnosis']
inputs = df.drop(['id','diagnosis'], axis = 1).copy()
print(inputs.head())

#used for training
X = inputs.values   #returns data without column headings

#used for testing
y = target.values   #returns category without column headings, which is just the diagnosis column

#this splits the data so that 75% is used for training and 25% is used for testing
#train_test_split does the split as 75 and 25 by default(but these values can be varied)
#it splits but the x and y are not mixed
x_train, x_test, y_train, y_test = train_test_split(X, y)

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

#using the classifier to predict outcomes for the test set
y_pred = dt.predict(x_test)

print(y_pred)

print(accuracy_score(y_test, y_pred) * 100)

print(confusion_matrix(y_test, y_pred))
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

#TN how many neg was correct but were correct
#FP how many pos cases said correct but it was wrong
#FN how many neg cases said wrong but was true
#TP how many postitive cases said correct was correct ( how many M cases were correct)

k_fold = KFold(n_splits=5)  #splits the rows in D randomly into 5 sections

accuracies=[]
dt2 = DecisionTreeClassifier()

#goes through each fold
#1st fold(iteration) section1 is test, rest is training data....2nd fold section2 is test, rest is training data.. and so on
for train_idx, test_idx in k_fold.split(X):
    train_X, test_X = X[train_idx], X[test_idx]
    train_y, test_y = y[train_idx], y[test_idx]
    dt2.fit(train_X, train_y)
    predictions = dt2.predict(test_X)
    accuracy = accuracy_score(test_y, predictions)*100  #similar as before but it appends because the test and training data are changed at every loop iteration(fold)
    accuracies.append(accuracy)

print("The mean accuracy is ", np.mean(accuracies))
print(accuracies)