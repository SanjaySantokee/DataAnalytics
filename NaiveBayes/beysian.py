import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree


filename = 'wisconsin.csv'
df = pd.read_csv(filename)  #getting dataframe (panda specific data structure that looks like a table format)

#print(df.head())

df = df.drop(['Unnamed: 32'], axis = 1).copy()  #this is needed since sometimes it may create a blank column and this is to remove it

#print(df.head())


#from sklearn.preprocessing import LabelEncoder
diagnosis_encoder = LabelEncoder()

#pass in column into fit function to label the unique items (B is assigned to 0, M is assigned to 1), essientially finding the numbers
diagnosis_encoder.fit(df['diagnosis'])  

df['diagnosis'] = diagnosis_encoder.transform(df['diagnosis'])  #changes the column to numbers(changes B and M to 0 and 1 respectfully)

#to get random sets of instances by shuffeling so data is not bias
df = shuffle(df)

#print(df.head())
#print(df['diagnosis'].value_counts())


target = df['diagnosis']
inputs = df.drop(['id','diagnosis'], axis = 1).copy()

#used for training
X = inputs.values   #returns data without column headings

#used for testing
y = target.values   #returns category without column headings, which is just the diagnosis column

#print(y)

#this splits the data so that 75% is used for training and 25% is used for testing
#train_test_split does the split as 75 and 25 by default(but these values can be varied)
#it splits but the x and y are not mixed
x_train, x_test, y_train, y_test = train_test_split(X, y)

#instantiate naive bayes classifier
clf = GaussianNB()
#train
clf.fit(x_train, y_train)
#model
y_pred = clf.predict(x_test)

#print('NB ', y_pred)

print("NB: ", accuracy_score(y_test, y_pred) * 100)

#To compare with decision tree, we can write code to do so and use same test and train data heredt = DecisionTreeClassifier()
dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

#using the classifier to predict outcomes for the test set
y_pred2 = dt.predict(x_test)

#print('DT ' , y_pred2)

print("DT: ", accuracy_score(y_test, y_pred2) * 100)

#navigate to folder in terminal and run 'dot -Tpng tree.dot -o tree.png'
## Visualize
tree.export_graphviz(
dt,
out_file = "tree.dot",
rounded = True,
filled = True)



#different ways of seeing how your code performs
#how classifier performs with positive??
#positive ones are maliginant (0)

print("Precision: ", precision_score(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print("F1-Score: ", f1_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred)) #Sensitivity same as recall

TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
print("Specificity: ", TN/(TN+FP))


#K Fold

k_fold = KFold(n_splits=5)  #splits the rows in D randomly into 5 sections

accuracies=[]
nb = GaussianNB()


#goes through each fold
#1st fold(iteration) section1 is test, rest is training data....2nd fold section2 is test, rest is training data.. and so on
for train_idx, test_idx in k_fold.split(X):
    train_X, test_X = X[train_idx], X[test_idx]
    train_y, test_y = y[train_idx], y[test_idx]
    nb.fit(train_X, train_y)
    predictions = nb.predict(test_X)
    accuracy = accuracy_score(test_y, predictions)*100  #similar as before but it appends because the test and training data are changed at every loop iteration(fold)
    accuracies.append(accuracy)

print("The mean accuracy is ", np.mean(accuracies))
print(accuracies)