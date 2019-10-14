import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
plt.style.use('ggplot')


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


#print(df.head())
#print(df['diagnosis'].value_counts())


X = df.drop(['id','diagnosis'], axis = 1).copy().values
y = df['diagnosis'].values

#Normalizing- feature scaling

#fit is creating Normalizer for X
#transform is actually changing the data in X
transformer = Normalizer().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)    # Its using 0.4 since 40% is test data and 60% is training data

#Setup arrays to store training and test accuracies

neighbours = np.arange(1,9)    # creates 1 - 8

# two empty arrays to store accuracies of train and test data
train_accuracy = np.empty(len(neighbours))
test_accuracy = np.empty(len(neighbours))


# This starts at 1, goes to 8. i,k where is is the index of where k is
#enumerate gives index
for i,k in enumerate(neighbours):
    # Setup a knn classifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model 
    knn.fit(X_train, y_train)
    
    #Compute accuracy on training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on training set
    test_accuracy[i] = knn.score(X_test, y_test)    

plt.title('KNN Varying number of neighbours')
plt.plot(neighbours, test_accuracy, label='Testing Accuracy')
plt.plot(neighbours, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of neighbours')
plt.ylabel('Accuracy')
plt.show()


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
y_pred = knn.predict(X_test)

#TN FP FN TP
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred))