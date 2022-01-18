# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 19:37:02 2021

@author: Prashant Kumar
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np


df = pd.read_csv('SteelPlateFaults-2class.csv')
print(df.shape)



df_0 = df[df["Class"]==0]


[X_train_0, X_test_0,
  X_label_train_0,
  X_label_test_0] = train_test_split(df_0,
                                   df_0['Class'], test_size=0.3,
                                   random_state=42, shuffle=True)


                                     

df_1 = df[df["Class"]==1]

[X_train_1, X_test_1,
  X_label_train_1,
  X_label_test_1] = train_test_split(df_1,
                                   df_1['Class'], test_size=0.3,
                                   random_state=42, shuffle=True)                            
## Joining the training of class 0 and 1
# and testing data of class 0 and 1
[X_train, X_test, X_label_train,
 X_label_test] = [X_train_0.append(X_train_1),
                  X_test_0.append(X_test_1),
                  X_label_train_0.append(X_label_train_1),
                  X_label_test_0.append(X_label_test_1)]
                  
# Saving the training and testing data in CSV files
X_train.to_csv('SteelPlateFaults-2class-train.csv', index=False)
X_test.to_csv('SteelPlateFaults-2class-test.csv', index=False)

# Function to classify given dataset using the KNN Classifier
def knn_classifier(x_train, x_test, x_label_test, x_label_train):
    for i in range(1, 6, 2):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, x_label_train)
        
        # Printing the Accuracies and Confusion matrix for each K
        print(' K = {:}'.format(i))
        print(' Accuracy : {:.3f}'
             .format(knn.score(x_test, x_label_test)))
        print(' Confusion Matrix :\n')
        print(confusion_matrix(x_label_test, knn.predict(x_test)),'\n')
        if(i == 5):
            return knn.score(x_test, x_label_test)
    return 0

# Performing the KNN classification technique
print('(1)\n')
best_accuracy_knn = knn_classifier(X_train[list(df)[:-1]], X_test[list(df)[:-1]], X_label_test, X_label_train)

#______________________________________________

min_max_scaler = MinMaxScaler()
X_train_normalised = min_max_scaler.fit_transform(X_train)
X_train_normalised = pd.DataFrame(X_train_normalised)
X_train_normalised.rename(columns={i: list(df)[i] for i in range(len(list(df)))}, inplace=True)
X_train_normalised.to_csv('SteelPlateFaults-2class-train-normalised.csv', index=False)

print(X_train_normalised)

# Dropping the tuples having out of bound values
# (As compared with the min. and max. from training data)
drop_tuple_indexes = set()
for i in range(len(list(df))):
    for j in X_test.index:
        if(X_test[list(X_test)[i]][j] < min_max_scaler.data_min_[i]):
            drop_tuple_indexes.add(j)
        if(X_test[list(X_test)[i]][j] > min_max_scaler.data_max_[i]):
            drop_tuple_indexes.add(j)

X_test_normalised = min_max_scaler.fit_transform(X_test.drop(list(drop_tuple_indexes), axis=0))
X_test_normalised = pd.DataFrame(X_test_normalised)
X_test_normalised.rename(columns={i: list(df)[i] for i in range(len(list(df)))}, inplace=True)
X_test_normalised.to_csv('SteelPlateFaults-2class-test-normalised.csv', index=False)

# Appying the KNN classfication technique
print('\n(2)\n')
best_accuracy_knn_normalised = knn_classifier(X_train_normalised[list(df)[:-1]],
                                              X_test_normalised[list(df)[:-1]],
                                              X_test_normalised['Class'],
                                              X_train_normalised['Class'])





#____________________________________________________
# Building a Bayes Classifer with given training data
# and testing on the testing data

# Dimension of the training and testing data
d = 27

# Function to calculate likelihood of a class for given test sample
def likelihood(x, mean, cov_matrix):
    x = np.array(x)
    mean = np.array(mean)
    cov_matrix = np.array(cov_matrix)
    val = (1/(((2*np.pi)**(d/2))*(np.linalg.det(cov_matrix)**0.5)))
    val *= np.exp(-0.5*np.dot(np.dot((x - mean).T, np.linalg.inv(cov_matrix)), (x - mean)))
    return val

# Priors of each class from the training data
prior_0 = list(X_train['Class']).count(0)/len(X_train['Class'])
prior_1 = list(X_train['Class']).count(1)/len(X_train['Class'])



df_0 = df_0[list(df_0)[:-1]]
df_1 = df_1[list(df_1)[:-1]]




# Mean matrices for each class
mean_0 = df_0.mean().to_numpy()
mean_1 = df_1.mean().to_numpy()


# Covariance matrices for each class
cov_matrix_0 = df_0.cov().to_numpy()
cov_matrix_1 = df_1.cov().to_numpy()



# Predicted test labels
X_label_test_predicted = []
for i in np.array(X_test[list(X_test)[:-1]]):
    likl_0 = likelihood(i, mean_0, cov_matrix_0)
    likl_1 = likelihood(i, mean_1, cov_matrix_1)
    posterior_0 = (likl_0 * prior_0)/ (likl_0 * prior_0 + likl_1 * prior_1)
    posterior_1 = (likl_1 * prior_1)/ (likl_0 * prior_0 + likl_1 * prior_1)
    if(posterior_0 > posterior_1):
        X_label_test_predicted.append(0)
    else:
        X_label_test_predicted.append(1)

print('\nBayes Classifier\n\n Confusion Matrix:')
print(confusion_matrix(X_label_test, X_label_test_predicted))
print('\n Accuracy:', accuracy_score(X_label_test, X_label_test_predicted))


# Tablulating the best results of each classifier
res = pd.DataFrame({'KNN':best_accuracy_knn,
                    'KNN Normalised':best_accuracy_knn_normalised,
                    'Bayes':accuracy_score(X_label_test, X_label_test_predicted)}.items(), columns=['Classifier', 'Accuracy'])
print('\n',res)