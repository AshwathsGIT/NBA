# SVM
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

# Read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv.csv')

# Column names
original_headers = list(nba.columns.values)

# "Position (pos)" is the class attribute we are predicting. 
class_column = 'Pos'

# Features or attributes input for training the classifier
feature_columns = [ 'FG%', '2P%','3P%','TRB', \
     'AST', 'STL', 'BLK','PS/G','FT']

# Using column selection to split the data into features and class. 
nba_feature = nba[feature_columns]
nba_class = nba[class_column]

# Normalization using MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1), copy=True)

# Splitting into train and test sets
train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25,random_state =0)
    
# Standardizing the features using scaler
nba_feature = scaler.fit_transform(nba_feature)
training_accuracy = []
test_accuracy = []

# Training the classifier using train features
linearsvm = LinearSVC(random_state=0,C=1.0).fit(train_feature, train_class)

# Prediction of the test features
prediction = linearsvm.predict(test_feature)
print("Accuracy:")
print("Test set score: {:.2f}".format(linearsvm.score(test_feature, test_class)))

# Confusion matrix
print("\nConfusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

# Cross-validation scores and it's average using cross_val_score function
scores = cross_val_score(linearsvm, nba_feature, nba_class, cv=10)
print("\nCross-validation scores: \n{}".format(scores))
print("\nAverage cross-validation score: {:.2f}".format(scores.mean()))


