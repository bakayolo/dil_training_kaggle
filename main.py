#!/usr/bin/python

import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

train_url = "./train.csv"
train = pd.read_csv(train_url)
# print(train.head())
# print(train.describe())

test_url = "./test.csv"
test = pd.read_csv(test_url)
# print(test.head())
# print(test.describe())

print "------------------ DATAFRAMES ------------------"
print
test_one = test.copy()
test_one["Survived"] = 0
test_one["Survived"][test_one["Sex"] == 'female'] = 1
print("TEST - Survived/Sex")
print(test_one["Survived"].value_counts(normalize = True))
print

print "--------------------- TREE ---------------------"
print
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Child"] = float('NaN')
train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"] = train["Embarked"].fillna('S')
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Child"] = float('NaN')
test["Child"][test["Age"] < 18] = 1
test["Child"][test["Age"] >= 18] = 0
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"] = test["Embarked"].fillna('S')
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test["Pclass"] = test["Pclass"].fillna(test["Pclass"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

target = train['Survived'].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
my_tree_one = tree.DecisionTreeClassifier()
#my_tree_one = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_one = my_tree_one.fit(features_one, target)
# print(my_tree_one.feature_importances_)
# print
print("TEST - Survived/Pbclass,Sex,Age,Fare")
print
print "Score"
print(my_tree_one.score(features_one, target))
print
test_two = test.copy()
test_features_one = test_two[["Pclass", "Sex", "Age", "Fare"]].values
my_prediction = my_tree_one.predict(test_features_one)
PassengerId = np.array(test_two["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print "Percentage Survivors"
print(my_solution["Survived"].value_counts(normalize = True))
print
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

train_two = train.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)
print("TEST - Survived/Pbclass,Sex,Age,Fare,SibSp,Parch,family_size")
print
print "Score"
print(my_tree_three.score(features_three, target))
print
test_three = test.copy()
test_three["family_size"] = test_three["SibSp"] + test_three["Parch"] + 1
test_features_three = test_three[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values
my_prediction_2 = my_tree_three.predict(test_features_three)
PassengerId_2 = np.array(test_three["PassengerId"]).astype(int)
my_solution_2 = pd.DataFrame(my_prediction_2, PassengerId_2, columns = ["Survived"])
print "Percentage Survivors"
print(my_solution_2["Survived"].value_counts(normalize = True))
print
my_solution_2.to_csv("my_solution_two.csv", index_label = ["PassengerId"])

print "----------------- RANDOMFOREST -----------------"
print
print("TEST - Survived/Pbclass,Sex,Age,Fare,SibSp,Parch,Embarked")
print
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)
print "Score"
print(my_forest.score(features_forest, target))
print
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
PassengerId_3 = np.array(test_three["PassengerId"]).astype(int)
my_solution_3 = pd.DataFrame(pred_forest, PassengerId_3, columns = ["Survived"])
print "Percentage Survivors"
print(my_solution_3["Survived"].value_counts(normalize = True))
print
my_solution_3.to_csv("my_solution_three.csv", index_label = ["PassengerId"])