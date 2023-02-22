import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from data_preprocessing import create_data, encode_features, modify_cols
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import csv

# creating the train, test, and validation data
test_df, train_df = create_data()
y_train = train_df['income']
X_train = train_df.drop('income', axis=1)
modify_cols(X_train)
modify_cols(test_df)
X_train, X_test = encode_features(X_train, test_df)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


def random_forest(X_trains, y_trains, X):
    rf = RandomForestClassifier(max_depth=20, random_state=0)
    rf.fit(X_trains, y_trains)
    y_predicted = rf.predict(X)
    return y_predicted


def decision_tree(X_trains, y_trains, X):
    rf = RandomForestClassifier(max_depth=11, random_state=0)
    rf.fit(X_trains, y_trains)
    y_predicted = rf.predict(X)
    return y_predicted


def knn(X_trains, y_trains, X):
    rf = KNeighborsClassifier(n_neighbors=30)
    rf.fit(X_trains, y_trains)
    y_predicted = rf.predict(X)
    return y_predicted


def svm(X_trains, y_trains, X):
    rf = LinearSVC(C=10.0)
    rf.fit(X_trains, y_trains)
    y_predicted = rf.predict(X)
    return y_predicted


def lr(X_trains, y_trains, X):
    rf = LogisticRegression(C=1000.0)
    rf.fit(X_trains, y_trains)
    y_predicted = rf.predict(X)
    return y_predicted


def ada_boost(X_trains, y_trains, X):
    rf = AdaBoostClassifier(n_estimators=27)
    rf.fit(X_trains, y_trains)
    y_predicted = rf.predict(X)
    return y_predicted


def random_forest_param():  # best = 20
    predicts = {}
    for i in range(2, 31):
        rf = RandomForestClassifier(max_depth=i, random_state=0)
        rf.fit(X_train, y_train)
        y_predicted = rf.predict(X_validation)
        # predicts[i] = f1_score(y_validation, y_predicted, pos_label=" >50K")
        predicts[i] = accuracy_score(y_validation, y_predicted)
    print('Random forest Accuracy Score as max depth parameter is increased')
    # print('Random forest f1 Score as max depth parameter is increased')
    print('Value of Max Depth     | accuracy-score')
    for comp, score in predicts.items():
        print(f'{comp}\t\t\t{score}')


def decision_tree_param():  # best = 11
    predicts = {}
    for i in range(2, 31):
        rf = DecisionTreeClassifier(max_depth=i, random_state=0)
        rf.fit(X_train, y_train)
        y_predicted = rf.predict(X_validation)
        predicts[i] = f1_score(y_validation, y_predicted, pos_label=" >50K")
        # predicts[i] = accuracy_score(y_validation, y_predicted)
    print('Decision Tree f1 Score as max depth param increases')
    print('Value of Max Depth     | f1-score')
    for comp, score in predicts.items():
        print(f'{comp}\t\t\t{score}')
    print(str(max(predicts, key=predicts.get)))


def knn_param():  # best = 30
    predicts = {}
    for i in range(2, 31):
        print(i)
        rf = KNeighborsClassifier(n_neighbors=i)
        rf.fit(X_train, y_train)
        y_predicted = rf.predict(X_validation)
        predicts[i] = f1_score(y_validation, y_predicted, pos_label=" >50K")
        # predicts[i] = accuracy_score(y_validation, y_predicted)
    # print('KNN Accuracy Score as number of neighbors increases')
    print('KNN f1 Score as number of neighbors increases')
    print('Number of neighbors    | f1-score')
    for comp, score in predicts.items():
        print(f'{comp}\t\t\t{score}')
    print(str(max(predicts, key=predicts.get)))


def svm_param():
    predicts = {}
    c_param = .00001
    for j in range(1, 10):
        svm = LinearSVC(class_weight='balanced', random_state=1, C=c_param)
        svm.fit(X_train, y_train)
        predicted = svm.predict(X_validation)
        # predicts[c_param] = f1_score(y_validation, predicted, pos_label=" >50K")
        predicts[c_param] = accuracy_score(y_validation, predicted)
        c_param *= 10
    # print('Value of C in SVM     | f1-score')
    print('Value of C in SVM     | accuracy-score')
    for comp, score in predicts.items():
        print(f'{comp}\t\t\t{score}')
    # print("Max f1-score when C = " + str(max(predicts, key=predicts.get)))
    print("Max accuracy-score when C = " + str(max(predicts, key=predicts.get)))


def lr_param():
    predicts = {}
    c_param = .00001
    for j in range(1, 10):
        lr = LogisticRegression(class_weight='balanced', random_state=1, C=c_param)
        lr.fit(X_train, y_train)
        predicted = lr.predict(X_validation)
        # predicts[c_param] = f1_score(y_validation, predicted, pos_label=' >50K')
        predicts[c_param] = accuracy_score(y_validation, predicted)
        c_param *= 10

    print('Value of C in LR     | accuracy-score')
    for comp, score in predicts.items():
        print(f'{comp}\t\t\t{score}')
    print("Max accuracy-score when C = " + str(max(predicts, key=predicts.get)))


def ada_boost_param():
    predicts = {}
    for j in range(1, 30):
        print(j)
        ada = AdaBoostClassifier(n_estimators=j, random_state=1)
        ada.fit(X_train, y_train)
        predicted = ada.predict(X_validation)
        predicts[j] = f1_score(y_validation, predicted, pos_label=' >50K')
        # predicts[j] = accuracy_score(y_validation, predicted)

    print('Value of n_estimators in AdaBoost     | f1-score')
    for comp, score in predicts.items():
        print(f'{comp}\t\t\t\t\t{score}')
    print("Max f1-score when C = " + str(max(predicts, key=predicts.get)))

# ada_boost_param()
# lr_param()
# svm_param()
# random_forest_param()
# decision_tree_param()
# knn_param()
