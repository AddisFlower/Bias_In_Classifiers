import csv

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_preprocessing import create_data, encode_features, modify_cols, improved_encode_features
from classify import random_forest, decision_tree, knn, svm, lr, ada_boost

test_df, train_df = create_data()
y_train = train_df['income']
X_train = train_df.drop('income', axis=1)
modify_cols(X_train)
modify_cols(test_df)

# set both part_3 and part_3_2 to False if you don't want to remove any features from the data set
# set part_3 to True and part_3_2 to False if you want to only remove race and sex
# set both part_3 and part_3_2 to True fi you want to remove attributes correlated to race and sex as well
part_3 = True
part_3_2 = True
if part_3:
    if part_3_2:
        X_train = X_train.drop('race', axis=1)
        X_train = X_train.drop('sex', axis=1)
        X_test = test_df.drop('race', axis=1)
        X_test = X_test.drop('sex', axis=1)
        X_train = X_train.drop('native-country', axis=1)
        X_train = X_train.drop('workclass', axis=1)
        X_train = X_train.drop('occupation', axis=1)
        X_train = X_train.drop('hours-per-week', axis=1)
        X_test = X_test.drop('native-country', axis=1)
        X_test = X_test.drop('workclass', axis=1)
        X_test = X_test.drop('occupation', axis=1)
        X_test = X_test.drop('hours-per-week', axis=1)
        X_train, X_test = improved_encode_features(X_train, X_test, 2)
    else:
        X_train = X_train.drop('race', axis=1)
        X_train = X_train.drop('sex', axis=1)
        X_test = test_df.drop('race', axis=1)
        X_test = X_test.drop('sex', axis=1)
        X_train, X_test = improved_encode_features(X_train, X_test)
else:
    X_train, X_test = encode_features(X_train, test_df)

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

predicted = random_forest(X_train, y_train, X_validation)

# uncomment the code below to see the accuracy and f1 scores for the classifier with the configurations set above
# acc = accuracy_score(y_validation, predicted)
# f1 = f1_score(y_validation, predicted, pos_label=" >50K")
# print("Accuracy Score: " + str(acc))
# print("F1 Score: " + str(f1) + "\n")

# uncomment the code below to create the prediction file using the classifier with configurations set above
# with open('3-2_prediction.csv', 'w+', newline='') as file:
#     writer = csv.writer(file)
#     labels = random_forest(X_train, y_train, X_test)
#     for i in range(len(labels)):
#         if labels[i] == " >50K":
#             writer.writerow([1])
#         else:
#             writer.writerow([0])


# calculates the average demographic disparity for race for the given classifier
def dem_disparity_race(classifier='forest'):
    if classifier == 'forest':
        labels = random_forest(X_train, y_train, X_validation)
    if classifier == 'tree':
        labels = decision_tree(X_train, y_train, X_validation)
    if classifier == 'knn':
        labels = knn(X_train, y_train, X_validation)
    if classifier == 'svm':
        labels = svm(X_train, y_train, X_validation)
    if classifier == 'lr':
        labels = lr(X_train, y_train, X_validation)
    if classifier == 'ada':
        labels = ada_boost(X_train, y_train, X_validation)
    races = train_df['race'].unique().tolist()
    race_tallies = [0, 0, 0, 0, 0]
    total_races = [0, 0, 0, 0, 0]
    for i in range(len(labels)):
        race = test_df.at[i, 'race']
        if labels[i] == " >50K":
            race_tallies[races.index(race)] += 1
        total_races[races.index(race)] += 1

    probabilities = [0, 0, 0, 0, 0]
    for i in range(len(probabilities)):
        probabilities[i] = race_tallies[i] / total_races[i]

    total_disparity = 0
    for i in range(len(probabilities)):
        disparity = 0
        for j in range(len(probabilities)):
            if i == j:
                continue
            disparity += abs(probabilities[i] - probabilities[j])
        total_disparity += disparity / (len(probabilities) - 1)
    avg_disparity = total_disparity / len(probabilities)
    return avg_disparity


# calculates the average demographic disparity for sex for the given classifier
def dem_disparity_sex(classifier='forest'):
    if classifier == 'forest':
        labels = random_forest(X_train, y_train, X_validation)
    if classifier == 'tree':
        labels = decision_tree(X_train, y_train, X_validation)
    if classifier == 'knn':
        labels = knn(X_train, y_train, X_validation)
    if classifier == 'svm':
        labels = svm(X_train, y_train, X_validation)
    if classifier == 'lr':
        labels = lr(X_train, y_train, X_validation)
    if classifier == 'ada':
        labels = ada_boost(X_train, y_train, X_validation)
    sexes = train_df['sex'].unique().tolist()
    sex_tallies = [0, 0]
    total_sexes = [0, 0]
    for i in range(len(labels)):
        sex = test_df.at[i, 'sex']
        if labels[i] == " >50K":
            sex_tallies[sexes.index(sex)] += 1
        total_sexes[sexes.index(sex)] += 1

    probabilities = [0, 0]
    for i in range(len(probabilities)):
        probabilities[i] = sex_tallies[i] / total_sexes[i]

    total_disparity = 0
    for i in range(len(probabilities)):
        disparity = 0
        for j in range(len(probabilities)):

            if i == j:
                continue
            disparity += abs(probabilities[i] - probabilities[j])
        total_disparity += disparity / (len(probabilities) - 1)
    avg_disparity = total_disparity / len(probabilities)
    return avg_disparity


# calculates the average equality of opportunity disparity and average equality of odds for sex for the given classifier
def equality_sex(classifier='forest'):
    if classifier == 'forest':
        labels = random_forest(X_train, y_train, X_validation)
    if classifier == 'tree':
        labels = decision_tree(X_train, y_train, X_validation)
    if classifier == 'knn':
        labels = knn(X_train, y_train, X_validation)
    if classifier == 'svm':
        labels = svm(X_train, y_train, X_validation)
    if classifier == 'lr':
        labels = lr(X_train, y_train, X_validation)
    if classifier == 'ada':
        labels = ada_boost(X_train, y_train, X_validation)
    actual_labels = []
    for index, value in y_validation.items():
        actual_labels.append(value)
    sexes = train_df['sex'].unique().tolist()
    pos_sex_tallies = [0, 0]
    pos_and_neg_sex_tallies = [0, 0]
    total_sexes = [0, 0]
    for i in range(len(labels)):
        sex = test_df.at[i, 'sex']
        if labels[i] == " >50K":
            if actual_labels[i] == " >50K":
                pos_sex_tallies[sexes.index(sex)] += 1
                pos_and_neg_sex_tallies[sexes.index(sex)] += 1
        if labels[i] == " <=50K":
            if actual_labels[i] == " <=50K":
                pos_and_neg_sex_tallies[sexes.index(sex)] += 1
        total_sexes[sexes.index(sex)] += 1

    pos_probabilities = [0, 0]
    pos_and_neg_probabilities = [0, 0]
    for i in range(len(pos_and_neg_probabilities)):
        pos_and_neg_probabilities[i] = pos_and_neg_sex_tallies[i] / total_sexes[i]
        pos_probabilities[i] = pos_sex_tallies[i] / total_sexes[i]

    total_equal_opp = 0
    total_equal_odd = 0
    for i in range(len(pos_probabilities)):
        equal_opp = 0
        equal_odd = 0
        for j in range(len(pos_probabilities)):
            if i == j:
                continue
            equal_opp += abs(pos_probabilities[i] - pos_probabilities[j])
            equal_odd += abs(pos_and_neg_probabilities[i] - pos_and_neg_probabilities[j])
        total_equal_opp += equal_opp / (len(pos_probabilities) - 1)
        total_equal_odd += equal_odd / (len(pos_and_neg_probabilities) - 1)
    avg_equal_opp = total_equal_opp / len(pos_probabilities)
    avg_equal_odd = total_equal_odd / len(pos_and_neg_probabilities)
    return avg_equal_opp, avg_equal_odd


# calculates the average equality of opportunity disparity and average equality of odds for race for the given classifier
def equality_race(classifier='forest'):
    if classifier == 'forest':
        labels = random_forest(X_train, y_train, X_validation)
    if classifier == 'tree':
        labels = decision_tree(X_train, y_train, X_validation)
    if classifier == 'knn':
        labels = knn(X_train, y_train, X_validation)
    if classifier == 'svm':
        labels = svm(X_train, y_train, X_validation)
    if classifier == 'lr':
        labels = lr(X_train, y_train, X_validation)
    if classifier == 'ada':
        labels = ada_boost(X_train, y_train, X_validation)
    actual_labels = []
    for index, value in y_validation.items():
        actual_labels.append(value)
    races = train_df['race'].unique().tolist()
    pos_race_tallies = [0, 0, 0, 0, 0]
    pos_and_neg_race_tallies = [0, 0, 0, 0, 0]
    total_races = [0, 0, 0, 0, 0]
    for i in range(len(labels)):
        race = test_df.at[i, 'race']
        if labels[i] == " >50K":
            if actual_labels[i] == " >50K":
                pos_race_tallies[races.index(race)] += 1
                pos_and_neg_race_tallies[races.index(race)] += 1
        if labels[i] == " <=50K":
            if actual_labels[i] == " <=50K":
                pos_and_neg_race_tallies[races.index(race)] += 1
        total_races[races.index(race)] += 1

    pos_probabilities = [0, 0, 0, 0, 0]
    pos_and_neg_probabilities = [0, 0, 0, 0, 0]
    for i in range(len(pos_and_neg_probabilities)):
        pos_and_neg_probabilities[i] = pos_and_neg_race_tallies[i] / total_races[i]
        pos_probabilities[i] = pos_race_tallies[i] / total_races[i]
    total_equal_opp = 0
    total_equal_odd = 0
    for i in range(len(pos_probabilities)):
        equal_opp = 0
        equal_odd = 0
        for j in range(len(pos_probabilities)):
            if i == j:
                continue
            equal_opp += abs(pos_probabilities[i] - pos_probabilities[j])
            equal_odd += abs(pos_and_neg_probabilities[i] - pos_and_neg_probabilities[j])
        total_equal_opp += equal_opp / (len(pos_probabilities) - 1)
        total_equal_odd += equal_odd / (len(pos_and_neg_probabilities) - 1)
    avg_equal_opp = total_equal_opp / len(pos_probabilities)
    avg_equal_odd = total_equal_odd / len(pos_and_neg_probabilities)
    return avg_equal_opp, avg_equal_odd


# prints the fairness metrics of a given classifier
def fairness(classifier='forest'):
    race_dem = dem_disparity_race(classifier)
    sex_dem = dem_disparity_sex(classifier)
    race_opp, race_odd = equality_race(classifier)
    sex_opp, sex_odd = equality_sex(classifier)
    print("Fairness metrics for " + classifier + " classifier")
    print("Average demographic disparity for race: " + str(race_dem))
    print("Average demographic disparity for sex: " + str(sex_dem))
    print("Average equality of opportunity disparity for race: " + str(race_opp))
    print("Average equality of opportunity disparity for sex: " + str(sex_opp))
    print("Average equality of odds disparity for race: " + str(race_odd))
    print("Average equality of odds disparity for sex: " + str(sex_odd))


# creates heatmap and prints correlation matrix
def correlation():
    temp_df = train_df.drop('income', axis=1)
    modify_cols(temp_df)
    label_encoder = LabelEncoder()
    temp_df.iloc[:, 1] = label_encoder.fit_transform(temp_df.iloc[:, 1]).astype('float64')
    temp_df.iloc[:, 2] = label_encoder.fit_transform(temp_df.iloc[:, 2]).astype('float64')
    temp_df.iloc[:, 4] = label_encoder.fit_transform(temp_df.iloc[:, 4]).astype('float64')
    temp_df.iloc[:, 5] = label_encoder.fit_transform(temp_df.iloc[:, 5]).astype('float64')
    temp_df.iloc[:, 6] = label_encoder.fit_transform(temp_df.iloc[:, 6]).astype('float64')
    temp_df.iloc[:, 7] = label_encoder.fit_transform(temp_df.iloc[:, 7]).astype('float64')
    temp_df.iloc[:, 8] = label_encoder.fit_transform(temp_df.iloc[:, 8]).astype('float64')
    temp_df.iloc[:, 9] = label_encoder.fit_transform(temp_df.iloc[:, 9]).astype('float64')
    temp_df.iloc[:, 10] = label_encoder.fit_transform(temp_df.iloc[:, 10]).astype('float64')
    temp_df.iloc[:, 11] = label_encoder.fit_transform(temp_df.iloc[:, 11]).astype('float64')
    temp_df.iloc[:, 12] = label_encoder.fit_transform(temp_df.iloc[:, 12]).astype('float64')
    corr = temp_df.corr()
    print(corr)
    sns.heatmap(corr)
    plt.show()

# fairness()
# fairness('knn')
# fairness('tree')
# fairness('svm')
# fairness('lr')
# fairness('ada')
# correlation()
