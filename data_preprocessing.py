from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


def create_data():
    test_df = pd.read_csv('test.csv')
    train_df = pd.read_csv('train.csv')
    return test_df, train_df


# normalizes age
def modify_age_column(df):
    for i in df.index:
        if df.at[i, 'age'] < 14:
            df.at[i, 'age'] = '0-14'
        elif 15 <= df.at[i, 'age'] < 25:
            df.at[i, 'age'] = '15-24'
        elif 25 <= df.at[i, 'age'] < 35:
            df.at[i, 'age'] = '25-34'
        elif 35 <= df.at[i, 'age'] < 45:
            df.at[i, 'age'] = '35-44'
        elif 45 <= df.at[i, 'age'] < 55:
            df.at[i, 'age'] = '45-54'
        elif 55 <= df.at[i, 'age'] < 65:
            df.at[i, 'age'] = '55-64'
        elif 65 <= df.at[i, 'age'] < 75:
            df.at[i, 'age'] = '65-74'
        elif 75 <= df.at[i, 'age'] < 85:
            df.at[i, 'age'] = '75-84'
        elif df.at[i, 'age'] >= 85:
            df.at[i, 'age'] = '85+'


# normalizes capital-gain and capital-loss
def modify_capital_cols(df):
    for i in df.index:
        if df.at[i, 'capital-loss'] == 0:
            df.at[i, 'capital-loss'] = '0'
        elif df.at[i, 'capital-loss'] <= 250:
            df.at[i, 'capital-loss'] = '<=250'
        elif df.at[i, 'capital-loss'] <= 500:
            df.at[i, 'capital-loss'] = '<=500'
        elif df.at[i, 'capital-loss'] <= 750:
            df.at[i, 'capital-loss'] = '<=750'
        elif df.at[i, 'capital-loss'] <= 1000:
            df.at[i, 'capital-loss'] = '<=1000'
        elif df.at[i, 'capital-loss'] <= 1250:
            df.at[i, 'capital-loss'] = '<=1250'
        elif df.at[i, 'capital-loss'] <= 1500:
            df.at[i, 'capital-loss'] = '<=1500'
        elif df.at[i, 'capital-loss'] <= 1750:
            df.at[i, 'capital-loss'] = '<=1750'
        elif df.at[i, 'capital-loss'] <= 2000:
            df.at[i, 'capital-loss'] = '<=2000'
        elif df.at[i, 'capital-loss'] <= 2250:
            df.at[i, 'capital-loss'] = '<=2250'
        elif df.at[i, 'capital-loss'] <= 2500:
            df.at[i, 'capital-loss'] = '<=2500'
        elif df.at[i, 'capital-loss'] <= 2750:
            df.at[i, 'capital-loss'] = '<=2750'
        elif df.at[i, 'capital-loss'] <= 3000:
            df.at[i, 'capital-loss'] = '<=3000'
        elif df.at[i, 'capital-loss'] <= 3250:
            df.at[i, 'capital-loss'] = '<=3250'
        elif df.at[i, 'capital-loss'] <= 3500:
            df.at[i, 'capital-loss'] = '<=3500'
        elif df.at[i, 'capital-loss'] <= 3750:
            df.at[i, 'capital-loss'] = '<=3750'
        elif df.at[i, 'capital-loss'] <= 4000:
            df.at[i, 'capital-loss'] = '<=4000'
        elif df.at[i, 'capital-loss'] <= 4250:
            df.at[i, 'capital-loss'] = '<=4250'
        elif df.at[i, 'capital-loss'] <= 4500:
            df.at[i, 'capital-loss'] = '<=4500'
        elif df.at[i, 'capital-loss'] <= 4750:
            df.at[i, 'capital-loss'] = '<=4750'
        elif df.at[i, 'capital-loss'] <= 5000:
            df.at[i, 'capital-loss'] = '<=5000'
        elif df.at[i, 'capital-loss'] <= 5250:
            df.at[i, 'capital-loss'] = '<=5250'
        elif df.at[i, 'capital-loss'] <= 5500:
            df.at[i, 'capital-loss'] = '<=5500'
        elif df.at[i, 'capital-loss'] > 5500:
            df.at[i, 'capital-loss'] = '>5500'

    for i in df.index:
        if df.at[i, 'capital-gain'] == 0:
            df.at[i, 'capital-gain'] = '0'
        elif df.at[i, 'capital-gain'] <= 6250:
            df.at[i, 'capital-gain'] = '<=6250'
        elif df.at[i, 'capital-gain'] <= 12500:
            df.at[i, 'capital-gain'] = '<=12500'
        elif df.at[i, 'capital-gain'] <= 18750:
            df.at[i, 'capital-gain'] = '<=18750'
        elif df.at[i, 'capital-gain'] <= 25000:
            df.at[i, 'capital-gain'] = '<=25000'
        elif df.at[i, 'capital-gain'] <= 31250:
            df.at[i, 'capital-gain'] = '<=31250'
        elif df.at[i, 'capital-gain'] <= 37500:
            df.at[i, 'capital-gain'] = '<=43750'
        elif df.at[i, 'capital-gain'] <= 50000:
            df.at[i, 'capital-gain'] = '<=50000'
        elif df.at[i, 'capital-gain'] > 50000:
            df.at[i, 'capital-gain'] = '>50000'


# normalizes hours-per-week
def modify_hours_per_week(df):
    for i in df.index:
        if df.at[i, 'hours-per-week'] <= 5:
            df.at[i, 'hours-per-week'] = '<=5'
        elif df.at[i, 'hours-per-week'] <= 10:
            df.at[i, 'hours-per-week'] = '<=10'
        elif df.at[i, 'hours-per-week'] <= 15:
            df.at[i, 'hours-per-week'] = '<=15'
        elif df.at[i, 'hours-per-week'] <= 20:
            df.at[i, 'hours-per-week'] = '<=25'
        elif df.at[i, 'hours-per-week'] <= 25:
            df.at[i, 'hours-per-week'] = '<=25'
        elif df.at[i, 'hours-per-week'] <= 30:
            df.at[i, 'hours-per-week'] = '<=30'
        elif df.at[i, 'hours-per-week'] <= 35:
            df.at[i, 'hours-per-week'] = '<=35'
        elif df.at[i, 'hours-per-week'] <= 40:
            df.at[i, 'hours-per-week'] = '<=40'
        elif df.at[i, 'hours-per-week'] <= 45:
            df.at[i, 'hours-per-week'] = '<=45'
        elif df.at[i, 'hours-per-week'] <= 50:
            df.at[i, 'hours-per-week'] = '<=50'
        elif df.at[i, 'hours-per-week'] <= 55:
            df.at[i, 'hours-per-week'] = '<=55'
        elif df.at[i, 'hours-per-week'] <= 60:
            df.at[i, 'hours-per-week'] = '<=60'
        elif df.at[i, 'hours-per-week'] <= 65:
            df.at[i, 'hours-per-week'] = '<=65'
        elif df.at[i, 'hours-per-week'] <= 70:
            df.at[i, 'hours-per-week'] = '<=70'
        elif df.at[i, 'hours-per-week'] <= 75:
            df.at[i, 'hours-per-week'] = '<=75'
        elif df.at[i, 'hours-per-week'] <= 80:
            df.at[i, 'hours-per-week'] = '<=80'
        elif df.at[i, 'hours-per-week'] <= 85:
            df.at[i, 'hours-per-week'] = '<=85'
        elif df.at[i, 'hours-per-week'] > 85:
            df.at[i, 'hours-per-week'] = '>85'


# encodes original dataset
def encode_features(X, X_test):
    OHE = OneHotEncoder(sparse=False, handle_unknown='ignore')
    OE = OrdinalEncoder()
    column_transform = make_column_transformer(
        (OHE, ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']),
        (OE, ['age', 'education', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']))
    X_1 = column_transform.fit_transform(X)
    X_2 = column_transform.transform(X_test)
    return X_1, X_2


# encodes dataset that was modified according to fairness
def improved_encode_features(X, X_test, part=1):
    OHE = OneHotEncoder(sparse=False, handle_unknown='ignore')
    OE = OrdinalEncoder()
    if part == 1:
        column_transform = make_column_transformer(
            (OHE, ['workclass', 'marital-status', 'occupation', 'relationship', 'native-country']),
            (OE, ['age', 'education', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']))
    else:
        column_transform = make_column_transformer(
            (OHE, ['marital-status', 'relationship']),
            (OE, ['age', 'education', 'education-num', 'capital-gain', 'capital-loss']))
    X_1 = column_transform.fit_transform(X)
    X_2 = column_transform.transform(X_test)
    return X_1, X_2


# normalizes age, capital-gain, capital-loss, and hours-per-week
def modify_cols(df):
    modify_age_column(df)
    modify_capital_cols(df)
    modify_hours_per_week(df)
