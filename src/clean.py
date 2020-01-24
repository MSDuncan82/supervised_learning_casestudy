import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 


def clean(df):
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['active_user'] = churn['last_trip_date'].apply(lambda x: 1 if x.month >= 6 else 0)
    start = np.array([x.month for x in df['signup_date']])
    end = np.array([x.month for x in df['last_trip_date']])
    df['months_as_user'] = end - start
    for i in ['avg_rating_by_driver', 'avg_rating_of_driver']:
        means = df[i].mean()
        df[i] = df[i].fillna(means)
    return df

def X_y(df):
    df['phone'] = df['phone'].apply(lambda x: 1 if x == 'iphone' else 0)
    cols = df.columns
    if 'city' in cols:
        df = pd.get_dummies(df, columns = ['city'], prefix = 'is')
    
    df.drop(['is_Winterfell', 'last_trip_date', 'signup_date'], axis = 1, inplace = True) 

    y = df.pop('active_user').values
    X = df.values
    return X, y


if __name__ == "__main__":
    churn = pd.read_csv('data/churn_train.csv')
    # churn['signup_date'] = pd.to_datetime(churn['signup_date'])
    # churn['last_trip_date'] = pd.to_datetime(churn['last_trip_date'])
    # churn['active_user'] = churn['last_trip_date'].apply(lambda x: True if x.month >= 6 else False)
    # start = np.array([x.month for x in churn['signup_date']])
    # end = np.array([x.month for x in churn['last_trip_date']])
    # churn['months_as_user'] = end - start
    # churn['months_as_user'] =churn['last_trip_date'].month - churn['signup_date'].month 

    df = clean(churn)

    
    # cols = df.columns.drop(['phone','city','signup_date', 'last_trip_date', 'months_as_user'])
    # churns = df[cols]
    # y = churns.pop('active_user').values  
    # X = churns

    # X_train, X_test, y_train, y_test =train_test_split(X, y, random_state = 1)

    