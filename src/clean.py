import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def clean(df):
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['active_user'] = df['last_trip_date'].apply(lambda x: True if x.month >= 6 else False)
    start = np.array([x.month for x in df['signup_date']])
    end = np.array([x.month for x in df['last_trip_date']])
    df['months_as_user'] = end - start
    for i in ['avg_rating_by_driver', 'avg_rating_of_driver']:
        means = df[i].mean()
        df[i] = df[i].fillna(means)
    return df

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
    cols = df.columns.drop(['phone','city','signup_date', 'last_trip_date', 'months_as_user'])
    churns = df[cols]
    y = churns.pop('active_user').values  
    X = churns

    X_train, X_test, y_train, y_test =train_test_split(X, y, random_state = 1)

    rf = RF(n_estimators = 100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    confuse = confusion_matrix(y_test, y_pred)
    print(acc)
    print(confuse)

    order = np.argsort(rf.feature_importances_) 
    column = X_train.columns[order][::-1]

    X1_train = X_train[column[1:]]
    X1_test = X_test[column[1:]]
    
    
    rf1 = RF(n_estimators = 300, oob_score=True)
    rf1.fit(X_train, y_train)
    y1_pred = rf1.predict(X_test)

    acc1 = accuracy_score(y_test, y1_pred)
    confuse1 = confusion_matrix(y_test, y1_pred)
    print('----------')
    print(acc1)
    print(confuse1)
    
    cols = df.columns.drop(['phone','city','signup_date', 'last_trip_date'])
    churns = df[cols]
    y = churns.pop('active_user').values  
    X = churns
    X_train, X_test, y_train, y_test =train_test_split(X, y, random_state = 1)

    logit = LR(sovler = 'lbfgs')
    logit.fit(X_train, y_train)
    y2_pred = logit.predict(X_test)

    print(logit.score(X_test,y_test))

    cols = df.columns.drop(['phone','city','signup_date', 'last_trip_date'])



