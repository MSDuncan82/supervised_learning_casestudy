import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/jacobtryba/DSI/assignments/supervised-learning-case-study/src/')
from clean_df import clean, X_y
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
import sklearn as sklearn

def cat_boost(X_train, X_test, y_train):
    model = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)
    # train the model
    model.fit(train_data, train_labels)
    # make the prediction using the resulting model
    preds_class = model.predict(test_data)
    preds_proba = model.predict_proba(test_data)
    print("class = ", preds_class)
    print("proba = ", preds_proba)
    return preds_class, preds_proba

if __name__ == "__main__":

    df_train = clean(pd.read_csv('/Users/jacobtryba/DSI/assignments/supervised-learning-case-study/data/churn_train.csv')).drop('months_as_user', axis =1)
    df_test = clean(pd.read_csv('/Users/jacobtryba/DSI/assignments/supervised-learning-case-study/data/churn_test.csv')).drop('months_as_user', axis =1)

    X_train, X_test, y_train, y_test = X_y(df_train)


    model = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)
    # train the model
    model.fit(X_train, y_train)
    # make the prediction using the resulting model
    preds_class = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)

    
    print(preds_class)
    print(preds_proba)

    print(sklearn.metrics.accuracy_score(y_test, preds_class, normalize=True, sample_weight=None))
    print(model.get_feature_importance())