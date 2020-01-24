import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
sys.path.append('/home/mike/dsi/case_studies/supervised-learning-case-study/src/')
from clean_df import clean, X_y
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    
    df_train = clean(pd.read_csv('../data/churn_train.csv')).drop('months_as_user', axis=1)
    df_test = clean(pd.read_csv('../data/churn_test.csv')).drop('months_as_user', axis=1)
    
    X_train, X_test, y_train, y_test = X_y(df_train)
    
    log_model = LogisticRegression(class_weight='balanced')
    log_model.fit(X_train, y_train)
    
    y_pred_p = log_model.predict_proba(X_test)
    y_pred = log_model.predict(X_test)
    
    score_train = log_model.score(X_train, y_train)
    score_test = log_model.score(X_test, y_test)
    
    
    
    # ## Validation
    
    # X_train_val, X_test_val, y_train_val, y_test_val = X_y(df_test)

    # score_val = log_model.score(X_test_val, y_test_val)