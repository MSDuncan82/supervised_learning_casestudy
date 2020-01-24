from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
import pandas as pd
from clean import clean, X_y

if __name__ == "__main__":
    churn = pd.read_csv('data/churn_train.csv')
    df = clean(churn)
    X_train, X_test, y_train, y_test = X_y(df)
    rf = RF(n_estimators=200, max_depth = 8)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    print(rf.score(X_test, y_test))