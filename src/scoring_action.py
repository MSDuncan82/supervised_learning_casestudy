import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
sys.path.append('/home/mike/dsi/case_studies/supervised-learning-case-study/src/')
from clean_df import clean, X_y
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
plt.style.use('fivethirtyeight')

if __name__ == "__main__":
    
    ### Get Data ###
    
    df_train = clean(pd.read_csv('../data/churn_train.csv')).drop(['avg_rating_by_driver', "city"], axis=1)
    df_test = clean(pd.read_csv('../data/churn_test.csv')).drop(['avg_rating_by_driver', "city"], axis=1)
   
    X, y = X_y(df_train)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
    X_cols = ['avg_dist', 'avg_rating_of_driver', 'avg_surge', 'phone', 'surge_pct',
       'trips_in_first_30_days', 'luxury_car_user', 'weekday_pct']

    
    ### Random Forest ###

    rf = RandomForestClassifier(n_estimators=200, max_depth = 8)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)        
    
    ### Logistic Regression ###
    
    # Ridge
    lr2 = LogisticRegression()
    lr2.fit(X_train, y_train)
    
    lr2_pred_p = lr2.predict_proba(X_test)
    y_pred = lr2.predict(X_test)
    
    # Lasso
    lr1 = LogisticRegression(penalty='l1')
    lr1.fit(X_train, y_train)
    
    lr1_pred_p = lr1.predict_proba(X_test)
    y_pred = lr1.predict(X_test) 
    
    ### CatBoost ###
    
    cat = CatBoostClassifier(iterations=2,
                        depth=2,
                        learning_rate=1,
                        loss_function='Logloss',
                        verbose=True)
    
    cat.fit(X_train, y_train)

    preds_class = cat.predict(X_test)
    preds_proba = cat.predict_proba(X_test)

    
    a_scores = {}
    f_scores = {}
    models = {'lr1':lr1, 'lr2':lr2, 'rf':rf, 'cat':cat}
    
    for model_str, model in models.items():
        a_scores[model_str] = model.score(X_test, y_test)
        if 'l' in model_str:
            f_scores[model_str] = model.coef_
        else:
            f_scores[model_str] = model.feature_importances_
        
    df_f_scores = pd.DataFrame(
        [arr.flatten() for arr in f_scores.values()],
         index=f_scores.keys(),
         columns = X_cols)
    
    df_f_scores.loc['rf'] = df_f_scores.loc['rf'] * 100
    
    plt.rcParams['font.size'] = 16
    
    fig, ax = plt.subplots(figsize=(10, 10))
        
    ind = np.arange(len(X_cols))  # the x locations for the groups
    width = 0.35       # the width of the bars
    order = np.argsort(df_f_scores.loc['rf'].values)

    rf_bars = ax.barh(ind, df_f_scores.iloc[2, order], width)
    cat_bars = ax.barh(ind+width, df_f_scores.iloc[3, order], width)

    # add some
    ax.set_xlabel('Relative Feature Importance')
    ax.set_title('Feature Importances')
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(np.array(X_cols)[order])
    
    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(90)

    ax.legend((rf_bars[0], cat_bars[0]), ('Random Forest', 'CatBoost'))

    plt.tight_layout()
    plt.savefig('../imgs/full_f_imp_action.png')
    
    fig, ax = plt.subplots(figsize=(10, 10))
        
    ind = np.arange(len(X_cols))  # the x locations for the groups
    width = 0.35       # the width of the bars
    order = np.argsort(df_f_scores.loc['lr1'].values)

    lf1_bars = ax.barh(ind, df_f_scores.iloc[0, order], width)
    lf2_bars = ax.barh(ind+width, df_f_scores.iloc[1, order], width)

    # add some
    ax.set_xlabel('Coefficients')
    ax.set_title('Logistic Coefficients')
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(np.array(X_cols)[order])
    
    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(90)

    ax.legend((lf1_bars[0], lf2_bars[0]), ('LASSO Log Regression', 'Ridge Log Regression'))

    plt.tight_layout()
    plt.savefig('../imgs/full_coefs_action.png')
    
    
    
    