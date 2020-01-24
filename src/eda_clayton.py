import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
# sys.path.append('/home/cv/dsi/case_studies/supervised-learning-case-study/src/')
# print(str(sys))
from clean_df import clean_df
from sklearn.linear_model import LogisticRegression
print(clean_df)

df_raw = clean_df(pd.read_csv('../data/churn_train.csv'))

df_test = clean_df(df_raw)

sns.pairplot(df_test)
sns.plt.show()

