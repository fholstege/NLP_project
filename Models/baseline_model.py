# -*- coding: utf-8 -*-
"""
Created on Sun May 30 10:40:48 2021

@author: flori
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from helpfunctions_models import regression_results
import statsmodels.api as sm
from scipy import stats



# read with topics 
df = pd.read_parquet('../data/clean/data_journals_merged_topics.gzip')


# get histogram
plt.hist(np.log( 1 + df['num_ref']), bins = 30)

# add logged 
df.columns

independent_var = ['Topic 1', 'Topic 2', 'Topic 3']

model_var = ['num_ref'] + independent_var

df_model = df[model_var]
df_model_noNA = df_model.dropna()

X = df_model_noNA[independent_var]
X2 = sm.add_constant(X)

y_ref = np.log(1 + df_model_noNA['num_ref'])


lm = LinearRegression(fit_intercept=True)
lm.fit(X, y_ref)
y_ref_pred = lm.predict(X)

regression_results(y_ref, y_ref_pred)
lm.coef_