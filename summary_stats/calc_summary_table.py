# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:59:24 2021

@author: flori
"""

import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk.util import ngrams
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# read the zipped files
df_jom_merged = pd.read_parquet('../Data/clean/journalofmarketing_merged.gzip')
df_jomr_merged = pd.read_parquet('../Data/clean/journalofmarketingresearch_merged.gzip')
df_jcr_merged = pd.read_parquet('../Data/clean/journalofconsumerresearch_merged.gzip')
df_jcp_merged = pd.read_parquet('../Data/clean/journalofconsumerpsych_merged.gzip')
df_jam_merged = pd.read_parquet('../Data/clean/journalacademyofmarketingscience_merged.gzip')

# add journal titles
df_jom_merged['Journal'] = 'Journal of Marketing' 
df_jomr_merged['Journal'] = 'Journal of Marketing Research' 
df_jcr_merged['Journal'] = 'Journal of Consumer Research' 
df_jcp_merged['Journal'] = 'Journal of Consumer Psychology' 
df_jam_merged['Journal'] = 'Journal Academy of Marketing Science' 

# get dataframes together
frames_journals = [df_jom_merged, df_jomr_merged, df_jcr_merged, df_jcp_merged, df_jam_merged]

# merge the journals in one dataframe, vertically
df_journals_merged = pd.concat(frames_journals)
df_count_obs = df_journals_merged.groupby(['Year']).size()

# 
plt.bar(df_count_obs.index, df_count_obs, color = 'Grey')
plt.xlabel('Year')
plt.ylabel('Number of articles in dataset')