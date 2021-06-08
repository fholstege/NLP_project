# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:26:48 2021

@author: flori
"""
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk.util import ngrams
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import display, Latex

df_journals_merged = pd.read_parquet('../Data/clean/all_journals_adject_nouns_merged_withLemmatized.gzip')

# merge the journals in one dataframe, vertically
df_journals_merged.columns

def lemmatize_keywords(keywords_list):
    # init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    keywords_lemmatized = []
    
    for keywords in keywords_list:
        
        if keywords is not None:
            keywords_split = keywords.split(';')
            
            for keyword in keywords_split:
                lemmatized_keyword = lemmatizer.lemmatize(keyword)
                keywords_lemmatized.append(lemmatized_keyword)
                
    return keywords_lemmatized


def extract_keywords(keywords_list, lemmatize = True, top = True, n=20,selected = None):
    
    if lemmatize:
        keywords_list = lemmatize_keywords(keywords_list)
    
    keyword_counter = Counter(keywords_list)
    
    if top:
        
        top_keywords = keyword_counter.most_common(n)
        return dict(top_keywords)
    else: 
        dict_keywords = dict(keyword_counter)
        dict_selected_keywords = dict((key,value) for key, value in dict_keywords.items() if key in selected)
    
        return dict_selected_keywords

# get top keywords
top_keywords_alltime = extract_keywords(df_journals_merged['keywords'], n=20, top = True)
top_keywords_list = list(top_keywords_alltime.keys())

# get table
df_top_keywords_alltime = pd.DataFrame({'keyword': top_keywords_alltime.keys(), 'count':top_keywords_alltime.values()})
df_top_keywords_alltime.to_latex()

# get years
years = df_journals_merged['year'].unique()
years_noNaN = years[~np.isnan(years)]

# get top keywords per year, fill dataframe
df_top_keywords =  pd.DataFrame(columns = top_keywords_list)
df_top_keywords['year'] = years_noNaN


for year in years_noNaN:
    df_journals_year = df_journals_merged[df_journals_merged['year'] == year]
    
    selected_keywords = extract_keywords(df_journals_year['keywords'], top = False,selected = top_keywords_list)
    
    for keyword, value in selected_keywords.items():
        df_top_keywords.loc[df_top_keywords['year'] == year, keyword] = value
    

df_top_keywords_melted = pd.melt(df_top_keywords, id_vars='year')
df_top_keywords_melted_noNA = df_top_keywords_melted.fillna(0)
df_top_keywords_melted_post08 = df_top_keywords_melted_noNA[df_top_keywords_melted_noNA['year']>= 2008]

# get colors for graphs
colors = cm.get_cmap('tab20').colors
index_color = 0

top_keywords_list

selected_keywords = [' Social media', ' Sustainability', 'Advertising', ' Innovation', ' Decision making', ' Emotions']

for top_keyword in selected_keywords:
    
    df_top_keyword = df_top_keywords_melted_post08[df_top_keywords_melted_post08['variable'] == top_keyword]
    df_top_keyword_ordered = df_top_keyword[::-1]
    plt.plot(df_top_keyword_ordered['year'], df_top_keyword_ordered['value'].rolling(window=3).mean(), label = top_keyword, color = colors[index_color])
    index_color += 1
    
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Year')
plt.ylabel('Number of times mentioned as keyword \n (3-year moving average)')