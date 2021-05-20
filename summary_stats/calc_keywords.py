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

top_keywords_alltime = extract_keywords(df_journals_merged['Author Keywords'], n=20, top = True)
top_keywords_list = list(top_keywords_alltime.keys())



years = df_journals_merged['Year'].unique()
years_noNaN = years[~np.isnan(years)]

df_top_keywords =  pd.DataFrame(columns = top_keywords_list)
df_top_keywords['Year'] = years_noNaN



for year in years_noNaN:
    df_journals_year = df_journals_merged[df_journals_merged['Year'] == year]
    
    selected_keywords = extract_keywords(df_journals_year['Author Keywords'], top = False,selected = top_keywords_list)
    
    for keyword, value in selected_keywords.items():
        df_top_keywords.loc[df_top_keywords['Year'] == year, keyword] = value
    

df_top_keywords_melted = pd.melt(df_top_keywords, id_vars='Year')
df_top_keywords_melted_noNA = df_top_keywords_melted.fillna(0)
df_top_keywords_melted_post08 = df_top_keywords_melted_noNA[df_top_keywords_melted_noNA['Year']>= 2008]

# get colors for graphs
colors = cm.get_cmap('tab20').colors
index_color = 0

for top_keyword in top_keywords_list:
    
    df_top_keyword = df_top_keywords_melted_post08[df_top_keywords_melted_post08['variable'] == top_keyword]
    df_top_keyword_ordered = df_top_keyword[::-1]
    plt.plot(df_top_keyword_ordered['Year'], df_top_keyword_ordered['value'].cumsum(), label = top_keyword, color = colors[index_color])
    index_color += 1
    
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')