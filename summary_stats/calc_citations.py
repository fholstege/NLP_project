# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:53:20 2021

@author: flori
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


# read the zipped files
df_journals_merged = pd.read_parquet('../Data/clean/all_journals_adject_nouns_merged_withLemmatized.gzip')

df_journals_merged.columns

fields = ['Accounting',
       'Agricultural and Biological Sciences', 'Arts and Humanities',
       'Business and Int. Mgmt.', 'Computer Science',
       'Economics / Econometrics / Finance', 'Environmental Science',
       'Marketing', 'Mathematics', 'Medicine', 'Mgmt. Information Systems',
       'Mgmt. Science and OR', 'Mgmt. Technology and Innovation',
       'Multidisciplinary', 'Neuroscience', 'News', 'Org. Behavior and HR',
       'Psychology', 'Social Sciences', 'Strategy and Mgmt.',
       'Tourism, Leisure, Hospitality Mgmt.']

# table
df_fields_avg_citations_table = df_journals_merged[fields].mean()
df_fields_avg_citations_table.sort_values(ascending = False)
round(df_fields_avg_citations_table.sort_values(ascending = False)*100, 1).to_latex()

df_fields_avg_citations = df_journals_merged.groupby(['year'])[fields].mean()


df_fields_avg_citations_melted = pd.melt(df_fields_avg_citations.reset_index(), id_vars='year')

fields_selected = [x for x in fields if x in ['Computer Science', 'Mathematics', 'Psychology', 'Medicine', 'Social Sciences', 'Statistics']]


# get colors for graphs
colors = cm.get_cmap('tab20').colors
index_color = 0
for field in fields_selected:
    df_field = df_fields_avg_citations_melted[df_fields_avg_citations_melted['variable'] == field]
    plt.plot(df_field['year'], df_field['value']*100, label = field, color = colors[index_color])
    index_color += 1

plt.xlabel("Year")
plt.ylabel("% of Journals in Reference list  ")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')