# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:53:20 2021

@author: flori
"""
import pandas as pd
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

fields = ['AI', 'Behavioral Neuroscience', 'Business / Int. Management',
       'Computer Science', 'Economics / Econometrics', 'Marketing', 'Medicine',
       'Mgmt Information Systems', 'Mgmt of Tech. and Innovation',
       'Mgmt. Science / OR', 'Org. Behavior / HR Management', 'Psychology',
       'Social Sciences', 'Sociology / Pol.Sc.', 'Statistics',
       'Strategy and Management']


df_fields_avg_citations = df_journals_merged.groupby(['Year'])[fields].mean()
df_fields_avg_citations_melted = pd.melt(df_fields_avg_citations.reset_index(), id_vars='Year')

fields_noMarketing = [x for x in fields if x != 'Marketing']


# get colors for graphs
colors = cm.get_cmap('tab20').colors
index_color = 0
for field in fields_noMarketing:
    df_field = df_fields_avg_citations_melted[df_fields_avg_citations_melted['variable'] == field]
    plt.plot(df_field['Year'], df_field['value'], label = field, color = colors[index_color])
    index_color += 1

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')