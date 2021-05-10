# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:20:00 2021

@author: flori
"""

from helpfunctions_preprocessing import create_dataset_publisher
from scrape_functions import scrape_article_sage, scrape_article_wiley, scrape_article_OUP, scrape_article_springer
import browser_cookie3
import pandas as pd


base_url_springer = 'https://link-springer-com.eur.idm.oclc.org/article/'
base_url_oup = 'https://doi-org.eur.idm.oclc.org/'
base_url_wiley = 'https://onlinelibrary-wiley-com.eur.idm.oclc.org/doi/'
base_url_sage = 'https://journals-sagepub-com.eur.idm.oclc.org/doi/full-xml/'

cj = browser_cookie3.firefox() 

# Before you start; make sure to have access through the respective publisher. This happens in three steps
#1. Log in on eur.worldcat 
#2. search the journal, and get access
#3. Test if you can access the full text of the article through the browser

df_jom = create_dataset_publisher(['Data/Raw/journalofmarketing_WoS.csv'], 
                                  cj,
                                  base_url_sage,
                                  scrape_article_sage)

df_jcs[0].to_parquet("data/scraped/journalofmarketing.gzip", compression='gzip')

df_jomr = create_dataset_publisher(['Data/Raw/journalofmarketingresearch_WoS.csv'],
                         cj, 
                         base_url_sage,
                         scrape_article_sage)

df_jomr[0].to_parquet("data/scraped/journalofmarketingresearch.gzip", compression='gzip')


df_jcr = create_dataset_publisher(['Data/Raw/journalofconsumerresearch_WoS.csv'],
                         cj, 
                         base_url_oup,
                         scrape_article_OUP)

# instead of html get text for rep_list
for row in range(0, df_jcr[0].shape[0]+1):
    df_jcr[0]['ref_list'].iloc[row] = df_jcr[0]['ref_list'].iloc[row].text
# no NAs

# save first as csv then as parquet to avoid issues with transformation to parquet
df_jcr[0].to_csv("data/scraped/journalofconsumerresearch.csv")
df = pd.read_csv("data/scraped/journalofconsumerresearch.csv")
df.to_parquet("data/scraped/journalofconsumerresearch.gzip", compression='gzip')

df_jams = create_dataset_publisher(['Data/Raw/journalacademyofmarketingscience_WoS.csv'],
                         cj, 
                         base_url_springer,
                         scrape_article_springer)   
df_jams[0].to_parquet("data/scraped/journalacademyofmarketingscience.gzip", compression='gzip')
# 7 NAs, either DOIs are not available anymore or only pdf full-text e.g. via SAGE

df_jcs = create_dataset_publisher(['Data/Raw/journalofconsumerpsych_WoS.csv'],
                         cj, 
                         base_url_wiley,
                         scrape_article_wiley)
df_jcs[0].to_parquet("data/scraped/journalofconsumerpsych.gzip", compression='gzip')

# for some the ref_list is length 0, e.g. after 29/972 is printed (so the 30th article)
# also article 402, 423, 600, 618, 828, 835, 837, 839, 845 - 856, 858 - 877, 892,
# 894, 895, 898, 900, 901, 906, 908, 910, 911, 915, 916, 927 - 934, 936 - 938, 
# 954 - 959, 961 - 972
# NAs: 71/972 (mostly dois coming pretty late, probably older articles?)