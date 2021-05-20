# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:20:00 2021

@author: flori
"""

from helpfunctions_preprocessing import create_dataset_publisher
from scrape_functions import scrape_article_sage, scrape_article_wiley, scrape_article_OUP, scrape_article_springer
import browser_cookie3
import pandas as pd
from bs4 import BeautifulSoup as bs



base_url_sage = 'https://journals-sagepub-com.eur.idm.oclc.org/doi/full-xml/'
base_url_springer = 'https://link-springer-com.eur.idm.oclc.org/article/'
base_url_oup = 'https://doi-org.eur.idm.oclc.org/'
base_url_wiley = 'https://onlinelibrary-wiley-com.eur.idm.oclc.org/doi/'

cj = browser_cookie3.firefox() 


# Before you start; make sure to have access through the respective publisher. This happens in three steps
#1. Log in on eur.worldcat 
#2. search the journal, and get access
#3. Test if you can access the full text of the article through the browser

# step 1; create dataframes
df_jom = create_dataset_publisher(['../Data/Raw/journalofmarketing_WoS.csv'], 
                                  cj,
                                  base_url_sage,
                                  scrape_article_sage)

df_jomr = create_dataset_publisher(['../Data/Raw/journalofmarketingresearch_WoS.csv'], 
                                  cj,
                                  base_url_sage,
                                  scrape_article_sage)

df_jams = create_dataset_publisher(['../Data/Raw/journalacademyofmarketingscience_WoS.csv'], 
                                  cj,
                                  base_url_springer,
                                  scrape_article_springer)

df_jcr = create_dataset_publisher(['../Data/Raw/journalofconsumerresearch_WoS.csv'],
                         cj, 
                         base_url_oup,
                         scrape_article_OUP)

df_jcs = create_dataset_publisher(['../Data/Raw/journalofconsumerpsych_WoS.csv'],
                         cj, 
                         base_url_wiley,
                         scrape_article_wiley)


# step 2; put in csv
df_jom[0].to_csv('../Data/Raw/journalofmarketing_data_lim.csv')
df_jomr[0].to_csv('../Data/Raw/journalofmarketingresearch_data_lim.csv')
df_jams[0].to_csv('../Data/Raw/journalacademyofmarketingscience_data_lim.csv')
df_jcr[0].to_csv('../Data/Raw/journalofconsumerresearch_data_lim.csv')
df_jcs[0].to_csv('../Data/Raw/journalofconsumerpsych_data_lim.csv')



# step 3; put csv in gzip 
df_jom_fromCSV = pd.read_csv('../Data/Raw/journalofmarketing_data_lim.csv')
df_jomr_fromCSV = pd.read_csv('../Data/Raw/journalofmarketingresearch_data_lim.csv')
df_jams_fromCSV = pd.read_csv('../Data/Raw/journalacademyofmarketingscience_data_lim.csv')
df_jcr_fromCSV = pd.read_csv('../Data/Raw/journalofconsumerresearch_data_lim.csv')
df_jcs_fromCSV = pd.read_csv('../Data/Raw/journalofconsumerpsych_data_lim.csv')

df_jom_fromCSV.to_parquet("../Data/scraped/journalofmarketing_data_lim.gzip", compression='gzip')
df_jomr_fromCSV.to_parquet("../Data/scraped/journalofmarketingresearch_data_lim.gzip", compression='gzip')
df_jams_fromCSV.to_parquet("../Data/scraped/journalacademyofmarketingscience_data_lim.gzip", compression='gzip')
df_jcr_fromCSV.to_parquet("../Data/scraped/journalofconsumerresearch_data_lim.gzip", compression='gzip')
df_jcs_fromCSV.to_parquet("../Data/scraped/journalofconsumerpsych_data_lim.gzip", compression='gzip')
