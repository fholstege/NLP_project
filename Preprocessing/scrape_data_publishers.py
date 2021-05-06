# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:20:00 2021

@author: flori
"""

from helpfunctions_preprocessing import create_dataset_publisher
from scrape_functions import scrape_article_sage, scrape_article_wiley, scrape_article_OUP, scrape_article_springer
import browser_cookie3


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


df_jomr = create_dataset_publisher(['Data/Raw/journalofmarketingresearch_WoS.csv'],
                         cj, 
                         base_url_sage,
                         scrape_article_sage)

df_jcr = create_dataset_publisher(['Data/Raw/journalofconsumerresearch_WoS.csv'],
                         cj, 
                         base_url_oup,
                         scrape_article_OUP)


df_jams = create_dataset_publisher(['Data/Raw/journalacademyofmarketingscience_WoS.csv'],
                         cj, 
                         base_url_springer,
                         scrape_article_springer)

df_jcs = create_dataset_publisher(['Data/Raw/journalofconsumerpsych_WoS.csv'],
                         cj, 
                         base_url_wiley,
                         scrape_article_wiley)