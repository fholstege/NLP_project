# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:42:59 2021

@author: flori
"""



import urllib.request
import requests
import browser_cookie3
import time
from bs4 import BeautifulSoup as bs
import pandas as pd

from helpfunctions_preprocessing import create_list_urls, check_items_from_soup_search, create_dataset_publisher



#df_WoS_jams = pd.read_csv('data/Raw/journalacademyofmarketingscience_WoS.csv')
#df_WoS_jams['DOI']





def scrape_article_springer(url, cookies):
    
    
    try:
        response = requests.get(url, cookies=cookies)
        response.raise_for_status()
    except requests.HTTPError as exception:
        print(exception) # or save to logfile together with url
        return "NA"
    
    # parse xml content
    soup = bs(response.content, "html.parser")
    
    # information to crosscheck with scopus data to confirm we have correct data
    # use title 
    title = soup.find_all("h1", {"class": "c-article-title"})[0].text
    
    doi = soup.find_all('a', {'data-track-action': 'view doi'})[0].get('href')
    print(doi)
    
    # get the abstract   
    abstract_search = soup.find_all("div", {"id": "Abs1-content"})
    abstract = check_items_from_soup_search(abstract_search)
    
    # get the acknowledgement
    acknowledge_search = soup.find_all('div', {'id': 'Ack1-content'})
    acknowledge = check_items_from_soup_search(acknowledge_search)
    
    author_notes_search = soup.find_all('div', {'id': 'author-information-content'})
    author_notes = check_items_from_soup_search(author_notes_search)
    
    # Body of text
    list_body_text = []
    sections = soup.find_all("div", {"class": "c-article-section"})
    
    for section in sections:
        # get section id
        id_section = section.get("id")
        
        # skip over abstract, stop when at notes or bibliography
        if id_section == 'Abs1-section':
            continue 
        elif id_section == 'notes-section' or id_section == 'Bib1-section':
            break
        else:         
           
            section_text = section.text
            
            list_body_text.append(section_text)
    

    
    body = ' '.join(map(str, list_body_text))
    
    ref_list_search = soup.find_all('section', {'data-title': 'References'})
    ref_list = check_items_from_soup_search(ref_list_search)
    
    list_keywords = soup.find_all('li', {'class': 'c-article-subject-list__subject'})
    keywords = ', '.join(map(str, list_body_text))
    
    fngroup_search =  soup.find_all('div', {'id': 'additional-information-content'})
    fngroup = check_items_from_soup_search(fngroup_search)
    
    return title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup
    

url = 'https://link-springer-com.eur.idm.oclc.org/article/10.1007/s11747-021-00778-y'

base_url = 'https://link-springer-com.eur.idm.oclc.org/article/'
cj = browser_cookie3.firefox() 

scrape_article_springer(url, cj)
