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

def check_items_from_soup_search(ReturnObj):
    
    if len(ReturnObj) == 0:
        return 'NA'
    else:
        return ReturnObj[0].text


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
    
       
    abstract_search = soup.find_all("div", {"id": "Abs1-content"})
    abstract = check_items_from_soup_search(abstract_search)
    
    acknowledgement_search = soup.find_all('div', {'id': 'Ack1-content'})
    acknowledgement = check_items_from_soup_search(acknowledgement_search)
    
    author_notes_search = soup.find_all('div', {'id': 'author-information-content'})
    author_notes = check_items_from_soup_search(author_notes_search)
    
    #1. Body of text
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
    
    
    references = soup.find_all('section', {'data-title': 'References'})[0].text
    
    list_keywords = soup.find_all('li', {'class': 'c-article-subject-list__subject'})
    
    keywords = ''
    for keyword in list_keywords:
        keywords = keywords + keyword.text + ', '
    
    additional_info =  soup.find_all('div', {'id': 'additional-information-content'})[0].text
    
    print(list_body_text)
    
url = 'https://link-springer-com.eur.idm.oclc.org/article/10.1007/s11747-021-00778-y'

cj = browser_cookie3.firefox() 


scrape_article_springer(url, cj)