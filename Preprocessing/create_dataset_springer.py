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




def scrape_article_springer(url, cookies):
    
    
    try:
        response = requests.get(url, cookies=cookies)
        response.raise_for_status()
    except requests.HTTPError as exception:
        print(exception) # or save to logfile together with url
        return "NA"
    
    # parse xml content
    soup = bs(response.content, "html")
    
    # information to crosscheck with scopus data to confirm we have correct data
    # use title 
    title = soup.find_all("h1", {"class": "c-article-title"})[0].text
    
    # or use doi
    doi = soup.find_all("article-id")[0].text
   
    
    print(title)
    
url = 'https://link-springer-com.eur.idm.oclc.org/article/10.1007/s11747-021-00778-y'

cj = browser_cookie3.firefox() 


scrape_article_springer(url, cj)