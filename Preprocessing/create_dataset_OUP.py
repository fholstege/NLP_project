# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:20:36 2021

@author: flori
"""


import urllib.request
import requests
import browser_cookie3
import time
from bs4 import BeautifulSoup as bs
import pandas as pd




def scrape_article_OUP(url, cookies):
    
    
    try:
        response = requests.get(url, cookies=cookies)
        response.raise_for_status()
    except requests.HTTPError as exception:
        print(exception) # or save to logfile together with url
        return "NA"
    
    # parse xml content
    soup = bs(response.content, "html")
    
    print(soup)
    
url = 'https://academic-oup-com.eur.idm.oclc.org/jcr/article/47/6/914/5890485'

cj = browser_cookie3.firefox() 


scrape_article_OUP(url, cj)