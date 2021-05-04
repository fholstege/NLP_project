# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:42:37 2021

@author: flori
"""

import urllib.request
import requests

opener = urllib.request.build_opener()
opener.addheaders = [('Accept', 'application/vnd.crossref.unixsd+xml')]
r = opener.open('https://doi.org/10.1177/0022242921991798')
print (r.info()['Link'])


response= requests.get('http://journals.sagepub.com/doi/full-xml/10.1177/0022242921991798')
response.raise_for_status()
response.content

data = response.json()
print(data)