#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import json
import glob
import re
from collections import Counter
from geopy.geocoders import Nominatim
from geopy import geocoders
import csv
import codecs
import time


# In[4]:


fi = open('irma_oneday_06.csv', 'r')
data = fi.read()
fi.close()
fo = open('irma_oneday_06_new.csv', 'w')
fo.write(data.replace('\x00', ''))
fo.close()


# In[8]:


temp_country = []
csvReader = csv.reader(codecs.open('irma_oneday_06_new.csv', 'rU', 'utf-8'))
for i,row in enumerate(csvReader):
    location = ""
    if i>0:
        if len(row) > 10:
            temp_country.append(row[-1])
    


# In[11]:


unique_locations = set(temp_country)


# In[13]:


unique_list = list(unique_locations)


# In[15]:


country = dict()


# In[17]:


gn = Nominatim(timeout=30)


# In[36]:


for i,row in enumerate(unique_list):
    try:
        location = ""
        temp_country = []
        if unique_list[i] not in country.keys():
            location = ""
            location = unique_list[i]
            if str(location) == '':
                country[location] = 'None'
            else:
                loc = gn.geocode(location,language='en',timeout=30)
                temp_country=str(loc).split(",")
                #print(temp_country)
                country[location] = temp_country[-1]
                #country.append(temp_country[-1])
                time.sleep(1)
                print(i)
        else:
            print(str(i) + "Passed")
            pass
    except:
        country[unique_list[i]] = 'None'
        print("Exception" + str(i))


# In[38]:


with open('country_dict.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in country.items():
        writer.writerow([key, value])


# In[ ]:




