#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import itertools


# In[2]:


df_tweets = pd.read_csv("/Volumes/GoogleDrive/My Drive/Solidarity_analysis/RVisTutorial-master/sashank_viz/irma_analysis_files/irma/processed/irma_text_convverted.csv")
df_emojis = pd.read_csv("/Volumes/GoogleDrive/My Drive/Solidarity_analysis/RVisTutorial-master/sashank_viz/irma_analysis_files/irma/processed/emoji_dict.csv")


# In[4]:


df_emojis = df_emojis.rename(columns={'r.encoding': 'encoding'})


# In[6]:


emoji_dict = dict(zip(df_emojis.encoding,df_emojis.description))


# In[7]:


emoji_dict


# In[8]:


encoding = df_emojis['encoding'].tolist()


# In[10]:


tweets = df_tweets['text'].tolist()


# In[13]:


final_emojis = []
for i,tweet in enumerate(tweets):
    if i % 10000 == 0:
        print(i)
    emojis = []
    for enc in encoding:
        matchList = re.findall(enc, tweet, flags=0)
        if len(matchList) == 0:
            pass
        else:
            emojis.append(matchList)
    final_emojis.append(emojis)


# In[17]:


emojis = list(itertools.chain.from_iterable(final_emojis))


# In[14]:


count = []
emoji_desc = []


# In[15]:


for em in final_emojis:
    if len(em) == 0:
        count.append(0)
        emoji_desc.append('None')
    else:
        string =""
        for emoji in em:
            count.append(len(emoji))
            for e in emoji:
                
                desc = emoji_dict[e]
                string += desc + "%"
        emoji_desc.append(string)
            


# In[ ]:


#df_tweets['count_raw'] = pd.DataFrame(count)
df_tweets['description_raw'] = pd.DataFrame(emoji_desc)


# In[17]:


emoji_split = []
for emoji in emoji_desc:
    emoji_split.append(emoji.split("%"))


# In[18]:


final_emoji_processed = []
for em in emoji_split:
    current_emoji = []
    for dec in em:
        if dec != '':
            current_emoji.append(dec)
    final_emoji_processed.append(current_emoji)


# In[19]:


count = []
for em in final_emoji_processed:
    if len(em) > 1:
        count.append(len(em))
    else:
        for emo in em:
            if emo == 'None':
                count.append(0)
            else:
                count.append(1)
        


# In[20]:


df_tweets['count_raw'] = pd.DataFrame(count)


# In[21]:


final_emoji_processed[count.index(6)]


# In[22]:


text = df_tweets['text'].tolist()
labels =df_tweets['labels'].tolist()
post_time = df_tweets['posted_time'].tolist()
locations = df_tweets['locations'].tolist()
lat = df_tweets['latitude'].tolist()
long = df_tweets['longitude'].tolist()


# In[23]:


from collections import Counter
z = Counter(final_emoji_processed[0])
len(z)


# In[24]:


for k,v in z.items():
    print(k,v)


# # make the duplicates - Sometimes a single tweet might contain more than one type of emoji. This created copies of the tweet with different emojis referenced to it

# In[25]:


new_text = []
new_labels = []
new_post_time = []
new_locations = []
new_lat = []
new_long = []
new_count = []
new_description = []


# In[26]:


for i in range(len(text)):
    key_list = Counter(final_emoji_processed[i])
    if len(key_list) == 1:
        new_text.append(text[i])
        new_labels.append(labels[i])
        new_post_time.append(post_time[i])
        new_locations.append(locations[i])
        new_lat.append(lat[i])
        new_long.append(long[i])
        for k,v in key_list.items():
            new_description.append(k)
            new_count.append(v)
    else:
        for j in range(len(key_list)):
            new_text.append(text[i])
            new_labels.append(labels[i])
            new_post_time.append(post_time[i])
            new_locations.append(locations[i])
            new_lat.append(lat[i])
            new_long.append(long[i])
        for k,v in key_list.items():
            new_description.append(k)
            new_count.append(v)


# In[29]:


df_dup = pd.DataFrame(index=range(len(new_text)))


# In[30]:


df_dup['text'] = pd.DataFrame(new_text)
df_dup['labels'] = pd.DataFrame(new_labels)
df_dup['posted_time'] = pd.DataFrame(new_post_time)
df_dup['locations'] = pd.DataFrame(new_locations)
df_dup['latitude'] = pd.DataFrame(new_lat)
df_dup['longitude'] = pd.DataFrame(new_long)
df_dup['count'] = pd.DataFrame(new_count)
df_dup['description'] = pd.DataFrame(new_description)


# In[31]:


df_dup.to_csv('duplicates_irma_timeseries.csv',index=False)


# In[ ]:




