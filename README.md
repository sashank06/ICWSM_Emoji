# ICWSM_Emoji

This repository contains the dataset for the ICWSM emoji paper titled [I stand with you: Using Emojis to study solidarity in crisis events](http://ceur-ws.org/Vol-2130/paper1.pdf) published at ICWSM Emoji workshop, 2018, Stanford, USA.

**Two large datasets made available**
```
Due to Twitter's restrictions, we are only able to share the tweet ID's
1. **Paris Novemer Attack** - Tweet Id's are available in the paris_dataset folder
2. **Irma Hurricane Dataset** - The files 06.txt to 12.txt contain the tweets ids of the english tweets of hurricane irma collected between Sept 6th 2017 to Sept 12 2017.
```

This repository contains a mixture of codebase written in R (predominantly) and Python.

Please download the tweets with the help of the tweet IDs using this [repository](https://github.com/sashank06/tweets_extraction) in Python or use the rtweet library for R.
```
**Parsing JSON dataset**
parsing_json.Rmd -> Extracting tweets and necessrary fields from JSON.
Depends on the annotation made available in the annotations folder to further pre-processs the dataset.
```

**Steps needed to produce the figures in the paper**
1. Prepare the excel file with required fields
```text (tweet), labels(-1 or 1), postedtime, locations, latitude, longitude, count (count of emojis in a tweet), description(type of emoji), country, community (affected or not), estTime (converted time), day and hour```

2. [Producing Time Series](https://github.com/sashank06/ICWSM_Emoji/blob/master/creating_irma_time_series_file.py) - this files gives an overview of how to produce the file that is needed for counting emojis and creating a time series file

3. [Geo-tagging](https://github.com/sashank06/ICWSM_Emoji/blob/master/geotagging_irma.py) sample code is made availble for the IRMA dataset. This helps give the location mentioned in the tweets with a latitude, longitude and country.

**Reproducing Co-occurence Network**
1. Run the [irma_tweets_emoji_analysis.Rmd](https://github.com/sashank06/ICWSM_Emoji/blob/master/Cooccurrence-network/irma_tweets_emoji_analysis.Rmd) and [paris_tweets_emoji_analysis.Rmd](https://github.com/sashank06/ICWSM_Emoji/blob/master/Cooccurrence-network/paris_tweets_emoji_analysis.Rmd) with the prepared dataset.

Should produce these figures
Irma Network outside US           |  Irma Network inside US 
:-------------------------:|:-------------------------:
![Irma Network outside US](https://github.com/sashank06/ICWSM_Emoji/blob/master/images/emoji_solidarity_irma_network_NotUS.png)  |  ![Irma Network inside US ](https://github.com/sashank06/ICWSM_Emoji/blob/master/images/emoji_solidarity_irma_network_US.png)

Please contact ssantha1@uncc.edu if you have any further questions.
