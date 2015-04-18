
# coding: utf-8

## Домашнее задание 4. Конструирование текстовых признаков из твитов пользователей

### Сбор данных

# Первый этап - сбор твитов пользователей. Необходимо подключаться к Twitter API и запрашивать твиты по id пользователя. 
# Подключение к API подробно описано в ДЗ 1.

# In[1]:

import twitter
import nltk
import json
import pickle
import re
import time
import numpy as np
import pandas as pd
import pygame

from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from pytagcloud import create_tag_image, make_tags

CONSUMER_KEY = "obfD8QyAByoGZRjgd1GeSKdUC"
CONSUMER_SECRET = "zqup2Sk9Sb5DWm7q8swBgXaVeW1TncacBb0hfPbywyNDmknAcH"

ACCESS_TOKEN_KEY = "103516608-CJ63nEox8ItY9DgDhYU2TBFeLnsZvYUUxslB1d3e"
ACCESS_TOKEN_SECRET = "65XCfoYcZFUbYRPyNfV5Tso2f8Kv1yi2LYF3QV2xcloFP"

TRAINING_SET_URL = "https://kaggle2.blob.core.windows.net/competitions-data/inclass/4277/twitter_train.txt?sv=2012-02-12&se=2015-04-19T13%3A31%3A49Z&sr=b&sp=r&sig=%2F37RhYx6edDa3cTP%2FRSdEY4bnIG1VPWR72tmrhipE6g%3D"

api = twitter.Api(consumer_key=CONSUMER_KEY, 
                  consumer_secret=CONSUMER_SECRET, 
                  access_token_key=ACCESS_TOKEN_KEY, 
                  access_token_secret=ACCESS_TOKEN_SECRET)


# In[2]:

df_users = pd.read_csv(TRAINING_SET_URL, sep=",", header=0, names=["user_id", "class"])


# Для получения твитов пользователя может быть использован метод GetUserTimeline из библиотеки python-twitter. Он позволяет получить не более 200 твитов пользователя.
# 
# Метод имеет ограничение по количеству запросов в секунду. Для получения информации о промежутке времени, которое необходимо подождать для повторного обращения к API может быть использован метод `GetSleepTime`. Для получения информации об ограничениях запросов с помощью метода `GetUserTimeLine` необходимо вызывать `GetSleepTime` с параметром "statuses/user_timeline".
# 
# Метод GetUserTimeline возвращает объекты типа Status. У этих объектов есть метод AsDict, который позволяет представить твит в виде словаря.
# 
# Id пользователей необходимо считать из файла, как было сделано в ДЗ 1.
# 
# Необходимо реализовать функцию `get_user_tweets(user_id)`. Входной параметр - id пользователя из файла. Возвращаемое значение - массив твитов пользователя, где каждый твит представлен в виде словаря. Предполагается, что информация о пользователе содержится в твитах, которые пользователь написал сам. Это означает, что можно попробовать отфильтровать ответы другим пользователям, ссылки и ретвиты, а так же картинки и видео, так как наша цель - найти текстовую информацию.

# In[3]:

def get_user_tweets(user_id):
    # returns list of tweets (without retweets)
    
    try:
        status = api.GetUserTimeline(user_id, count=200, trim_user=1, include_rts=0)
        tweets_dict = [status[i].AsDict() for i in xrange(len(status))]
        status = [tweets_dict[i]["text"] for i in xrange(len(tweets_dict))]
    except twitter.TwitterError:
        status = []
    
    return status


# In[4]:

"""
tweets = [None] * len(df_users["user_id"])
file_name = "raw_tweets"
file_ext  = ".txt"

for i, user_id in enumerate(df_users["user_id"]):
    
    try:
        sleep_time = api.GetSleepTime("statuses/user_timeline")
    except twitter.TwitterError:
        sleep_time = 900
        
    if sleep_time > 0:
        print sleep_time + 1
        time.sleep(sleep_time + 1)
        print "Wake up!"
    
    tweets[i] = get_user_tweets(user_id)
    
    if i % 100 == 0:
        print i, " users loaded..."
        file_path = file_name + str(i) + file_ext
        pickle.dump(tweets, open(file_path, "wb"))
        
pickle.dump(tweets, open("../files/raw_tweets_full.txt", "wb"))
"""

tweets = pickle.loads(open("../files/raw_tweets_full.txt", "rb").read())


### Разбор текста твита

# Обработка текста предполагает разбиение текста на отдельные элементы - параграфы, предложения, слова. Мы будем преобразовывать текст твита к словам. Для этого текст необходимо разбить на слова. Сделать это можно, например, с помощью регулярного выражения.
# 
# Необходимо реализовать функцию, `get_words(text)`. Входной параметр - строка с текстом. Возвращаемое значение - массив строк (слов). Обратите внимание, что нужно учесть возможное наличие пунктуации и выделить по возможности только слова. 

# In[5]:

def get_words(text):
    # Returns list of words from text
    
    return text.split()

print get_words("Here are different words!")


# Далее полученные слова необходимо привести к нормальной форме. То есть привести их к форме единственного числа настоящего времени и пр. Сделать это можно с помощью библиотеки nltk. Информацию по загрузке, установке библиотеки и примерах использования можно найти на сайте http://www.nltk.org/
# 
# Для загрузки всех необходимых словарей можно воспользоваться методом download из библиотеки nltk.

# Для дальнейшей обработки слова должны быть приведены к нижнему регистру. 
# 
# Для приведения к нормальной форме можно использовать `WordNetLemmatizer` из библиотеки nltk. У этого класса есть метод `lemmatize`.
# 
# Также необходимо убрать из текста так называемые стоп-слова. Это часто используемые слова, не несущие смысловой нагрузки для наших задач. Сделать это можно с помощью `stopwords` из nltk.corpus

# Необходимо реализовать функцию `get_tokens(words)`. Входной параметр - массив слов. Возвращаемое значение - массив токенов.

# In[6]:

def get_tokens(text):
    # Returns list of tokens

    words = get_words(text)
    words = [re.sub(r'(^[\'\-]+|[\'\-]+$)', '', w) for w in words]

    stopwords = nltk.corpus.stopwords.words('english')

    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(w).lower() for w in words]
    tokens_nonstop = [t for t in tokens if t not in stopwords]

    return tokens_nonstop

print get_tokens("here are different words")


# Необходимо реализовать функцию `get_tweet_tokens(tweet)`. Входной параметр - текст твита. Возвращаемое значение -- токены твита. 

# In[7]:

def get_tweet_tokens(tweet):
    # Returns list of tweet tokens

    tweet = tweet.encode('ascii', 'ignore')               # non-ascii characters
    tweet = re.sub(r'\bhttps?://\S*\b',       ' ', tweet) # urls
    tweet = re.sub(r'@\b\w+\b',               ' ', tweet) # replies
    tweet = re.sub(r'&amp;',                  ' ', tweet) # &
    tweet = re.sub(r'[^\w\-\' ]',             ' ', tweet) # all symbols except ' and -
    tweet = re.sub(r'\b\d+\b' ,               ' ', tweet) # numbers
    tweet = re.sub(r'\s\-+\s',                ' ', tweet) # separate symbols -
    tweet = re.sub(r'\s\'+\s',                ' ', tweet) # separate symbols '

    return get_tokens(tweet)


# Необходимо реализовать функцию `collect_users_tokens()`. Функция должна сконструировать матрицу признаков пользователей. В этой матрице строка - пользователь. Столбец - токен. На пересечении - сколько раз токен встречается у пользователя.
# Для построения матрицы можно использовать `DictVectorizer` из `sklearn.feature_extraction`.

# In[8]:

def get_user_dict(user_tweets):
    # Returns dictionary of single user

    tokens = [get_tweet_tokens(t) for t in user_tweets]
    all_tokens = [word for i in tokens for word in i if word != '']
    user_dict = dict.fromkeys(all_tokens)

    for key in user_dict:
        user_dict[key] = all_tokens.count(key)

    return user_dict


# In[9]:

def collect_users_tokens(df_users, tweets):
    # Returns users list and list of user dicts. Each dict contains frequence of user tokens

    return df_users["user_id"], [get_user_dict(user_tweets) for user_tweets in tweets]


# Сохраним полученные данные в файл. Используется метод savez из numpy.

# In[17]:

"""
users, users_tokens = collect_users_tokens(df_users, tweets)
v = DictVectorizer()
vs = v.fit_transform(users_tokens)

np.savez("../files/out_4.dat", data=vs, users=users, users_tokens=users_tokens)

features = v.get_feature_names()
vs_count = np.array(vs.sum(axis=0)).flatten()

tag_counts = [None] * len(features)

for i, f in enumerate(features):
    tag_counts[i] = [f, vs_count[i]]

pickle.dump(tag_counts, open("../files/tag_counts.txt", "wb"))
"""

tag_counts = pickle.loads(open("../files/tag_counts.txt", "rb").read())


# Далее для получения представления о полученной информацию о токенах предлагается отобразить ее в виде облака тэгов. [Подсказка](http://anokhin.github.io/img/tag_cloud.png). 

# In[33]:

def draw_tag_cloud(tag_counts, num_tags=100, maxsize=100):
    # Draws tag cloud of found tokens
    
    counts = np.empty(len(tag_counts))

    for i in xrange(len(tag_counts)):
        counts[i] = tag_counts[i][1]

    num_tags = 100
    popular = np.argsort(-counts)[:num_tags]
    popular_tags = [[tag_counts[i][0], tag_counts[i][1]] for i in popular]

    tags = make_tags(popular_tags, maxsize=maxsize)
    create_tag_image(tags, 'cloud_large.png', size=(900, 600), fontname='Lobster')

    return


# In[35]:

draw_tag_cloud(tag_counts, 150, 100)

