
# coding: utf-8

## Домашнее задание 1. Сбор данных

# При решении реальных задач мы почти никогда не имеем дело с "хорошими" исходными данными, уже подготовленными для обработки и анализа. Как правило на входе имеются неструкткрированные данные в "грязном" виде, например необработанные тексты, изображения или аудио файлы. Иногда нет даже этого, и данные приходится собирать из разных доступных источников: разнообразных баз данных, внешних сервисов и даже электронных таблиц. После того, как данные получены, их унифицируют, очищают от шума, преобразовывают в нужный вид и сохраняют для дальнейшего анализа. В одном из традиционных подходов к [Data Mining](http://www.wikiwand.com/en/Online_analytical_processing) этот процесс называется Extract-Transform-Load ([ETL](http://www.wikiwand.com/en/Extract,_transform,_load)).
# 
# Цель этого задания собрать первые данные о пользователях из обучающей выборки и провести простейший качественный анализ. В ходе решения будут использованы:
# 1. [numpy](http://www.numpy.org/) -- библиотека для работы с многомерными массивами
# 2. [pandas](http://pandas.pydata.org/) -- библиотека, позволяющая удобно работать с различными типами данных
# 3. [requests](http://docs.python-requests.org/en/latest/) -- библиотека, которую можно использовать для вызова HTTP запросов
# 4. [python-twitter](https://github.com/bear/python-twitter/tree/master/twitter) -- обертка для Twitter API
# 5. [matplotlib](http://matplotlib.org/) -- библиотека для рисования графиков в python
# 
# Первым делом импортируем необходимые библиотеки и убеждаемся, что все установлено.

# In[1]:

import pandas as pd
import numpy as np
import pylab as pl
import mpl_toolkits.basemap as bm
import twitter
import requests
import json
import datetime
import dateutil
import csv
from requests_oauthlib import OAuth1

# Plotting config
get_ipython().magic(u'pylab inline')


### Чтение исходных данных из файла

# Считываем исходные данные из файла в data frame библиотеки pandas. Полученный data frame должен иметь целочисленный ключ и две колонки:
# 1. user_id -- идентификатор пользователя
# 2. class -- числовой номер класса

# In[2]:

TRAINING_SET_URL = "https://kaggle2.blob.core.windows.net/competitions-data/inclass/4277/twitter_train.txt?sv=2012-02-12&se=2015-03-04T15%3A54%3A20Z&sr=b&sp=r&sig=RKsea6k%2F%2BgauZoZdAtahEsj%2FAJDBKZbttSu7xQdpvA8%3D" # insert file path here
df_users = pd.read_csv(TRAINING_SET_URL, sep=",", header=0, names=["user_id", "class"])
df_users.head()


# Построим распределение целевой переменной. Требуется получить [barchart](http://www.wikiwand.com/en/Bar_chart), в котором высота столбика, соответствующего каждому из классов, пропорциональна количеству пользователей этого класса. По горизонтальной оси отложены классы (positive, negative), а по вертикальной -- количество пользователей.

# In[3]:

# Compute the distribution of the target variable
counts, bins = np.histogram(df_users["class"], bins=[0,1,2])

# Plot the distribution
pl.figure(figsize=(6,6))
pl.bar(bins[:-1], counts, width=0.5, alpha=0.4)
pl.xticks(bins[:-1] + 0.3, ["negative", "positive"])
pl.xlim(bins[0] - 0.5, bins[-1])
pl.ylabel("Number of users")
pl.title("Target variable distribution")
pl.show()


### Сбор данных

# Для того чтобы разработать модель, которая предсказывает значение целевой переменной для произвольного пользователя, недостаточно знать только значения идентификаторов пользоватей. Причина в том, что _user_id_ для пользователя никак не зависит от целевой переменной -- идентификатор генерируется на основании времени регистрации, сервера, обрабатывающего запрос, и номера пользователя ([подробности](https://dev.twitter.com/overview/api/twitter-ids-json-and-snowflake)).
# 
# Поэтому нам потребуется загрузить дополнительную информацию о каждом пользователе, иначе говоря провести сбор данных (data collection). Наиболее важную информацию можно загрузить из [Twitter](https://dev.twitter.com/rest/public) [API](http://www.wikiwand.com/en/Representational_state_transfer). При желании можно воспользоваться и другими источниками -- об этом ниже.
# 
# Для того, чтобы получить доступ к API прежде всего необходимо зарегистрироваться в Twitter в качестве разработчика и создать свое [приложение](https://apps.twitter.com/). После создания приложения будет доступен набор ключей, которые мы будем использовать для аутентификации. Эти ключи необходимо скопировать в соответствующие константы ниже. Подробнее о том, как работает аутентификация в Twitter API можно почитать [по ссылке](https://dev.twitter.com/oauth/application-only), хотя это нужно скорее для ознакомления: библиотека обращения с API позаботится о механизме аутентификации за нас.

# In[4]:

CONSUMER_KEY = "obfD8QyAByoGZRjgd1GeSKdUC"
CONSUMER_SECRET = "zqup2Sk9Sb5DWm7q8swBgXaVeW1TncacBb0hfPbywyNDmknAcH"

ACCESS_TOKEN_KEY = "103516608-CJ63nEox8ItY9DgDhYU2TBFeLnsZvYUUxslB1d3e"
ACCESS_TOKEN_SECRET = "65XCfoYcZFUbYRPyNfV5Tso2f8Kv1yi2LYF3QV2xcloFP"

"""
api = twitter.Api(consumer_key=CONSUMER_KEY, 
                  consumer_secret=CONSUMER_SECRET, 
                  access_token_key=ACCESS_TOKEN_KEY, 
                  access_token_secret=ACCESS_TOKEN_SECRET)
"""


# Twitter API предоставляет информацию о местонахождении пользователя, но эта информация представлена в текстовом виде, например так:
# ```
# "location": "San Francisco, CA"
# ```
# Такие текстовый описания не слишком удобны для анализа, поэтому наша цель -- получить более структурированную информацию, такую как географические координаты, страна, город и т.д. Для этого удобно использовать геоинформационный сервис, например [GeoNames](http://www.geonames.org/export/web-services.html). Для его использования также необходимо зарегистрироваться, подтвердить регистрацию и включить поддержку API. После этого можно будет посылать запросы для нахождения нужной информации. Например на запрос
# ```
# http://api.geonames.org/search?q="San Francisco, CA"&maxRows=10&username=demo
# ```
# возвращается результат,
# ```javascript
# {
#     "totalResultsCount": 2112,
#     "geonames": [
#         {
#             "countryId": "6252001",
#             "adminCode1": "CA",
#             "countryName": "United States",
#             "fclName": "city, village,...",
#             "countryCode": "US",
#             "lng": "-122.41942",
#             "fcodeName": "seat of a second-order administrative division",
#             "toponymName": "San Francisco",
#             "fcl": "P",
#             "name": "San Francisco",
#             "fcode": "PPLA2",
#             "geonameId": 5391959,
#             "lat": "37.77493",
#             "adminName1": "California",
#             "population": 805235
#         },
#         {
#             "countryId": "6252001",
#             "adminCode1": "CA",
#             "countryName": "United States",
#             "fclName": "spot, building, farm",
#             "countryCode": "US",
#             "lng": "-122.3758",
#             "fcodeName": "airport",
#             "toponymName": "San Francisco International Airport",
#             "fcl": "S",
#             "name": "San Francisco International Airport",
#             "fcode": "AIRP",
#             "geonameId": 5391989,
#             "lat": "37.61882",
#             "adminName1": "California",
#             "population": 0
#         }
#     ]
# }
# ```
# из которого легко извлечь нужную информацию.
# 
# **Замечание: для запросов необходимо использовать своего пользователя. Кроме того количество запросов ограничего 30к в день**.
# 
# Первым делом нам понадобится функция, которая возвращает информацию о местоположении для данного текстового запроса. Требуется реализовать функцию `get_coordinates_by_location`, принимающую на вход строку с местоположением и возвращает кортеж вида (долгота, широта, город).

# In[5]:

# Input your user name
GEO_USER_NAME = "marinae"

def get_coordinates_by_location(location):
    # get data from GeoNames
    url = "http://api.geonames.org/searchJSON?q=" + location + "&maxRows=1&username=" + GEO_USER_NAME
    r = requests.get(url)
    data = json.loads(r.text)
    
    # check result
    if 'totalResultsCount' in data:
        if data['totalResultsCount'] > 0:
            # city exists
            i = data['geonames'][0]
            # return its coordinates and country
            return (i['lat'], i['lng'], i['name'])
        else:
            return (0, 0, u'')
    else:
        return (0, 0, u'')


# Следующий шаг -- вызов Twitter API для сбора данных и сохранения их в data frame. После чего data frame c собранными данными совмещается с data frame, содержащим данные исходной обучающей выборки. 
# 
# В этой части задания нужно реализовать функцию `get_user_records`, которая принимает на вход прочитанный из файла `data frame` и возвращает список словарей, каждый из которых представляет данные одного пользователя. Для того, чтобы получить из объекта класса [`User`](https://github.com/bear/python-twitter/blob/master/twitter/user.py) словарь в правильном формате, нужно использовать функцию `twitter_user_to_dataframe_record` (5 баллов).

# In[6]:

ts_parser = lambda date_str: dateutil.parser.parse(date_str) if pd.notnull(date_str) else None

def twitter_user_to_dataframe_record(user):
    record = {
        "user_id": user['id'],
        "name": user['name'],
        "screen_name": user['screen_name'],        
        "created_at": ts_parser(user['created_at']),        
        "followers_count": user['followers_count'],
        "friends_count": user['friends_count'],
        "statuses_count": user['statuses_count'],
        "favourites_count": user['favourites_count'],
        "listed_count": user['listed_count'],
        "verified": user['verified']
    }
    
    if user['description'] is not None and user['description'].strip() != "":
        record["description"] = user['description']
        
    if user['location'] is not None and user['location'].strip() != "":
        record["location"] = user['location']
        record["lat"], record["lon"], record["country"] = get_coordinates_by_location(user['location'])
    
    return record

def get_user_records(df):
    user_records = []
    count = len(df['user_id'])
    lower = 0
    upper = 100
    
    # OAuth authorization keys and tokens
    auth = OAuth1(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)
    
    while lower < count:
        # next 100 users
        part_records = df['user_id'][lower:upper]
        
        # construct url
        url = "https://api.twitter.com/1.1/users/lookup.json?user_id="
        for i in part_records:
            url = url + str(i) + "," 
        url = url[:-1]
        
        # send request
        r = requests.post(url, auth=auth)
        data = json.loads(r.text)
        
        # convert response to records
        for i in data:
            record = twitter_user_to_dataframe_record(i)
            user_records.append(record)
        
        # increment
        lower = upper
        upper = upper + 100
  
    return user_records

user_records = get_user_records(df_users)
        
print "Creating data frame from loaded data"
df_records = pd.DataFrame(user_records, columns=["user_id", "name", "screen_name", "description", "verified", "location", "lat", "lon", "country", "created_at", "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count"])
print "Merging data frame with the training set"
df_full = pd.merge(df_users, df_records, on="user_id", how="left")
print "Finished building data frame"


### Exploratory Data Analysis

# Для того, чтобы лучше понять, как устроена наша обучающая выборка, построим несколько графиков. Сначала построим долю "положительных" пользователей в зависимости от дня создания аккаунта. По горизонтальной оси отложим день создания аккаунта, а по вертикальной -- долю "положительных" пользователей ([подсказка](http://anokhin.github.io/img/hw1_distr.png)). Необходимо дописать код функции count_users. В функции необходимо посчитать пользователей в каждой группе (2 балла).

# In[27]:

def count_users(grouped):
    """
    Counts number of positive and negative users
    created at each date.
    
    Returns:
        count_pos -- 1D numpy array with the counts of positive users created at each date
        count_neg -- 1D numpy array with the counts of negative users created at each date
        dts -- a list of date strings, e.g. ['2014-10-11', '2014-10-12', ...]
    """
    dts, count_pos, count_neg = [], [], []

    for i in grouped:
        # number of accounts created at this month
        count_all = len(i[1]['class'])
        
        # number of positive accounts
        pos = len(np.nonzero(i[1]['class'])[0])
        
        # number of negative accounts
        neg = count_all - pos
        
        count_pos.append(pos)
        count_neg.append(neg)
        
        # month and year
        dts.append(i[0])

    return np.array(count_pos), np.array(count_neg), dts


grouped = df_full.groupby(map(lambda dt: dt.strftime("%Y-%m") if pd.notnull(dt) else "NA", df_full["created_at"]))
count_pos, count_neg, dts = count_users(grouped)
    
fraction_pos = count_pos / (count_pos + count_neg + 1e-10)
fraction_neg = 1 - fraction_pos

sort_ind = np.argsort(dts)
    
pl.figure(figsize=(20, 3))
pl.bar(np.arange(len(dts)), fraction_pos[sort_ind], width=1.0, color='red', alpha=0.6, linewidth=0, label="Positive")
pl.bar(np.arange(len(dts)), fraction_neg[sort_ind], bottom=fraction_pos[sort_ind], width=1.0, color='green', alpha=0.6, linewidth=0, label="Negative")
pl.xticks(np.arange(len(dts)) + 0.4, sorted(dts), rotation=90)
pl.title("Class distribution by account creation month")
pl.xlim(0, len(dts))
pl.legend()
pl.show()


# Видно, что доля "положительных" аккаунтов в выборке растет с увеличением времени. Посмотрим, где живут пользователи положительной и отрицательной категории. Для этого отметим на карте каждого положительного пользователя красным, а отрицательного -- зеленым цветом ([подсказка](http://anokhin.github.io/img/hw1_map.png)). Необходимо реализовать функцию plot_points_on_map. В функции необходимо отобразить на карте пользователей из разных классов (3 балла).

# In[113]:

pl.figure(figsize=(20,12))

m = bm.Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')

m.drawcountries(linewidth=0.2)
m.fillcontinents(color='lavender', lake_color='#000040')
m.drawmapboundary(linewidth=0.2, fill_color='#000040')
m.drawparallels(np.arange(-90,90,30),labels=[0,0,0,0], color='white', linewidth=0.5)
m.drawmeridians(np.arange(0,360,30),labels=[0,0,0,0], color='white', linewidth=0.5)

def plot_points_on_map(df_full):
    """
    Plot points on the map. Be creative.
    """
    
    # leave only necessary data
    cutted = pd.DataFrame(df_full, columns=['class', 'lat', 'lon'])
    
    # check if longitude (latitude as well) is not a number
    filtered = cutted[cutted.lon == cutted.lon]
    
    # remove records with lon = 0 & lat = 0
    filtered_nn = filtered[(filtered.lon != 0) & (filtered.lat != 0)]
    
    # group filtered dataset by coordinates
    grouped = filtered_nn.groupby([filtered_nn['lat'], filtered_nn['lon']])
    
    for i in grouped['lat', 'lon']:
        # count positive users from this area
        positive = len(np.nonzero(i[1]['class'])[0])
        
        # count negative users from this area
        negative = len(i[1]['class']) - positive
        
        lat = i[0][0]
        lon = i[0][1]
        
        # draw circle
        if positive > negative:
            m.plot(lon, lat, 'ro', markersize=6)
        else:
            m.plot(lon, lat, 'go', markersize=6)
    return

plot_points_on_map(df_full)

pl.title("Geospatial distribution of twitter users")
pl.legend()
pl.show()


# В последней картинке есть проблема: сервис геоинформации определяет координаты с точностью до города, поэтому точки, соответствующте нескольким пользователям, могут накладываться. Предложите и реализуйте способ, позволяющий справиться с этой проблемой.
# 
# Смотрим на полученный data frame и сохраняем его в .csv файл.

# In[114]:

OUT_FILE_PATH = "files/hw1_out.csv"
print "Saving output data frame to %s" % OUT_FILE_PATH
df_full.to_csv(OUT_FILE_PATH, sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
df_full.head()


# In[9]:



