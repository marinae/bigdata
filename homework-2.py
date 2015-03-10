
# coding: utf-8

## Домашнее задание 2. Преобразование данных

# Цель этого задания -- преобразовать имеющиеся атрибуты пользователей в признаки так, чтобы полученная матрица признаков была пригодна для подачи в алгоритм кластеризации. Этап конструирования признаков -- самый важный и обычно самый долгий. К нему возвращаются много раз на протяжении решения задачи анализа данных.
# 
# Кроме библиотек, использованных в первом задании, нам понадобятся следующие библиотеки:
# 1. [scikit-learn](http://scikit-learn.org/stable/) -- библиотека, реализующая множество алгоритмов машинного обучения и сопутствующих алгоритмов

# In[1]:

import pandas as pd
import numpy as np
import pylab as pl
import sklearn.preprocessing as sp
import csv
import re
import dateutil

np.set_printoptions(linewidth=150, precision=3, suppress=True)

# Plotting config
get_ipython().magic(u'pylab inline')


# In[2]:

ts_parser = lambda date_str: dateutil.parser.parse(date_str) if pd.notnull(date_str) else None
df_users = pd.read_csv("files/hw1_out.csv", sep=";", encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC, converters={"created_at": ts_parser})

# Remove rows with users not found
df_users = df_users[pd.notnull(df_users['name'])]
df_users["lat"].fillna(value=0, inplace=True)
df_users["lon"].fillna(value=0, inplace=True)


# Далее необходимо ввести новые признаки. Для каждого пользователя предлагается ввести следующие признаки:
# - name_words - количество слов в имени
# - screen_name_length - количество символов в псевдониме
# - description_length - длина описания
# - created_year - год создания аккаунта
# - country_code - код страны
# - verified - предлагается перевести в тип int

# In[3]:

def create_new_features(df_users, features):
    
    # Introduce new features
    new_features = ["name_words", "screen_name_length", "description_length", "created_year", "country_code", "verified"]
    
    # rename column "verified"
    df_users = df_users.rename(columns = {"verified": "verified_old"})
    
    # initialize empty columns
    name_words = []
    screen_name_length = []
    description_length = []
    created_year = []
    country_code = []
    verified = []
    
    # array for counting countries
    countries = []
    
    for index, row in df_users.iterrows():
        
        # count number of words in "name"
        name_words.append(len(row["name"].split()))
        
        # count screen name length
        screen_name_length.append(len(row["screen_name"]))
        
        # count description length
        if row["description"] == row["description"]:
            description_length.append(len(row["description"]))
        else:
            description_length.append(0)
            
        # find year in which account was created
        created_year.append(int(str(row["created_at"])[0:4]))
        
        # find country code
        if row["country"] == row["country"]:
            # new country ?
            if row["country"] not in countries:
                countries.append(row["country"])
            # zero is for NaN coutry
            country_code.append(countries.index(row["country"]) + 1)
        else:
            country_code.append(0)
        
        # find if account is verified
        if row["verified_old"] == True:
            verified.append(1)
        else:
            verified.append(0)
        
    # insert new columns
    df_users.insert(len(df_users.columns), "name_words", name_words)
    df_users.insert(len(df_users.columns), "screen_name_length", screen_name_length)
    df_users.insert(len(df_users.columns), "description_length", description_length)
    df_users.insert(len(df_users.columns), "created_year", created_year)
    df_users.insert(len(df_users.columns), "country_code", country_code)
    df_users.insert(len(df_users.columns), "verified", verified)
    df_users = df_users.drop("verified_old", axis=1)
    
    # concatenate features
    features = features + new_features
    
    return df_users, features


# In[4]:

features = ["class", "lat", "lon", "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count"]
df_users, features = create_new_features(df_users, features)

# save current result
df_users.to_csv("files/new_features.csv", sep=";", encoding="utf-8")

x = df_users[features].values
y = df_users["class"].values


# Посмотрим, являются ли какие-либо из выбранных признаков сильно скоррелированными. Для этого посчитаем матрицу корреляций и выберем те пары признаков, фбсолютное значения коэффициента корреляции между которыми больше 0.2. Необходимо реализовать функцию find_correlated_features, в которой нужно рассчитать коэффициенты корелляции и вывести те, которые больше 0.2. Подсказка: предлагается найти необходимую функцию в библиотеке np и реализовать find_correlated_features с использованием не более 5 строк кода.

# In[5]:

def find_correlated_features(x, features):
    
    # find correlation matrix
    corr_matrix = np.corrcoef(x)
    
    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):
            if i < j and corr_matrix[i][j] > 0.5:
                print "Correlated features: %s + %s -> %.2f" % (feature_i, feature_j, corr_matrix[i][j])
    
find_correlated_features(x, features)


# Выделилось 3 группы признаков:
# 1. Основанные на географии:  "lat", "lon", "country_code"
# 2. Основанные на социальной активности:  "verified", "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count", "created_year"
# 3. Остальные:  "name_words", "screen_name_length", "description_length"
# 
# Построим взаимные распределения пар признаков в каждой из групп, а также гистограмму значений каждого из признаков с учетом целевой переменной.

                Необходимо реалищовать функции: plot_two_features_scatter для построения взаимного распределения пары признаков, plot_feature_histogram для построения гистограммы значений, plot_dataset для построения набора графиков по разным парам признаков.
                
# In[6]:

def plot_two_features_scatter(x_i, x_j, y, ax):
    
    # divide arrays by classes
    pos_i = []
    pos_j = []
    neg_i = []
    neg_j = []
    
    # iterating through indices
    for i in xrange(len(x_i)):
        if y[i] == 1:
            pos_i.append(x_i[i])
            pos_j.append(x_j[i])
        else:
            neg_i.append(x_i[i])
            neg_j.append(x_j[i])
    
    ax.scatter(pos_i, pos_j, color='red', s=2, alpha=0.3)
    ax.scatter(neg_i, neg_j, color='green', s=2, alpha=0.3) 
    # set axis to empty
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
  
def plot_feature_histogram(x_i, y, ax):
    
    # divide array by classes
    pos = []
    neg = []
    
    # iterating through indices
    for i in xrange(len(x_i)):
        if y[i] == 1:
            pos.append(x_i[i])
        else:
            neg.append(x_i[i])
        
    ax.hist([pos, neg], color=['red', 'green'], edgecolor='none', alpha=0.6)
    # set axis to empty
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

def plot_dataset(x, y, features):
    
    size = len(features)
    f, ax = pl.subplots(size, size, figsize=(size*3, size*3))

    for i, feature_i in enumerate(features):
        for j, feature_j in enumerate(features):

            # Do actual plotting
            if i != j:
                plot_two_features_scatter(x[:, i], x[:, j], y, ax[i, j])  
            else:
                plot_feature_histogram(x[:, i], y, ax[i, j])
               
            # set axis labels
            if i == 0:
                ax[i, j].set_xlabel(feature_j)
                ax[i, j].xaxis.set_label_position('top')
                
            if j == 0:
                ax[i, j].set_ylabel(feature_i)
    
    pl.show()


# Построим попарные распределения географических признаков ([подсказка](http://anokhin.github.io/img/hw2_geo.png)).

# In[7]:

geo_features_new = ["lat", "lon", "country_code"]

geo_features = [f for f in features if f in geo_features_new]
geo_feature_ind = [i for i, f in enumerate(features) if f in geo_features]

plot_dataset(x[:, geo_feature_ind], y, geo_features)


# Четко видны очертания карты и то, что большинство пользователей происходят из небольшого набора стран. Если принять во внимание конечную цель -- кластеризацию пользователей -- логично предположить, что использование географических признаков для описания пользователя может оказаться не очень полезным. Причина в том, что эти признаки четко пространственно разделены (как минимум, океанами и морями). Поэтому мы рискуем вместо "интересной" кластеризации получить просто кластеры, которые будут представлять разные страны. В дальнейшем мы исключим географические признаки из рассмотрения при кластеризации пользователей.
# 
# Далее построим попарные распределения социальных признаков ([подсказка](http://anokhin.github.io/img/hw2_social1.png)).

# In[8]:

social_features_new = ["verified", "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count", "created_year"]
      
social_features = [f for f in features if f in social_features_new]
social_feature_ind = [i for i, f in enumerate(features) if f in social_features]

plot_dataset(x[:, social_feature_ind], y, social_features)


# Из графиков видно, что признаки "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count" сильно смещены в сторону небольших значений. В таком случае удобно сделать логарифмическое преобразрвание этих признаков, то есть применить к их значениям $x_{ij}$ функцию $\log(1 + x_{ij})$. Сделаем это и построим новые распределения ([подсказка](http://anokhin.github.io/img/hw2_social2.png)). Необходимо реализовать функцию log_transform_features, которая выполняет указанное логарифмическое преобразование.

# In[9]:

def log_transform_features(data, features, transformed_features):

    ind = [i for i, f in enumerate(features) if f in transformed_features]
    
    # transform selected features with log function
    for i in xrange(len(data)):
        for j in ind:
            data[i, j] = log(1 + data[i, j])
    
    return data


# In[10]:

transformed_features = ["followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count"]
x = log_transform_features(x, features, transformed_features)

# Re-plot features
plot_dataset(x[:, social_feature_ind], y, social_features)


# Сразу бросается в глаза, что признак "verified" сильно смещен -- верифицированных пользователей очень мало. Более того, все верифицированные пользователи имеют много фолловеров, поэтому часть информации о верификации дублируется в признаке "followers_count". По этой причине в дальнейшем не будем рассмтаривать признак "verified".
# 
# После того как мы с помощью логарифмического преобразования избавились от сильной скошенности признаков, можно наблюдать некоторые интересные зависимости. Например, пользователи, имеющие много фолловеров, обязательно имеют много статусов. Следовательно, чтобы стать популярным, обязательно нужно много писать. Анализ других зависимостей остается как упражнение.
# 
# Наконец построим попарные распределения остальных признаков ([подсказка](http://anokhin.github.io/img/hw2_other.png)).

# In[11]:

other_features_new = ["name_words", "screen_name_length", "description_length"]
other_features = [f for f in features if f in other_features_new]
other_feature_ind = [i for i, f in enumerate(features) if f in other_features]
plot_dataset(x[:, other_feature_ind], y, other_features)


# Итак после первичной обработки данных мы имеем 9 числовых признаков, каждый из которых распределен в некотором своем интервале. Для того, чтобы ни один признак не получил перевеса при кластеризации, нормализуем данные так, что каждый признак распределен на отрезке $[0, 1]$. 

# In[12]:

selected_features = ["followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count", "created_year", "name_words", "screen_name_length", "description_length"]
selected_features_ind = [i for i, f in enumerate(features) if f in selected_features]

x_1 = x[:, selected_features_ind]

# Replace nan with 0-s
# Is there a smarter way?
x_1[np.isnan(x_1)] = 0
x_min = x_1.min(axis=0)
x_max = x_1.max(axis=0)
x_new = (x_1 - x_min) / (x_max - x_min)


# Упакуем полученную матрицу в pandas DataFrame и сохраним в файл "hw2_out.csv". В следующем задании мы будем кластеризовать пользователей на оновании этих данных.

# In[13]:

df_out = pd.DataFrame(data=x_new, index=df_users["user_id"], columns=[f for f in selected_features])
df_out.to_csv("files/hw2_out.csv", sep=";")


# In[ ]:



