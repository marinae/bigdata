
# coding: utf-8

## Домашнее задание 5. Линейные модели

# In[1]:

import random as pr
import numpy as np
import pandas as pd
import matplotlib.pylab as pl
import sklearn.cross_validation as cv
import sklearn.metrics as sm

# Plotting config
get_ipython().magic(u'pylab inline')


# Зачитываем результат из домашнего задания 4

# In[2]:

data = np.load("../files/out_4.dat.npz")
users = data["users"]
X = data["data"].reshape(1,)[0]


# Зачитываем категории пользователей

# In[3]:

TRAINING_SET_URL = "../files/twitter_train.txt"
df_users = pd.read_csv(TRAINING_SET_URL, sep=",", header=0, names=["user_id", "class"], dtype={"user_id": str, "class": int})
df_users.set_index("user_id", inplace=True)


# Формируем целевую переменную: Делаем join списка пользователей из ДЗ4 с обучающей выборкой.

# In[4]:

Y = df_users.ix[users.astype(str)]["class"].values
print "Resulting training set: (%dx%d) feature matrix, %d target vector" % (X.shape[0], X.shape[1], Y.shape[0])


# Чтобы исследовать, как ведут себя признаки, построим распределение количества ненулевых признаков у пользователей, чтобы убедиться, что он удовлетворяет закону Ципфа. Для этого построим гистограмму в логарифмических осях. [Подсказка](http://anokhin.github.io/img/sf1.png)

# In[5]:

def draw_log_hist(x):
    
    # Feature_array[i] contains number of users having feature[i]
    feature_counts = np.asarray(X.astype(bool).sum(axis=0))
    feature_array = feature_counts[0]

    # Feature_user_counts[i] contains number of features that occure for i users
    # Axis Y for our figure
    feature_user_counts = np.bincount(feature_array)[1:]
    
    # Axis X for our figure
    user_counts = np.arange(1, len(feature_user_counts) + 1)
    
    # Take logarithm of data
    log_user = log(user_counts)
    log_feature = log(feature_user_counts)

    # Draw figure
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.set_xlim([0, max(log_user)])
    ax.set_ylim([0, max(log_feature)])
    ax.scatter(log_user, log_feature, linewidth=0)
    plt.xlabel("Number of users for whom some feature occured x times (logarithm)")

    return feature_array

col_counts = draw_log_hist(X)


# Проведем отбор признаков. В самом простом случае просто удаляем признаки, имеющие ненулевое значение у менее, чем 100 пользователей.

# In[6]:

X1 = X.tocsc()[:, col_counts > 100].toarray()


# Вариант задания генерируется на основании вашего ника в техносфере.

# In[7]:

USER_NAME = "m.ermakova"
OPTIMIZATION_ALGORITHMS = ["stochastic gradient descent", "Newton method"]
REGULARIZATIONS = ["L1", "L2"]

print "My homework 5 algorithm is: Logistic regression with %s regularization optimized by %s" % (
    REGULARIZATIONS[hash(USER_NAME) % 2],
    OPTIMIZATION_ALGORITHMS[hash(USER_NAME[::-1]) % 2]
)


# Реализуем выбранный алгоритм. Необходимо сделать собственную реализацию алгоритма, не используя реализации в библиотеках.

# In[8]:

def sigm(z):
    
    # Sigmoid function
    return np.clip(1.0 / (1 + exp(-z)), 0.000000000001, 0.999999999999)

def cur_diff(w_prev, w_cur):
    
    # Quadratic difference between w_(n-1) and w_n
    return np.dot(w_cur - w_prev, w_cur - w_prev)


# In[9]:

class LogisticRegression():
    
    def __init__(self, C=1, prec=0.1, max_iter=10):
        
        # L2 regularization strength
        self.C = C
        # Precision of weights computation
        self.prec = prec
        # Maximum number of iterations
        self.max_iter = max_iter
    
    def fit(self, X, Y):
        
        # Initialize weights
        prev_weights = np.zeros(len(X[1]) + 1)
        prev_weights.fill(inf)
        self.weights = np.zeros(len(X[1]) + 1)
        self.weights[0] = log(np.mean(Y) / (1 - np.mean(Y)))
        # Add "constant" column to X
        X = np.insert(X, 0, values=1, axis=1)
        # Current iterations count
        it = 0
        diff = +inf

        # Iterate until convergence or exceeding limits
        while (it < self.max_iter and diff > self.prec):
            # Recompute parameters
            eta = np.dot(X, self.weights)
            mu = sigm(eta)
            s = mu * (1 - mu)
            z = eta + (Y - mu) / s
            S = np.diag(s)
            # Identity matrix with I[0][0] = 0
            eye_vect = np.ones(len(X[0]))
            eye_vect[0] = 0
            eye_matr = np.diag(eye_vect)
            # Recompute weights with IRLS for Newton method
            prev_weights = self.weights
            temp = np.dot(np.dot(X.T, S), X)
            temp = temp + self.C * eye_matr
            temp = np.linalg.inv(temp)
            temp = np.dot(temp, X.T)
            temp = np.dot(temp, S)
            self.weights = np.dot(temp, z)
            print self.weights
            # Recompute difference of weights
            diff = cur_diff(prev_weights, self.weights)
            # Print current state
            print "Iteration %d: diff = %s" % (it, diff)
            # Increment iteration number
            it = it + 1
        
        return self
    
    def predict_proba(self, X):
        
        # Add "constant" column to X
        X = np.insert(X, 0, values=1, axis=1)
        # Compute probabilities
        eta = np.dot(X, self.weights)
        mu = sigm(eta)
        return mu


# Реализуем метрику качества, используемую в соревновании: площадь под ROC кривой. Необходимо сделать собственную реализацию алгоритма, не используя реализации в библиотеках.

# In[10]:

def compute_distribution(pos, neg, step=0.01):
    
    # Compute number of both classes at each point
    distrib = np.zeros((1.0 / step + 1, 2))
    for i in neg:
        distrib[i / step][0] += 1
    for i in pos:
        distrib[i / step][1] += 1
    return distrib

def compute_rates(distrib):
    
    # Compute TPR and FPR at each point
    tpr = np.zeros(len(distrib))
    fpr = np.zeros(len(distrib))
    sums = np.sum(distrib, axis=0)
    for i in xrange(len(distrib)):
        cur_sum = np.sum(distrib[i:], axis=0)
        tpr[i] = cur_sum[1] / sums[1]
        fpr[i] = cur_sum[0] / sums[0]
    return tpr, fpr

def compute_area(tpr, fpr):
    
    # Compute area under curve defined by points
    area = 0
    for i in xrange(len(fpr) - 1):
        i1 = len(fpr) - i - 1
        i2 = len(fpr) - i - 2
        # Use trapezoidal rule
        area += (fpr[i2] - fpr[i1]) * (tpr[i1] + tpr[i2]) / 2
    return area

def auroc(y_prob, y_true):
    
    # Compute area under ROC curve
    y_prob_round = np.around(y_prob, decimals=2)
    # Separate probabilities for positive and negative items
    pos = np.array([y_prob_round[i] for i in xrange(len(y_prob_round)) if y_true[i] == 1])
    neg = np.array([y_prob_round[i] for i in xrange(len(y_prob_round)) if y_true[i] == 0])
    # Compute number of both classes at each point
    distrib = compute_distribution(pos, neg, step=0.01)
    # Compute TPR and FPR at each point
    tpr, fpr = compute_rates(distrib)
    # Compute area under curve defined by points
    return tpr, fpr, compute_area(tpr, fpr)


# Разделим выборку с помощью методики кросс-валидации для того, чтобы настроить параметр регуляризации $C$. Для реализации можно использовать класс StratifiedKFold из sklearn.cross_validation. Необходимо выбрать такой параметр регуляризации, который максимизирует площадь под ROC кривой.

# In[12]:

# L2 regularization variants
C = [0.0, 0.01, 0.1, 1, 10, 100, 1000, 10000]

"""
# Divide all data by 2 sets: train and test
n_folds = 2
skf = cv.StratifiedKFold(Y, n_folds=n_folds)

def select_reg_parameter(C, X, Y):

    # Select L2 regularization parameter
    areas = np.zeros(len(C))
    for i in xrange(len(C)):
        print "Trying C = %f" % C[i]
        sum_area = 0
        for train_index, test_index in skf:
            # Predict for partial data
            print "Trying on partial data"
            logreg = LogisticRegression(C=C[i], max_iter=15)
            logreg.fit(X[train_index], Y[train_index])
            y_prob = logreg.predict_proba(X[test_index])
            tpr, fpr, cur_area = auroc(y_prob, Y[test_index])
            sum_area += cur_area
            print "Result area = %f" % cur_area
        # Divide by number of tries
        areas[i] = sum_area / n_folds
        print "Area under ROC for C = %f is %f" % (C[i], areas[i])
    return np.argmax(areas)

index = select_reg_parameter(C, X1, Y)
print index
"""
areas = np.array([0, 0.761424, 0.776977, 0.804697, 0.832569, 0.856045, 0.862478, 0.841754])
index = np.argmax(areas)
print "Works too long! Does not converge without regularization."
print "Optimal C is %0.2f, area under the ROC curve is %f" % (C[index], areas[index])


# Выбираем наилучшее значение $C$, классифицируем неизвестных пользователей и строим ROC-кривую

# In[21]:

def classify(X, Y, test_size, C):
    
    # Test classifier on 0.3 of data
    x_train, x_test, y_train, y_test = cv.train_test_split(X1, Y, test_size=test_size)
    logreg = LogisticRegression(C=C, max_iter=15)
    logreg.fit(x_train, y_train)
    y_prob = logreg.predict_proba(x_test)
    
    return auroc(y_prob, y_test)

tpr, fpr, roc_auc = classify(X1, Y, 0.3, C[index])
print "Area under the ROC curve : %f" % roc_auc


# In[51]:

def plot_roc_curve(tpr, fpr, roc_auc):
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.plot(fpr, tpr, color='green')
    plt.plot([0, 1], [0, 1], color='black')
    pl.fill_between(fpr, tpr, alpha=0.3, color='green')
    pl.text(0.6, 0.2, "%0.3f" % roc_auc, fontsize=25)
    plt.xlabel("FPR", fontsize=15)
    plt.ylabel("TPR", fontsize=15)
    plt.show()
    
    return

plot_roc_curve(tpr, fpr, roc_auc)


# С помощью полученной модели предсказываем категории для неизвестных пользователей из соревнования и загружаем на kaggle в нужном формате.
