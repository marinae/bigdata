import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer

def get_words(text):
    # Returns list of words from text
    
    return text.split()

def get_tokens(text):
    # Returns list of tokens

    words = get_words(text)

    stopwords = nltk.corpus.stopwords.words('english')

    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(w).lower() for w in words]
    tokens_nonstop = [t for t in tokens if t not in stopwords]

    return tokens_nonstop

def get_tweet_tokens(tweet):
    # Returns list of tweet tokens

    tweet = tweet.encode('ascii', 'ignore')
    tweet = re.sub(r'\bhttps?://\S*\b',       ' ', tweet)
    tweet = re.sub(r'@\b\w+\b',               ' ', tweet)
    tweet = re.sub(r'&amp;',                  ' ', tweet)
    tweet = re.sub(r'[^\w\-\' ]',             ' ', tweet)
    tweet = re.sub(r'\b\d+\b' ,               ' ', tweet)
    tweet = re.sub(r'(\s\'\s|\s\'\b|\b\'\s)', ' ', tweet)
    tweet = re.sub(r'(\s\-\s|\s\-\b|\b\-\s)', ' ', tweet)

    return get_tokens(tweet)

def get_user_dict(user_tweets):
	# Returns dictionary of single user

	tokens = [get_tweet_tokens(t) for t in user_tweets]
	all_tokens = [word for i in tokens for word in i]
	user_dict = dict.fromkeys(all_tokens)

	for key in user_dict:
		user_dict[key] = all_tokens.count(key)

	return user_dict

def collect_users_tokens(df_users, tweets):
    # Returns users list and list of user dicts. Each dict contains frequence of user tokens

    return df_users["user_id"], [get_user_dict(user_tweets) for user_tweets in tweets]

# Main code #############################################

a = pickle.loads(open("../files/raw_tweets_full.txt", "rb").read())

print get_user_dict(a[7999])


#pickle.dump(a, open("clean_tweets.txt", "wb"))

#users, users_tokens = collect_users_tokens(df_users)
#v = DictVectorizer()
#vs = v.fit_transform(users_tokens)
