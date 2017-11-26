"""
Functions to clean and normalize the input tweets before parsing them.
Tweets are often not suitable for automatic parsers, for technical reasons as upper cases, semicolons, etc.
"""
import re

def normalize_tweets(tweets, ignore=False):
    """
    Clean and Normalize the tweets, for better improving it's parsing and extraction. 
    :param tweets: a {tweet_id, tweet_text} dict    
    :param ignore: whether to ignore (i.e. erase) especially "noisy" tweets  
    :return: a {tweet_id, normalized_tweet_text} dict
    """
    normed_tweets = {}
    for tweet_id, text in tweets.iteritems():
        #TODO - clean and manipulate text of tweet
        text = remove_characters(text, ["'", "~"])

        # add normalized tweet to output dict
        normed_tweets[tweet_id] = text
    return normed_tweets

def remove_characters(text, chars_to_remove):
    return re.sub('['+str(chars_to_remove)+']', '', text)

