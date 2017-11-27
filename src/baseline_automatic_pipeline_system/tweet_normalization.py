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

        # remove noisy characters and patterns
        text = remove_characters(text, ["'", "~"])
        text = remove_urls(text)

        # validate end of sentence - remove noisy traces and make sure the sentence is added with a period (or '?'\'!')
        text = remove_signatures(text)
        text = validate_ending(text)

        # add normalized tweet to output dict
        normed_tweets[tweet_id] = text
    return normed_tweets

def remove_characters(text, chars_to_remove):
    return re.sub('['+str(chars_to_remove)+']', '', text)


def remove_signatures(text):
    #define signs for a signature (stating the author of the tweet, not part of the grammatical sentence)
    signature_signs = ["|", " - "]
    # remove signatures
    for sign in signature_signs:
        text = sign.join(text.split(sign)[:-1]) # remove last chunk starting from sign
    return text

def validate_ending(text):
    """
    Remove noisy last-characters and make sure sentence will end with period (or "?" or "!") 
    """
    # remove noisy last-characters
    while True:
        text = ' '.strip()
        noisy_traces = [":", ";", "-", ",", "#"] # chars that we want to remove if occurring at the end of the sentence
        if text[-1] in noisy_traces:
            text = text[:-1]    # remove last char
        else:
            break
    # set last character as period
    if text[-1] not in [".", "?", "!"]:
        text += "."

    return text

def remove_urls(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_in_text = re.findall(url_regex, text)
    for url in urls_in_text:
        text = text.replace(url, "")
    return text