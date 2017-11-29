"""
Functions to clean and normalize the input tweets before parsing them.
Tweets are often not suitable for automatic parsers, for technical reasons as upper cases, semicolons, etc.
"""
import re
import sys

# characters to remove from the tweets:
chars_to_remove = ['>', '<', "~"]
# characters to strip from the end of the tweet:
chars_to_strip = [' ','\t','\r','\n','-','_','|',':',';','~','>','<','+','*']

def normalize_tweets(tweets, ignore=False):
    """
    Clean and Normalize the tweets, for better improving it's parsing and extraction.
    :param tweets: a {tweet_id, tweet_text} dict
    :param ignore: whether to ignore (i.e. erase) especially "noisy" tweets
    :return: a {tweet_id, normalized_tweet_text} dict
    """

    normed_tweets = {}
    twitterLexicon = get_twitter_lexicon()
    emojiList = get_twitter_emoji()

    for tweet_id, text in tweets.iteritems():

        # ignore tweets ending with an ellipsis since this is likely a cutoff tweet:
        if text.endswith('...'):
            normed_tweets[tweet_id] = ''
            continue

        # remove unneeded prefixes and suffixes:
        text = remove_unneeded_prefixes(text)
        text = remove_unneeded_suffixes(text)

        # remove noisy characters and patterns
        text = clean_hashtags(text)
        text = remove_urls(text)
        text = replace_expanded_characters(text)
        text = remove_characters(text, chars_to_remove)
        text = replace_strings(text, [(emoticon, '. ') for emoticon in emojiList])
        text = replace_strings(text, [('... ', '. '), ('...', '. '), ('.. ', '. '), ('..', '. '), (' -- ', ', ')])
        text = remove_brackets(text)
        text = split_connected_words(text)
        text = replace_words_by_lexicon(text, twitterLexicon)

        # TODO: reformat to correct casing (use most frequent casing throughout all sentences)

        # validate end of sentence - remove noisy traces and make sure the sentence is added with a period (or '?'\'!')
        text = strip_characters(text, chars_to_strip)
        text = validate_ending(text)

        # add normalized tweet to output dict
        normed_tweets[tweet_id] = text

    return normed_tweets

def remove_characters(text, chars_to_remove):
    # for the regex, given a list of characters, put a '\' before each one, and separate them by a '|':
    return re.sub('|'.join(['\\'+c for c in chars_to_remove]), '', text)

def remove_strings(text, strs):
    for s in strs:
        text = text.replace(s, '')
    return text

def replace_strings(text, strs):
    '''
    Replace a string with another one.
    :param text:
    :param strs: List of tuples (<str to replace>, <str to replace with>)
    :return:
    '''
    for s, r in strs:
        text = text.replace(s, r)
    return text

def remove_brackets(text):
    """
    Remove everything within brackets or parentheses. Assuming no brackets within brackets.
    Note:   This removes from an opening bracket to the first closing bracket after it.
            So "abc (def(gh)ij) klm" -> "abc ij) klm". This can be improved.
    """
    return re.sub(r"[\(\[].*?[\)\]]", "", text)


def validate_ending(text):
    """
    Remove noisy last-characters and make sure sentence will end with period (or "?" or "!")
    """
    # remove noisy last-characters
    noisy_traces = [":", ";", "-", ",", "#", " ", "\t"] # chars that we want to remove if occurring at the end of the sentence
    text = text.strip(''.join(noisy_traces))
    # set last character as period
    if len(text) > 0 and text[-1] not in [".", "?", "!"]:
        text += "."

    return text

def remove_urls(text):
    # remove patterns of websites starting with http[s]://
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_in_text = re.findall(url_regex, text)
    for url in urls_in_text:
        text = text.replace(url, "")

    # also remove URLS starting with www and not with http[s]://
    url_regex = 'www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_in_text = re.findall(url_regex, text)
    for url in urls_in_text:
        text = text.replace(url, "")

    return text

def replace_expanded_characters(text):
    '''
    Some characters have been formatted due to encoding. These are undone.
    :param text:
    :return:
    '''
    # replace some character representations:
    text = text.replace('&gt;', '>')
    text = text.replace('&lt;', '<')
    text = text.replace("&amp;", "and")

    return text

def strip_characters(text, chars_to_strip):
    '''
    Removes characters from the end of the text.
    :param text: The string to manipulate
    :param chars_to_strip: A list of characters
    :return: The updated text string
    '''
    return text.strip(''.join(chars_to_strip))

def remove_unneeded_prefixes(text):
    # line has one two or three words and then an attribution separating character, so remove that beginning:
    tweetWords = text.split()
    prefixSeparators = [':', '>', '*', '!', '?']
    for separator in prefixSeparators:
        if len(tweetWords) > 3:
            if (tweetWords[0][-1] == separator) or (tweetWords[1][-1] == separator) or (tweetWords[2][-1] == separator):
                text = text[text.find(separator) + 1:]

    return text

def remove_unneeded_suffixes(text):
    # remove attributions at the end of the line (like "... via ...")
    while True:
        tweetWords = text.split()
        if len(tweetWords) > 3:
            if tweetWords[-2][-1] == ':' or tweetWords[-1] == 'via':
                text = ' '.join(tweetWords[:-1])
            elif tweetWords[-2] == '|' or tweetWords[-2] == 'via' or tweetWords[-2] == '-' or tweetWords[-2] == 'by' or tweetWords[-3][-1] == ':':
                text = ' '.join(tweetWords[:-2])
            elif tweetWords[-3] == '|' or tweetWords[-3] == 'via' or tweetWords[-3] == '-' or tweetWords[-3] == 'by' or tweetWords[-4][-1] == ':':
                text = ' '.join(tweetWords[:-3])
            elif tweetWords[-4] == '|' or tweetWords[-4] == 'via' or tweetWords[-4] == '-' or tweetWords[-4] == 'by':
                text = ' '.join(tweetWords[:-4])
            elif len(tweetWords) > 4 and (tweetWords[-5] == '|' or tweetWords[-5] == 'via' or tweetWords[-5] == '-'):
                text = ' '.join(tweetWords[:-5])
            else:
                break
        else:
            break

    return text

def clean_hashtags(text):
    # remove all hashtag words at the end of the sentence:
    tweetWords = text.split()
    if len(tweetWords) > 3:
        idx = -1
        while abs(idx) <= len(tweetWords) and tweetWords[idx][0] == '#':
            text = text[:tweetWords.rfind('#') - 1]
            idx -= 1

    # for hastags in the middle of the sentence, just remove the hashtag symbol and keep the word:
    text = text.replace("#", "")
    text = text.replace("@", "")

    return text

def get_twitter_lexicon():
    '''
    Creates and returns the Twitter lexicon.
    :return: A dictionary of key:TwitterWord value:RealWord
    '''
    twitter_lexicon = {}
    with open('../../resources/normalization/twitter_lexicon.txt', 'r') as fIn:
        for line in fIn:
            parts = line.split()
            twitter_lexicon[parts[0]] = parts[1]

    return twitter_lexicon

def get_twitter_emoji():
    '''
    Creates and returns the Twitter emoji list.
    :return: A list of emoticon strings.
    '''
    emoji = []
    with open('../../resources/normalization/twitter_emoji.txt', 'r') as fIn:
        for line in fIn:
            emoji.append(line.strip())
    return emoji

def replace_words_by_lexicon(text, lexicon):
    '''
    Replaces the words that are of Twitter lexicon (e.g. 2mrw) to actual words from the given lexicon dictionary.
    :param text:
    :param lexicon: Dictionary with key:TwitterWord value:actualWord
    :return:
    '''
    # ignore a tweets that is all uppercase:
    if text.isupper():
        return text

    # if a word is not all uppercase and it is in the lexicon, replace it:
    tweetWords = text.split()
    newWordsList = [lexicon[word] if not word.isupper() and word.lower() in lexicon else word for word in tweetWords]
    # There is also a possiblity of stripping away punctuation from a word before checking if it's in the lexicon,
    # but then put it back. (Use word.strip('.!?;:,\'"')
    newText = ' '.join(newWordsList)
    #if text != newText:
    #    print(text+'\n'+newText+'\n\n')
    return  newText

def isDottedAcronym(word):
    '''
    Is the given word a dotted acronym such as "u.s.", "i.e.", etc.
    :param word:
    :return:
    '''
    # get all the acronyms that have dots (ignore one letter and dot):
    matches = re.finditer(r'(?:[A-Z|a-z]\.)+', word)
    dottedAcronyms = [m.group(0) for m in matches if len(m.group(0)) > 2]
    # check if the acronym is equal to the original word:
    for acr in dottedAcronyms:
        if acr == word:
            return True
    return False


def split_connected_words(text):
    '''
    Splits words on camelCase or on basic punctuation (sets dotted acronyms to uppercase).
    :param text:
    :return:
    '''
    tweetWords = text.split()
    newWordsList = [split_connected_word(word) for word in tweetWords]
    newText = ' '.join(newWordsList)
    return newText

def split_connected_word(word):
    '''
    Splits a word on camelCase or on basic punctuation.
    :param word:
    :return:
    '''
    # ignore dotted abbreviations (just capitalize them):
    if isDottedAcronym(word):
        return word.upper()

    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[.!?:;,])(?=[A-Z]|[a-z])|$)', word)
    newWordsList = [m.group(0) for m in matches]
    newText = ' '.join(newWordsList)
    #if word != newText:
    #    print(word + '\n' + newText + '\n\n')
    return newText

'''
For testing. Pass in a text file with sentences (line by line).
Pass in -file to write output to a file (inputFile.out) or -console to write to the terminal.
A testing file exists in ../../examples/tweetsToNormalizeTest.txt

Run:
    python tweet_normalization.py ../../examples/tweetsToNormalizeTest.txt -file
To see output in file run:
    vi ../../examples/tweetsToNormalizeTest.txt.out
'''
if __name__ == '__main__':
    if len(sys.argv) > 2:
        # read in the sentences and normalize them:
        with open(sys.argv[1], 'r') as fIn:
            sentences = {i:sent.strip() for i, sent in enumerate(fIn.readlines())}
        normalizedSentences = normalize_tweets(sentences)

        # build up the texts:
        texts = []
        for sentenceId in normalizedSentences:
            text = sentences[sentenceId] + '\n' + normalizedSentences[sentenceId] + '\n-----\n'
            if sentences[sentenceId] != normalizedSentences[sentenceId]:
                text = '!!!\n' + text
            texts.append(text)
        fullText = ''.join(texts)

        # write out:
        if sys.argv[2] == '-file':
            with open(sys.argv[1]+'.out', 'w') as fOut:
                fOut.write(fullText)
        elif sys.argv[2] == '-console':
            print(fullText)

    else:
        print('Usage: python tweet_normalization.py <sentencesFile> <-file|-console>')