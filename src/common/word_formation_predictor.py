from spacy import en

STOP_WORDS = en.STOP_WORDS


class WordFormationPredictor:
    '''
    This class trains and predicts the capitalization of words in sentences.
    It first "trains" on given sentences by counting frequencies of formations of bigrams and unigrams.
    For "prediction" the given sentence is "fixed" according to the statistically most frequent formations.
    '''

    def __init__(self, trainingSentences):
        self.unigramReplacements, self.bigramReplacements = self._learnFromSentences(trainingSentences)


    def predictSentenceWordFormations(self, sentence):
        '''
        Set the capitalization of the given sentence according to the trained data.
        First tries to use known bigram formations and then unigrams.
        :param sentence: A sentence as a string.
        :return: A new string for the "fixed" sentence.
        '''
        sentenceWords = sentence.split()
        sentenceLen = len(sentenceWords)

        sentenceWordsNew = ['' for word in sentenceWords]
        newSentenceCurInd = 0

        for wordInd in range(sentenceLen):

            # get the current and next words (and strip off punctuation):
            curWord = sentenceWords[wordInd]
            curPrefix, curStrippedWord, curSuffix = self._getStrippedWord(curWord)
            curStrippedWordLower = curStrippedWord.lower()
            nextWord = sentenceWords[wordInd + 1] if not self._isSentenceEnd(sentenceWords, wordInd) else ''
            nextPrefix, nextStrippedWord, nextSuffix = self._getStrippedWord(nextWord)
            nextStrippedWordLower = nextStrippedWord.lower()

            # first check if there's a known bigram to use for the word pair:
            if curStrippedWordLower in self.bigramReplacements and nextStrippedWordLower in self.bigramReplacements[curStrippedWordLower]:
                newWordOption, nextNewWordOption = self.bigramReplacements[curStrippedWordLower][nextStrippedWordLower]
                sentenceWordsNew[wordInd] = ''.join([curPrefix, newWordOption, curSuffix])
                sentenceWordsNew[wordInd+1] = ''.join([nextPrefix, nextNewWordOption, nextSuffix])
                newSentenceCurInd = wordInd + 2 # advance the pointer ahead another one since we set a bigram
            # if the current word was not already set by the previous word (with a bigram):
            elif newSentenceCurInd == wordInd:
                if curStrippedWordLower != '':
                    # replace the actual word:
                    useWord = self.unigramReplacements[curStrippedWordLower]
                    # if the word is the first in a sentence, the fist letter should be capitalized:
                    if self._isSentenceStart(sentenceWords, wordInd):
                        useWord = useWord[0].upper() if len(useWord) == 1 else useWord[0].upper() + useWord[1:]
                else:
                    useWord = ''

                # put back the prefix and suffix:
                useWord = ''.join([curPrefix, useWord, curSuffix])
                sentenceWordsNew[wordInd] = useWord
                newSentenceCurInd = wordInd + 1

        newSentence = ' '.join(sentenceWordsNew)
        return newSentence




    def _learnFromSentences(self, trainingSentences, ignoreSameCaseSentence=True):
        '''
        Returns frequencies of the different word formations in all tweets.
        :param allTweets: Dictionary of key:TweetId value:TweetText
        :param ignoreSameCaseSentence: Should sentences that are all in upper or lowercase be ignored for counts? The word is kept, but it is not counted.
        :return: Dictionary of key:wordInLowercase value:replacementWord
        '''
        unigramCounts = {} # counts of each word formation (excluding words after a period)
        bigramCounts = {} # counts of bigram formation (excluding words after a period)

        for sentence in trainingSentences.values():
            # check whether to not count the words of this sentence (still left in dictionary but without a tally):
            if ignoreSameCaseSentence and self._isIgnorableSentence(sentence):
                dontCount = True
            else:
                dontCount = False

            sentenceWords = sentence.split()
            for wordInd, word in enumerate(sentenceWords):

                # get the current word:
                word = self._getStrippedWord(word)[1]
                wordLower = word.lower()

                # get the next word:
                if wordInd+1 < len(sentenceWords):
                    nextWord = self._getStrippedWord(sentenceWords[wordInd+1])[1]
                    nextWordLower = nextWord.lower()
                else:
                    nextWord = nextWordLower = ''

                # tally the current word formation as a unigram:
                if not wordLower in unigramCounts:
                    unigramCounts[wordLower] = {}
                unigramCounts[wordLower][word] = unigramCounts[wordLower].get(word, 0) + 1

                # untally the unigram if we need to ignore it, or if this is the first word of a sentence:
                if dontCount or self._isSentenceStart(sentenceWords, wordInd):
                    unigramCounts[wordLower][word] -= 1

                # tally the current words formation as a bigram:
                if not dontCount and wordLower != '' and nextWordLower != '' and not self._isSentenceStart(sentenceWords, wordInd) and not self._isSentenceStart(sentenceWords, wordInd+1) and wordLower not in STOP_WORDS and nextWordLower not in STOP_WORDS:
                    if not wordLower in bigramCounts:
                        bigramCounts[wordLower] = {}
                    if not nextWordLower in bigramCounts[wordLower]:
                        bigramCounts[wordLower][nextWordLower] = {}
                    bigram = (word, nextWord)
                    bigramCounts[wordLower][nextWordLower][bigram] = bigramCounts[wordLower][nextWordLower].get(bigram, 0) + 1

        return self._prepareWordReplacements(unigramCounts, bigramCounts)

    def _prepareWordReplacements(self, unigramCounts, bigramCounts):
        unigramWordReplacements = self._prepareWordReplacementsUnigram(unigramCounts)
        bigramWordReplacements = self._prepareWordReplacementsBigram(bigramCounts)
        return unigramWordReplacements, bigramWordReplacements

    def _prepareWordReplacementsUnigram(self, unigramCounts):

        # now build a dictionary of replacements (lowercase word -> replacement word):
        unigramWordReplacement = {}
        for word in unigramCounts:
            # find the word forms with the highest frequency (there could be several with the same frequency):
            maxFreq = -1
            maxVals = []
            for wordForm, formFreq in unigramCounts[word].iteritems():
                if formFreq > maxFreq:
                    maxFreq = formFreq
                    maxVals = [wordForm]
                elif formFreq == maxFreq:
                    maxVals.append(wordForm)

            # if the most frequent form has a zero count (the word only appeared in allLower, allUpper or sentence starts),
            # then just use the lowercase form:
            if maxFreq == 0:
                unigramWordReplacement[word] = word.lower()
            # if there's one max, use it:
            elif len(maxVals) == 1:
                unigramWordReplacement[word] = maxVals[0]
            # if there are several max forms, either use the one that is all lowercase, or just the first one:
            else:
                foundWord = ''
                # first try to find an all-lowercase version:
                for wordForm in maxVals:
                    if wordForm.islower():
                        foundWord = wordForm
                        break
                # otherwise use the first one:
                if foundWord == '':
                    foundWord = maxVals[0]
                unigramWordReplacement[word] = foundWord

        #for word in unigramCounts:
        #    if len(unigramCounts[word]) > 1:
        #        print(word + ': ' + str(unigramCounts[word]) + ' || ' + unigramWordReplacement[word])

        return unigramWordReplacement

    def _prepareWordReplacementsBigram(self, bigramCounts):
        '''
        Prepares the dictionary of bigram replacement.
        :param bigramCounts: Dictionary of word1 -> word2 -> bigram tuple form -> count
        :return: Dictionary of word1 -> word2 -> bigram tuple form
        '''

        # build a dictionary of replacements (lowercase bigram as word1 -> word2 -> replacement bigram):
        bigramWordReplacement = {}
        for word1 in bigramCounts:
            for word2 in bigramCounts[word1]:

                # find the word forms with the highest frequency (there could be several with the same frequency):
                maxFreq = -1
                maxVals = []
                for bigramForm, formFreq in bigramCounts[word1][word2].iteritems():
                    if formFreq > maxFreq:
                        maxFreq = formFreq
                        maxVals = [bigramForm]
                    elif formFreq == maxFreq:
                        maxVals.append(bigramForm)

                if not word1 in bigramWordReplacement:
                    bigramWordReplacement[word1] = {}

                if len(maxVals) == 1:
                    bigramWordReplacement[word1][word2] = maxVals[0]
                # if there are several max forms, either use the one that is all lowercase, or just the first one:
                else:
                    foundBigramForm = None
                    # first try to find an all-lowercase version:
                    for bigramForm in maxVals:
                        if bigramForm[0].islower() and bigramForm[1].islower():
                            foundBigramForm = bigramForm
                            break
                    # otherwise use the first one:
                    if foundBigramForm == None:
                        foundBigramForm = maxVals[0]
                    bigramWordReplacement[word1][word2] = foundBigramForm

        #for word1 in bigramCounts:
        #    for word2 in bigramCounts[word1]:
        #        print(word1 + ', ' + word2 + ' : ' + str(bigramWordReplacement[word1][word2]))

        return bigramWordReplacement



    def fix_words_capitalizations(self, text, wordReplacements):
        '''
        Fixes word capitalization according to the given word replacements dictionary.
        :param text: The text to fix.
        :param wordReplacements: Dictionary of key:lowercaseWord and value:replacementWord
        :return: The fixed text.
        '''
        tweetWords = text.split()
        newWords = []
        for wordInd, word in enumerate(tweetWords):
            # split the word:
            prefix, strippedWord, suffix = self._getStrippedWord(word)
            if strippedWord != '':
                # replace the actual word:
                useWord = wordReplacements[strippedWord.lower()]
                # if the word is the first in a sentence, the fist letter should be capitalized:
                if self._isSentenceStart(tweetWords, wordInd):
                    useWord = useWord[0].upper() if len(useWord) == 1 else useWord[0].upper() + useWord[1:]
            else:
                useWord = ''

            # put back the prefix and suffix:
            useWord = ''.join([prefix, useWord, suffix])
            newWords.append(useWord)
        newText = ' '.join(newWords)

        return newText


    def _isIgnorableSentence(self, sentence):
        # check for allUpper, allLower and python isTitle (includes stop words):
        if sentence.isupper() or sentence.islower() or sentence.istitle():
            return True

        # check if the sentence is a title also without stop words:
        sentenceWords = sentence.split()
        sentenceNoStopWords = ' '.join([word for word in sentenceWords if word.lower() not in STOP_WORDS])
        if sentenceNoStopWords.istitle():
            return True

        # check for sequences of only uppercase words, and if there's one longer than 3 words, the sentence is ignorable:
        maxUpperSequenceCount = 0
        upperSequenceCount = 0
        for word in sentenceWords:
            if word.isupper():
                upperSequenceCount += 1
            elif word.islower():
                if upperSequenceCount > maxUpperSequenceCount:
                    maxUpperSequenceCount = upperSequenceCount
                upperSequenceCount = 0
        if upperSequenceCount > maxUpperSequenceCount:
            maxUpperSequenceCount = upperSequenceCount
        if maxUpperSequenceCount > 3:
            return True

        return False

    def _getStrippedWord(self, word):
        '''
        Finds the prefix and suffix of the given word, defined as non-alpha and non-digit character sequences.
        If the word is all non-alpha and non-digit characters, then the suffix will contain the whole given word.
        :param word:
        :return: a triple - prefix, actualWord, suffix
        '''
        wordLen = len(word)
        if wordLen == 0:
            return '', '', ''

        indPre = 0
        while indPre < wordLen and not self._isAlphaDigit(word[indPre]):
            indPre += 1

        indSuf = wordLen - 1
        while indSuf >= 0 and not self._isAlphaDigit(word[indSuf]):
            indSuf -= 1

        prefix = word[0: indPre % wordLen]
        suffix = word[(indSuf+1) : wordLen]
        actualWord = word[(indPre % wordLen) : (indSuf+1)]

        return prefix, actualWord, suffix



    def _isAlphaDigit(self, char):
        return char.isalpha() or char.isdigit()

    def _isSentenceStart(self, textWordsList, wordInd):
        '''
        Is the word at the index specified the beginning of a sentence?
        Note that in "Dr. Who", "Who" would be considered the beginning of a sentence. This is alright in our
        case since it is used for capitalization, and in most case this word needs to be capitalized as well.
        :param textWordsList: The list of words in the text.
        :param wordInd: The word index in the list.
        :return: True iff the specified word is the beginning of a sentence.
        '''
        if wordInd == 0:
            return True
        elif (textWordsList[wordInd - 1][-1] == '.') or (textWordsList[wordInd - 1][-1] == '!') or (textWordsList[wordInd - 1][-1] == '?'):
            return True
        return False

    def _isSentenceEnd(self, textWordsList, wordInd):
        if wordInd == len(textWordsList) - 1:
            return True
        elif (textWordsList[wordInd][-1] == '.') or (textWordsList[wordInd][-1] == '!') or (textWordsList[wordInd][-1] == '?'):
            return True
        elif (wordInd + 1 < len(textWordsList)) and (textWordsList[wordInd+1] == '.' or textWordsList[wordInd+1] == '!' or textWordsList[wordInd+1] == '?'):
            return True
        return False