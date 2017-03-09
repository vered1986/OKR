"""
Receives a test set and evaluates performance of entity mention.
We use the spaCy NER model in addition to annotating all of the nouns and adjectives as entities.

Author: Rachel Wities
"""

import sys
sys.path.append('../common')


import numpy as np

from okr import *
from spacy.en import English
from nltk.corpus import wordnet as wn

NOUNS = [u'NNP', u'NN', u'NNS', u'NNPS', u'CD', u'PRP', u'PRP$']
ADJECTIVES = [u'JJ', u'JJR', u'JJS']
VERBS = [u'VB', u'VBN', u'VBD', u'VBG', u'VBP']
GET_ORIGINAL_SCORE=False #set to true to recieve originaly reported score of 0.58

nom_file = 'nominalizations/nominalizations.reuters.txt'
NOM_LIST = [line.split('\t')[0] for line in open(nom_file)]



# Don't use spacy tokenizer, because we originally used NLTK to tokenize the files and they are already tokenized
nlp = English()
def replace_tokenizer(nlp):
    old_tokenizer = nlp.tokenizer
    nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(string.split())
if not GET_ORIGINAL_SCORE:
    replace_tokenizer(nlp)

#the old and ugly way to fix the fact that spacy tokenization is not similar to NLTK's:
def fix_tokenization(string_):	
	return string_.replace("-","~").replace("'","|")


def evaluate_entity_mention_single(gold_entities_set, pred_graph):
    """
    Receives the gold standard entity mentions and the predicted OKR graph and evaluates it for entity mentions.
    :param gold_entities_set: the gold standard entity mentions
    :param pred_graph: the predicted OKR graph
    :return: F1, recall, and precision
    """
    graph1_ent_mentions = set.union(*[set(map(str, entity.mentions.values()))
                                      for entity in pred_graph.entities.values()])
    graph2_ent_mentions = gold_entities_set
    consensual_mentions = graph1_ent_mentions.intersection(graph2_ent_mentions)

    recall = len(consensual_mentions) * 1.0 / len(graph1_ent_mentions)
    precision = len(consensual_mentions) * 1.0 / len(graph2_ent_mentions)

    f1 = 2.00 * (recall * precision) / (recall + precision) if (recall + precision) > 0.0 else 0.0
    return f1, recall, precision


def evaluate_entity_mention(test_graphs):
    """
    Receives the predicted test graphs and the gold standard entity mentions for them and evaluates them
    for entity mentions.
    :param test_graphs: the predicted OKR graphs
    :return: F1, recall, and precision
    """


    scores = []

    for graph in test_graphs:

	if GET_ORIGINAL_SCORE:
        	sents = {num: unicode(uncap_sentence(fix_tokenization(' '.join(sentence)))) for num, sentence in graph.sentences.iteritems()}
	else:
		sents = {num: unicode(uncap_sentence(' '.join(sentence))) for num, sentence in graph.sentences.iteritems()}

	sents = {key: sentence for key, sentence in sents.iteritems() if len(nlp(sentence)) == len(graph.sentences[key])}
        # NER entity
        ner_singles = [[s_num, num, tok.ent_iob, tok.tag_] for s_num, sentence in sents.iteritems() for num, tok in
                       enumerate(nlp(sentence)) if tok.ent_iob in [1, 3]]
        ner_wpos = convert_iob_to_seq(ner_singles)

        # Remove determiners and possesives
        ner_wpos = [[mention[0], [index for num, index in enumerate(mention[1]) if not mention[2][num] == u'DT'],
                     [pos for pos in mention[2]]] for mention in ner_wpos]
        ner_wpos = [[mention[0], [index for num, index in enumerate(mention[1]) if not mention[2][num] == u'POS'],
                     [pos for pos in mention[2]]] for mention in ner_wpos]

        # Remove entities that contain verbs
        ner_wpos = [mention for mention in ner_wpos if len(set(mention[2]).intersection(set(VERBS))) == 0]

        # Convert to string
        ner = [str(item[0]) + str(item[1]) for item in ner_wpos]

        # Every noun or adjective is an entity, except nominalizations
        nouns_wword = [[str(s_num) + "[" + str(num) + "]", tok, tok.tag_] for s_num, sentence in sents.iteritems() for
                       num, tok in enumerate(nlp(sentence)) if tok.tag_ in NOUNS or tok.tag_ in ADJECTIVES]

        nouns = set([noun[0] for noun in nouns_wword])

        # Exclude indices collected by NER
        nouns = nouns - set([str(mention[0]) + "[" + str(index) + "]" for mention in ner_wpos for index in mention[1]])
        scores.append(evaluate_entity_mention_single(nouns.union(ner), graph))
    # Return the average
    score = np.mean(scores, axis=0)[0]
    return score


def is_nominalization(word):
    """
    Returns whether this word is a nominalization, by checking if it has a WordNET derivationally_related_form
    relation to a verb
    :param word: the word
    :return: whether this word is a nominalization
    """
    if len(wn.synsets(word)) == 0:
        return False

    if len(wn.synsets(word)[0].lemmas()[0].derivationally_related_forms()) == 0:
        return False

    derive = wn.synsets(word)[0].lemmas()[0].derivationally_related_forms()[0]
    return str(derive).find('.v.') > -1


def pred_to_nom(pred_string):
    """
    Receives a noun phrase and returns the lemmatized nouns
    :param pred_string: the noun phrase
    :return: the lemmatized nouns
    """
    gold_nom_list = [word.lemma_ for word in nlp(unicode(pred_string))
                     if not word.is_stop and not word.tag_ == u'IN' and word.tag_ in NOUNS]
    return gold_nom_list


def uncap_sentence(sent):
    """
    Workaround for tweets: sometimes the entire sentence is in upper case. If more than 99%
    of the sentence is in upper case, this function will convert it to lower case.
    :param sent: the tweet
    :return: the sentence, lower cased if it is all in upper case
    """
    ssent = sent.split()
    ssent_cap = [word for word in ssent if word[0].isupper()]
    if len(ssent_cap) * 1.00 / len(ssent) > 0.9:
        return sent.lower()
    else:
        return sent


def convert_iob_to_seq(ner_iob_annotations):
    """
    NER is annotated using IOB tags. In this scheme, each token is tagged with one of three special chunk tags,
    I (inside), O (outside), or B (begin). A token is tagged as B if it marks the beginning of a chunk.
    Subsequent tokens within the chunk are tagged I. All other tokens are tagged O.
    This function returns the sequences of named entities.
    :param ner_iob_annotations: the NER annotations in IOB format
    :return: the sequences
    """

    # 1 = i, 2 = o, 3 = b
    sequence = []

    for item in ner_iob_annotations:
        sent_num, token_id, token_iob, tag = item

        # begin NER (b)
        if token_iob == 3:
            sequence.append([sent_num, [token_id], [tag]])
        # continue NER (i)
        else:
            sequence[-1][1].append(token_id)
            sequence[-1][2].append(tag)

    return sequence

