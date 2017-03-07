"""
Receives a test set and evaluates performance of entity coreference.
We cluster the entity mentions based on simple lexical similarity metrics (e.g., lemma matching and
Levenshtein distance)

Author: Rachel Wities
"""

import sys

sys.path.append('../agreement')

import spacy
import numpy as np

from munkres import *
from entity_coref import *
from fuzzywuzzy import fuzz
from spacy.en import English
from num2words import num2words
from nltk.corpus import wordnet as wn
from clustering_common import cluster_mentions

# Don't use spacy tokenizer, because we originally used NLTK to tokenize the files and they are already tokenized
nlp = spacy.load('en')

def is_stop(w):
	return w in spacy.en.STOP_WORDS


def replace_tokenizer(nlp):
    old_tokenizer = nlp.tokenizer
    nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(string.split())

replace_tokenizer(nlp)


def evaluate_entity_coref(test_graphs):
    """
    Receives the OKR test graphs and evaluates them for entity coreference
    :param test_graphs: the OKR test graphs
    :return: the coreference scores: MUC, B-CUBED, CEAF and MELA (CoNLL F1).
    """
    scores = []

    for graph in test_graphs:

        # Cluster the entities
        entities = [(str(mention), unicode(mention.terms)) for entity in graph.entities.values() for mention in
                    entity.mentions.values()]
        clusters = cluster_mentions(entities, score)
        clusters = [set([item[0] for item in cluster]) for cluster in clusters]

        # Evaluate
        curr_scores = eval_clusters(clusters, graph)
        scores.append(curr_scores)

    scores = np.mean(scores, axis=0).tolist()

    return scores


def eval_clusters(clusters, graph):
    """
    Receives the predicted clusters and the gold standard graph and evaluates (with coref metrics) the entity
    coreferences
    :param clusters: the predicted clusters
    :param graph: the gold standard graph
    :return: the predicate coreference metrics and the number of singletons
    """
    graph1_ent_mentions = [set(map(str, entity.mentions.values())) for entity in graph.entities.values()]
    graph2_ent_mentions = clusters

    # Evaluate
    muc1, bcubed1, ceaf1 = muc(graph1_ent_mentions, graph2_ent_mentions), \
                           bcubed(graph1_ent_mentions, graph2_ent_mentions), \
                           ceaf(graph1_ent_mentions, graph2_ent_mentions)

    mela1 = np.mean([muc1, bcubed1, ceaf1])
    return np.array([muc1, bcubed1, ceaf1, mela1])


def score(mention, cluster):
    """
    Receives an entity mention (mention, head_lemma, head_pos)
    and a cluster of entity mentions, and returns a numeric value denoting the
    similarity between the mention to the cluster (% of similar head lemma mentions in the cluster)
    :param mention: the mention
    :param cluster: the cluster
    :return: a numeric value denoting the similarity between the mention to the cluster
    """
    return len([other for other in cluster if similar_words(other[1], mention[1])]) / (1.0 * len(cluster))


def similar_words(x, y):
    """
    Returns whether x and y are similar
    :param x: the first mention
    :param y: the second mention
    :return: whether x and y are similar
    """
    return same_synset(x, y) or fuzzy_fit(x, y) or partial_match(x, y)


def same_synset(x, y):
    """
    Returns whether x and y share a WordNet synset
    :param x: the first mention
    :param y: the second mention
    :return: whether x and y share a WordNet synset
    """
    x_synonyms = set([lemma.lower().replace('_', ' ') for synset in wn.synsets(x) for lemma in synset.lemma_names()])
    y_synonyms = set([lemma.lower().replace('_', ' ') for synset in wn.synsets(y) for lemma in synset.lemma_names()])

    return len([w for w in x_synonyms.intersection(y_synonyms) if not is_stop(w)]) > 0


def fuzzy_fit(x, y):
    """
    Returns whether x and y are similar in fuzzy string matching
    :param x: the first mention
    :param y: the second mention
    :return: whether x and y are similar in fuzzy string matching
    """
    if fuzz.ratio(x, y) >= 90:
        return True

    # Convert numbers to words
    x_words = [num2words(int(w)).replace('-', ' ') if w.isdigit() else w for w in x.split()]
    y_words = [num2words(int(w)).replace('-', ' ') if w.isdigit() else w for w in y.split()]

    return fuzz.ratio(' '.join(x_words), ' '.join(y_words)) >= 85


def partial_match(x, y):
    """
    Return whether these two mentions have a partial match in WordNet synset.
    :param x: the first mention
    :param y: the second mention
    :return: Whether they are aligned
    """

    # Allow partial matching
    if fuzz.partial_ratio(' ' + x + ' ', ' ' + y + ' ') == 100:
        return True

    x_words = [w for w in x.split() if not is_stop(w)]
    y_words = [w for w in y.split() if not is_stop(w)]

    if len(x_words) == 0 or len(y_words) == 0:
        return False

    x_synonyms = [set([lemma.lower().replace('_', ' ') for synset in wn.synsets(w) for lemma in synset.lemma_names()])
                  for w in x_words]
    y_synonyms = [set([lemma.lower().replace('_', ' ') for synset in wn.synsets(w) for lemma in synset.lemma_names()])
                  for w in y_words]

    # One word - check whether there is intersection between synsets
    if len(x_synonyms) == 1 and len(y_synonyms) == 1 and \
                    len([w for w in x_synonyms[0].intersection(y_synonyms[0]) if not is_stop(w)]) > 0:
        return True

    # More than one word - align words from x with words from y
    cost = -np.vstack([np.array([len([w for w in s1.intersection(s2) if not is_stop(w)]) for s1 in x_synonyms])
                       for s2 in y_synonyms])
    m = Munkres()
    cost = pad_to_square(cost)
    indices = m.compute(cost)

    # Compute the average score of the alignment
    average_score = np.mean([-cost[row, col] for row, col in indices])

    if average_score >= 0.75:
        return True

    return False
