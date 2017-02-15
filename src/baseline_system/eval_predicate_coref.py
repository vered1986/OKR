"""
Receives a test set and evaluates performance of predicate coreference.
We cluster the mentions based on simple lexical similarity metrics (e.g., lemma matching and Levenshtein distance)

Author: Shyam Upadhyay
"""

import sys
sys.path.append('../common')

import numpy as np

from okr import *
from entity_coref import *
from clustering_common import cluster_mentions
from parsers.spacy_wrapper import spacy_wrapper


def evaluate_predicate_coref(test_graphs):
    """
    Receives the OKR test graphs and evaluates them for predicate coreference
    :param test_graphs: the OKR test graphs
    :return: the coreference scores: MUC, B-CUBED, CEAF and MELA (CoNLL F1).
    """
    parser = spacy_wrapper()

    scores = []

    for graph in test_graphs:

        # Cluster the mentions
        prop_mentions = []
        for prop in graph.propositions.values():
            for mention in prop.mentions.values():

                if mention.indices == [-1]:
                    continue

                head_lemma, head_pos = get_mention_head(mention, parser, graph)
                prop_mentions.append((mention, head_lemma, head_pos))

        clusters = cluster_mentions(prop_mentions, score)
        clusters = [set([item[0] for item in cluster]) for cluster in clusters]

        # Evaluate
        curr_scores, _ = eval_clusters(clusters, graph)
        scores.append(curr_scores)

    scores = np.mean(scores, axis=0).tolist()

    return scores


def score(prop, cluster):
    """
    Receives a proposition mention (mention, head_lemma, head_pos)
    and a cluster of proposition mentions, and returns a numeric value denoting the
    similarity between the mention to the cluster (% of same head lemma mentions in the cluster)
    :param prop: the mention
    :param cluster: the cluster
    :return: a numeric value denoting the similarity between the mention to the cluster
    """
    return len([other for other in cluster if other[1] == prop[1]]) / (1.0 * len(cluster))


def eval_clusters(clusters, graph):
    """
    Receives the predicted clusters and the gold standard graph and evaluates (with coref metrics) the predicate
    coreferences
    :param clusters: the predicted clusters
    :param graph: the gold standard graph
    :return: the predicate coreference metrics and the number of singletons
    """
    graph1_ent_mentions = []
    graph2_ent_mentions = clusters

    # Get the gold standard clusters
    for prop in graph.propositions.values():
        mentions_to_consider = set([mention for mention in prop.mentions.values() if mention.indices != [-1]])
        if len(mentions_to_consider) > 0:
            graph1_ent_mentions.append(mentions_to_consider)

    graph1_ent_mentions = [set(map(str, cluster)) for cluster in graph1_ent_mentions]
    graph2_ent_mentions = [set(map(str, cluster)) for cluster in graph2_ent_mentions]

    # Evaluate
    muc1, bcubed1, ceaf1 = muc(graph1_ent_mentions, graph2_ent_mentions), \
                           bcubed(graph1_ent_mentions, graph2_ent_mentions), \
                           ceaf(graph1_ent_mentions, graph2_ent_mentions)
    mela1 = np.mean([muc1, bcubed1, ceaf1])

    singletons = len([cluster for cluster in graph1_ent_mentions if len(cluster) == 1])
    return np.array([muc1, bcubed1, ceaf1, mela1]), singletons


def get_distance_to_root(token, parser):
    """
    Receives a token and returns its distance from the root
    :param token: the token
    :param parser: the spacy wrapper object
    :return: the distance from the token to the root
    """
    dist = 0
    while parser.get_head(token) != token:
        token = parser.get_head(token)
        dist += 1
    return dist


def get_mention_head(mention, parser, graph):
    """
    Gets a mention and returns its head
    :param mention: the mention
    :param parser: the spacy wrapper object
    :param graph: the OKR graph
    :return: the mention head
    """
    distances_to_root = []
    curr_head_and_pos = []
    sentence = graph.sentences[mention.sentence_id]

    joined_sentence = ' '.join(sentence)
    parser.parse(joined_sentence)

    for index in mention.indices:
        child = parser.get_word(index)
        child_lemma = parser.get_lemma(index)
        child_pos = parser.get_pos(index)
        head = parser.get_word(parser.get_head(index))

        if parser.get_head(index) in mention.indices and head != child:
            continue

        distances_to_root.append(get_distance_to_root(index, parser))
        curr_head_and_pos.append((child_lemma, child_pos))

    # Get the closest to the root
    best_index = np.argmin(distances_to_root)
    curr_head, curr_pos = curr_head_and_pos[best_index]

    return curr_head, curr_pos
