"""
This file will contain the coreference algorithms for the automatic pipeline system.
Author: Shany Barhom (based on the code of Hitesh Golchha)
"""

import clustering_common
import os
import sys
import logging

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

#TODO create new algorithm for entity-coreference instead of relying on clustering_common baseline
def cluster_entity_mentions(mention_list):
    """
    currently using clustering_common.
    :param mention_list: the mentions to cluster
    :return: clusters of mentions
    """
    import eval_entity_coref
    return clustering_common.cluster_mentions(mention_list, eval_entity_coref.score)


def cluster_proposition_mentions(mention_list, entities_clustering):
    """
    Receives predicate mentions and entity clusters and perform predicate coreference resolution -
    cluster the predicate mentions in a greedy way: assign each predicate to the first
    cluster with similarity score > 0.5, the score is based on lexical similarity
    and arguments similarity between proposition mentions.
    :param mention_list: the mentions to cluster
    :param entities_clustering: a clustering of entity-mentions to coreference chains. 
        format: list of lists of tuples, each tuple stands for an entity-mention - (mention-id, terms)
    :return: clusters of mentions
    """
    argument_clusters = []
    for cluster in entities_clustering:
        ids_of_cluster = [mention_tuple[0] for mention_tuple in cluster]
        argument_clusters.append(ids_of_cluster)

    return greedy_prop_clustring(mention_list, predicate_score, argument_clusters, lexical_wt=0.75, argument_match_ratio=0.75)


def predicate_score(prop, cluster, argument_clusters, lexical_wt, argument_match_ratio):
    """
    Receives a proposition info (head_lemma, arguments, bare predicate, etc.)
    and a cluster of proposition mentions, and returns a numeric value denoting the
    similarity between the mention to the cluster (based on the lexical similarity and arguments similarity)
    :param prop: the mention
    :param cluster: the cluster
    :param argument_clusters: a clustering of entity-mentions to coreference chains.
    :param lexical_wt: the lexical weight for the scoring function which compares two proposition clusters.
     (the argument weight is 1- lexical_wt)
    :param argument_match_ratio: the minimum argument alignment threshold between propositions
    :return: a numeric value denoting the similarity between the mention to the cluster
    """

    from eval_predicate_coref import some_word_match
    implicit_mentions = 0.0
    if prop[1] != 'IMPLICIT':
        prop_terms = prop[1]
        lexical_matches = 0.0
        for other_prop in cluster:
            implicit_mentions = 0.0
            if other_prop[1] != 'IMPLICIT':
                if some_word_match(other_prop[1], prop_terms):
                    lexical_matches += 1
            else:
                implicit_mentions += 1

        explicit_mentions = 1.0 * (len(cluster) - implicit_mentions)
        lexical_score = lexical_matches / explicit_mentions if explicit_mentions > 0 else 0.0

    else:
        lexical_score = 0
        lexical_wt = 0

    # calculate the ratio of proposition mentions in the cluster who has at
    # least argument_match_ratio percent of arguments corefer with the mention being compared.
    argument_score = len([other for other in cluster if
                            (some_arg_match(other[2], prop[2], argument_clusters, argument_match_ratio))]) / (
                        1.0 * len(cluster))

    return lexical_wt * lexical_score + (1 - lexical_wt) * argument_score


def greedy_prop_clustring(mention_list, score, argument_clusters, lexical_wt, argument_match_ratio):
    """
    Cluster the predicate mentions in a greedy way: assign each predicate to the first
    cluster with similarity score > 0.5. If no such cluster exists, start a new one.
    :param mention_list: the mentions to clustert
    :param score: the score function that receives a mention and a cluster and returns a score
    :param argument_clusters: a clustering of entity-mentions to coreference chains.
    :param lexical_wt: the lexical weight for the scoring function which compares two proposition clusters.
     (the argument weight is 1- lexical_wt)
    :param argument_match_ratio: the minimum argument alignment threshold between propositions
    :return: clusters of mentions
    """
    clusters = []

    for mention in mention_list:
        found_cluster = False
        for cluster in clusters:
            cluster_mention_score = score(mention, cluster, argument_clusters, lexical_wt, argument_match_ratio)
            if cluster_mention_score > 0.5:
                cluster.append(mention)
                found_cluster = True
                break

        if not found_cluster:
            clusters.append([mention])

    return clusters


def some_arg_match(prop_mention1_info, prop_mention2_info, argument_clusters, arg_match_ratio):
    """
    Finds if at least some % of arguments of two propositions are coreferent.
    :param prop_mention1_info: First proposition mention
    :param prop_mention2_info: Second proposition mention
    :param argument_clusters: The clusters of arguments obtained by coreference
    :param arg_match_ratio: criteria of argument overlap = number of aligned arguments / number of arguments(minimum of proposition1 and proposition2)
    :return: True if at least arg_match_ratio of arguments of two propositions are coreferent.
    """

    matched_arguments = 0.0
    prop_mention1_args = [arg_data['sentence_id'] +'_' + arg_key for arg_key,arg_data in
                          prop_mention1_info['Arguments'].iteritems()]

    prop_mention2_args = [arg_data['sentence_id'] +'_' + arg_key for arg_key,arg_data in
                          prop_mention2_info['Arguments'].iteritems() ]

    for m_id1 in prop_mention1_args:
        pair_found = False
        for m_id2 in prop_mention2_args:
            for cluster in argument_clusters:
                if m_id1 in cluster and m_id2 in cluster:
                    matched_arguments += 1
                    pair_found = True
                    break
            if pair_found == True:
                break

    if matched_arguments == 0:
        return False
    elif len(prop_mention1_args) + len(prop_mention2_args) - matched_arguments == 0:
        logging.error('problem here:')
        logging.error('Matched arguments: ' + str(matched_arguments))
        logging.error('Number of arguments in the two propositions: {0}, {1}'.format( str(len(prop_mention1_args)), str(len(prop_mention2_args))))

        return False

    else:
        return (matched_arguments / (len(prop_mention1_args) + len(
            prop_mention2_args) - matched_arguments) >= arg_match_ratio)