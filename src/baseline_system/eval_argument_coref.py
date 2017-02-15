"""
Receives a test set and evaluates performance of argument coreference.
Argument coreference is simply predicted by marking coreference if and only if the arguments are both mentions
of the same entity coreference chain.

Author: Shyam Upadhyay
"""

import sys
sys.path.append('../common')

from okr import *
from entity_coref import *
from eval_entity_coref import *
from clustering_common import cluster_mentions


def evaluate_argument_coref(test_graphs):
    """
    Receives the OKR test graphs and evaluates them for argument coreference
    :param test_graphs: the OKR test graphs
    :return: the coreference scores: MUC, B-CUBED, CEAF and MELA (CoNLL F1).
    """
    scores = []

    for graph in test_graphs:

        arg_clustering = {}
        for prop_id, prop in graph.propositions.iteritems():

            # Cluster the arguments
            all_args = [arg for mention in prop.mentions.values() for arg in mention.argument_mentions.values()]
            score = lambda mention, cluster : same_entity(cluster, mention, graph)
            clusters = cluster_mentions(all_args, score)
            clusters = [set([str(mention) for mention in cluster]) for cluster in clusters]
            arg_clustering[prop_id] = clusters

        # Evaluate
        curr_scores = eval_clusters(graph, arg_clustering)
        scores.append(curr_scores)

    scores = np.mean(scores, axis=0).tolist()

    return scores


def eval_clusters(gold, arg_clustering):
    """
    Receives an annotated graph and a predictaed clustering of arguments for that graph and computes the
    evaluation on the argument coreference
    :param gold: the gold standard OKR graph
    :param arg_clusters: predicted argument clustering - as a dictionary of
    proposition_id:argument_clusters (list of sets)
    :return: MUC, B-CUBED, CEAF and MELA scores.
    """

    # Get argument mentions
    gold_arg_mentions_dicts = { prop_id : [{ m_id : str(mention)
                                               for m_id, mention in mention.argument_mentions.iteritems()}
                                             for mention in prop.mentions.values()]
                                  for prop_id, prop in gold.propositions.iteritems() }

    # Clusters of arguments per proposition
    gold_arg_mentions = { p_id : [set([mention_dict[str(arg_num)]
                                         for mention_dict in mention_lst if str(arg_num) in mention_dict])
                                    for arg_num in range(0, 10)]
                            for p_id, mention_lst in gold_arg_mentions_dicts.iteritems()}

    # Remove empty arguments
    gold_arg_mentions = {k: [s for s in v if len(s) > 0] for k, v in gold_arg_mentions.iteritems()}

    pred_arg_mentions = arg_clustering

    # Within each proposition, compute coreference scores:
    scores = []

    for prop_id in gold.propositions.keys():

        if prop_id not in gold_arg_mentions:
            continue

        if prop_id not in pred_arg_mentions:
            continue

        # No arguments
        if len(gold_arg_mentions[prop_id]) == 0 or len(pred_arg_mentions[prop_id]) == 0:
            continue

        muc1, bcubed1, ceaf1 = muc(gold_arg_mentions[prop_id], pred_arg_mentions[prop_id]), \
                           bcubed(gold_arg_mentions[prop_id], pred_arg_mentions[prop_id]), \
                           ceaf(gold_arg_mentions[prop_id], pred_arg_mentions[prop_id])
        mela1 = np.mean([muc1, bcubed1, ceaf1])

        scores.append([muc1, bcubed1, ceaf1, mela1])

    return np.mean(scores, axis=0).tolist()


def same_entity(cluster, argument, graph):
    """
    Receives a cluster and an argument mention and returns whether this argument mention
    is similar to the cluster, i.e. if the arguments in the clusters are all mentions of the same
    coreference chain.
    :param cluster: the cluster of mentions
    :param argument: the mention
    :param graph: the OKR graph
    :return: whether this argument mention is similar to the cluster.
    """
    match = []

    for m in cluster:

        # Entity argument
        if m.parent_id in graph.entities and argument.parent_id in graph.entities:
            val = str(graph.entities[m.parent_id]) == str(graph.entities[argument.parent_id])

        # Predicate argument
        elif m.parent_id in graph.propositions and argument.parent_id in graph.propositions:
            val = str(graph.propositions[m.parent_id]) == str(graph.propositions[argument.parent_id])

        else:
            return False

        match.append(val)

    return all(match)

