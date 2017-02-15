"""
Receives a test set and evaluates performance of argument mention.
For argument detection we attach the components (entities and propositions) as arguments of predicates when the
components are syntactically dependent on them.
Author: Gabi Stanovsky
"""
import sys
sys.path.append('../common')
sys.path.append("../agreement")

import logging
import numpy as np

from argument_mention import compute_argument_mention_agreement

logging.basicConfig(level=logging.DEBUG)


def evaluate_argument_mention(test_graphs, threshold):
    """
    Compute performance of argument mentions on a list of graphs.
    :param test_graphs: the OKR graphs for the test sets
    :param threshold: the distance to the predicate under which components are considered argument mentions
    :return performance of argument mentions on a list of graphs
    """
    pred_graphs = [predict_argument_mention(test_graph, threshold) for test_graph in test_graphs]
    return np.mean([compute_argument_mention_agreement(test_graph, pred_graph)[0]
                    for test_graph, pred_graph in zip(test_graphs, pred_graphs)])


def predict_argument_mention(test_graph, threshold):
    """
    Predict argument mention on a single graph.
    :param test_graph: the OKR graph
    :param threshold: the distance to the predicate under which components are considered argument mentions
    :return a predicted version of the input graph.
    """
    ret = test_graph.clone()

    for prop_id, prop in ret.propositions.iteritems():
        for mention_id, mention in prop.mentions.iteritems():
            pred_indices = mention.indices
            sent_id = mention.sentence_id
            possible_ents = get_entity_mention_by_sent_id(test_graph, sent_id)
            arg_ments = [MockArgumentMention(sent_id, pred_indices, ent.indices)
                         for ent in get_close_entity_mentions(pred_indices, possible_ents, threshold)]

            # override -- this is the only field we predict
            mention.argument_mentions = dict(zip(range(len(arg_ments)), arg_ments))

    return ret


def get_close_entity_mentions(pred_indices, possible_ents, threshold):
    """
    Get entities up to a distance threshold from pred_indices
    :param pred_indices: the predicate indices
    :param possible_ents: the argument mention candidates
    :param threshold: the distance to the predicate under which components are considered argument mentions
    """
    ret = set([ent for ent in possible_ents.values() for ent_ind in ent.indices for pred_ind in pred_indices
               if abs(ent_ind - pred_ind) <= threshold])

    return list(ret)


def calibrate_threshold(test_graphs):
    """
    Find best threshold for distance
    :param test_graphs: the OKR graphs
    :return best threshold for distance
    """
    best_threshold = None
    best_result = None
    for threhold in range(1, 50):
        cur_res = evaluate_argument_mention(test_graphs, threhold)
        if (best_result is None) or (cur_res > best_result):
            best_result = cur_res
            best_threshold = threhold
    return (best_threshold, best_result)


class MockArgumentMention:
    """
    Simple override for agreement purposes
    overrides str_p to imitate an argument.
    """

    def __init__(self, sent_id, pred_indices, arg_indices):
        """
        str_p_ret is the only real value in this argument, and will be returned from calling the
        str_p function, regardless of its input.
        """
        self.str_p_ret = "{0}[{1}]_{0}[{2}]".format(sent_id,
                                                    ', '.join(map(str, pred_indices)),
                                                    ', '.join(map(str, arg_indices)))

    def str_p(self, x):
        return self.str_p_ret


def get_entity_mention_by_sent_id(graph, sent_id):
    """
    Returns all entity mentions in a given sentence.
    :param graph: the OKR graph
    :param sent_id: the sentence ID
    :return all entity mentions in a given sentence
    """
    return { ent_id : mention for ent_id, ent in graph.entities.iteritems()
             for mention_id, mention in ent.mentions.iteritems()
             if mention.sentence_id == sent_id }