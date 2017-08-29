"""
Author: Rachel Wities and Vered Shwartz

    Receives two annotated graphs and computes the agreement on the predicate mentions.
    We average the accuracy of the two annotators, each computed while taking the other as a gold reference.
"""
import sys

sys.path.append('../common')

from mention_common import *
from constants import NULL_VALUE
from filter_propositions import filter_verbal, filter_non_verbal


def compute_predicate_mention_agreement(graph1, graph2):
    """
    Compute predicate mention agreement on two graphs
    :param graph1: the first annotator's graph
    :param graph2: the second annotator's graph
    :return predicate mention accuracy and the consensual graphs
    """

    # Get the consensual mentions and the mentions in each graph
    consensual_mentions, graph1_prop_mentions, graph2_prop_mentions = extract_consensual_mentions(graph1, graph2)

    # Compute the accuracy, each time taking one annotator as the gold
    accuracy1 = len(consensual_mentions) * 1.0 / len(graph1_prop_mentions) if len(graph1_prop_mentions) > 0 else 0.0
    accuracy2 = len(consensual_mentions) * 1.0 / len(graph2_prop_mentions) if len(graph1_prop_mentions) > 0 else 0.0

    prop_mention_acc = (accuracy1 + accuracy2) / 2

    consensual_graph1 = filter_mentions(graph1, consensual_mentions)
    consensual_graph2 = filter_mentions(graph2, consensual_mentions)

    return prop_mention_acc, consensual_graph1, consensual_graph2


def compute_predicate_mention_agreement_verbal(graph1, graph2):
    """
    Compute predicate mention agreement only on verbal predicates
    :param graph1: the first annotator's graph
    :param graph2: the second annotator's graph
    :return predicate mention accuracy on verbal predicates
    """
    verbal_graph1 = filter_verbal(graph1)
    verbal_graph2 = filter_verbal(graph2)
    accuracy, _, _ = compute_predicate_mention_agreement(verbal_graph1, verbal_graph2)
    return accuracy


def compute_predicate_mention_agreement_non_verbal(graph1, graph2):
    """
    Compute predicate mention agreement only on non verbal predicates
    :param graph1: the first annotator's graph
    :param graph2: the second annotator's graph
    :return predicate mention accuracy on non verbal predicates
    """
    non_verbal_graph1 = filter_non_verbal(graph1)
    non_verbal_graph2 = filter_non_verbal(graph2)
    accuracy, _, _ = compute_predicate_mention_agreement(non_verbal_graph1, non_verbal_graph2)
    return accuracy


def filter_mentions(graph, consensual_mentions):
    """
    Remove mentions that are not consensual
    :param graph: the original graph
    :param consensual_mentions: the mentions that both annotators agreed on
    :return: the graph, containing only the consensual mentions
    """
    consensual_graph = graph.clone()

    for prop in consensual_graph.propositions.values():
        prop.mentions = { id : mention for id, mention in prop.mentions.iteritems()
                          if str(mention) in consensual_mentions}

        # Remove them also from the entailment graph
        if prop.entailment_graph != NULL_VALUE:
            prop.entailment_graph.mentions_graph = [(m1, m2) for (m1, m2) in prop.entailment_graph.mentions_graph
                                                    if m1 in consensual_mentions and m2 in consensual_mentions]

        # Remove propositions with no mentions
        if len(prop.mentions) == 0:
            consensual_graph.propositions.pop(prop.id, None)

    return consensual_graph


def extract_consensual_mentions(graph1, graph2):
    """
    Receives two graphs, and returns the consensual predicate mentions, and the predicate mentions in each graph.
    :param graph1: the first annotator's graph
    :param graph2: the second annotator's graph
    :return the consensual predicate mentions, and the predicate mentions in each graph
    """

    # Get the predicate mentions in both graphs
    graph1_prop_mentions = set.union(*[set(map(str, prop.mentions.values())) for prop in graph1.propositions.values()])
    graph2_prop_mentions = set.union(*[set(map(str, prop.mentions.values())) for prop in graph2.propositions.values()])

    # Exclude sentence that weren't annotated by both annotators
    common_sentences = set([x.split('[')[0] for x in graph1_prop_mentions]).intersection(
        set([x.split('[')[0] for x in graph2_prop_mentions]))

    graph1_prop_mentions = set([a for a in graph1_prop_mentions if a.split('[')[0] in common_sentences])
    graph2_prop_mentions = set([a for a in graph2_prop_mentions if a.split('[')[0] in common_sentences])

    # Exclude ignored words
    # TODO: Rachel - document ignored words
    if not graph2.ignored_indices == None:
        graph1_prop_mentions = set([a for a in graph1_prop_mentions if len(overlap_set(a, graph2.ignored_indices)) == 0])

    if not graph1.ignored_indices == None:
        graph2_prop_mentions = set([a for a in graph2_prop_mentions if len(overlap_set(a, graph1.ignored_indices)) == 0])

    # Compute the accuracy, each time treating a different annotator as the gold
    consensual_mentions = graph1_prop_mentions.intersection(graph2_prop_mentions)

    return consensual_mentions, graph1_prop_mentions, graph2_prop_mentions
