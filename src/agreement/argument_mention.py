"""
Author: Rachel Wities and Vered Shwartz

    Receives two annotated graphs and computes the agreement on the propositions arguments mentions.
    We average the accuracy of the two annotators, each computed while taking the other as a gold reference.
"""
import sys

sys.path.append('../common')

from mention_common import *


def compute_argument_mention_agreement(graph1, graph2):
    """
    Compute argument mention agreement on two graphs
    :param graph1: the first annotator's graph
    :param graph2: the second annotator's graph
    :return argument mention accuracy and the consensual graphs
    """

    # Get the consensual mentions and the mentions in each graph
    consensual_mentions, graph1_arg_mentions, graph2_arg_mentions = extract_consensual_mentions(graph1, graph2)

    # Compute the accuracy, each time taking one annotator as the gold
    if len(graph1_arg_mentions) == 0 and len(graph2_arg_mentions) == 0:
        return 1.0

    accuracy1 = len(consensual_mentions) * 1.0 / len(graph1_arg_mentions) if len(graph1_arg_mentions) else 0.0
    accuracy2 = len(consensual_mentions) * 1.0 / len(graph2_arg_mentions) if len(graph2_arg_mentions) else 0.0

    arg_mention_acc = (accuracy1 + accuracy2) / 2

    consensual_graph1 = filter_mentions(graph1, consensual_mentions)
    consensual_graph2 = filter_mentions(graph2, consensual_mentions)

    return arg_mention_acc, consensual_graph1, consensual_graph2


def filter_mentions(graph, consensual_mentions):
    """
    Remove mentions that are not consensual
    :param graph: the original graph
    :param consensual_mentions: the mentions that both annotators agreed on
    :return: the graph, containing only the consensual mentions
    """

    consensual_graph = graph.clone()

    for prop in consensual_graph.propositions.values():
        for mention in prop.mentions.values():
            mention.argument_mentions = { id : arg_mention for id, arg_mention in mention.argument_mentions.iteritems()
                                          if arg_mention.str_p(mention) in consensual_mentions }

        # TODO: consensual argument entailment is missing!

    return consensual_graph


def extract_consensual_mentions(graph1, graph2):
    """
    Receives two graphs, and returns the consensual argument mentions, and the argument mentions in each graph.
    :param graph1: the first annotator's graph
    :param graph2: the second annotator's graph
    :return the consensual argument mentions, and the entity mentions in each graph
    """

    # Get the argument mentions in both graphs
    graph1_arg_mentions = set.union(*[set.union(*[set([arg.str_p(mention) for arg in mention.argument_mentions.values()])
                                                  for mention in  prop.mentions.values()])
                                      for prop in graph1.propositions.values()])

    graph2_arg_mentions = set.union(*[set.union(*[set([arg.str_p(mention) for arg in mention.argument_mentions.values()])
                                                  for mention in  prop.mentions.values()])
                                      for prop in graph2.propositions.values()])

    # Exclude sentence that weren't anotated by both:
    common_sentences = set([x.split('[')[0] for x in graph1_arg_mentions]).intersection(
        set([x.split('[')[0] for x in graph2_arg_mentions]))
    graph1_arg_mentions = set([a for a in graph1_arg_mentions if a.split('[')[0] in common_sentences])
    graph2_arg_mentions = set([a for a in graph2_arg_mentions if a.split('[')[0] in common_sentences])

    # Exclude ignored_words, for versions 5 and up:
    if not graph2.ignored_indices == None:
        graph1_arg_mentions = set([a for a in graph1_arg_mentions if len(overlap_set(a, graph2.ignored_indices)) == 0])

    if not graph1.ignored_indices == None:
        graph2_arg_mentions = set([a for a in graph2_arg_mentions if len(overlap_set(a, graph1.ignored_indices)) == 0])

    consensual_mentions = graph1_arg_mentions.intersection(graph2_arg_mentions)

    return consensual_mentions, graph1_arg_mentions, graph2_arg_mentions

