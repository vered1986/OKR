"""
Author: Rachel Wities and Vered Shwartz

    Receives two annotated graphs and computes the agreement on the entity mentions.
    We average the accuracy of the two annotators, each computed while taking the other as a gold reference.
"""
import sys

sys.path.append('../common')

from mention_common import *


def compute_entity_mention_agreement(graph1, graph2):
    """
    Compute entity mention agreement on two graphs
    :param graph1: the first annotator's graph
    :param graph2: the second annotator's graph
    :return entity mention accuracy and the consensual graphs
    """

    # Get the consensual mentions and the mentions in each graph
    consensual_mentions, graph1_ent_mentions, graph2_ent_mentions = extract_consensual_mentions(graph1, graph2)

    # Compute the accuracy, each time taking one annotator as the gold
    accuracy1 = len(consensual_mentions) * 1.0 / len(graph1_ent_mentions)
    accuracy2 = len(consensual_mentions) * 1.0 / len(graph2_ent_mentions)

    entity_mention_acc = (accuracy1 + accuracy2) / 2

    consensual_graph1 = filter_mentions(graph1, consensual_mentions)
    consensual_graph2 = filter_mentions(graph2, consensual_mentions)

    return entity_mention_acc, consensual_graph1, consensual_graph2


def filter_mentions(graph, consensual_mentions):
    """
    Remove mentions that are not consensual
    :param graph: the original graph
    :param consensual_mentions: the mentions that both annotators agreed on
    :return: the graph, containing only the consensual mentions
    """

    consensual_graph = graph.clone()

    for entity in consensual_graph.entities.values():
        entity.mentions = {id: mention for id, mention in entity.mentions.iteritems()
                           if str(mention) in consensual_mentions}

        # Remove them also from the entailment graph
        entity.entailment_graph.mentions_graph = [(m1, m2) for (m1, m2) in entity.entailment_graph.mentions_graph
                                                  if m1 in consensual_mentions and m2 in consensual_mentions]

        # Remove entities with no mentions
        if len(entity.mentions) == 0:
            consensual_graph.entities.pop(entity.id, None)

    return consensual_graph


def extract_consensual_mentions(graph1, graph2):
    """
    Receives two graphs, and returns the consensual entity mentions, and the entity mentions in each graph.
    :param graph1: the first annotator's graph
    :param graph2: the second annotator's graph
    :return the consensual entity mentions, and the entity mentions in each graph
    """

    # Get the entity mentions in both graphs
    graph1_ent_mentions = set.union(*[set(map(str, entity.mentions.values())) for entity in graph1.entities.values()])
    graph2_ent_mentions = set.union(*[set(map(str, entity.mentions.values())) for entity in graph2.entities.values()])

    # Exclude sentence that weren't anotated by both:
    common_sentences = set([x.split('[')[0] for x in graph1_ent_mentions]).intersection(
        set([x.split('[')[0] for x in graph2_ent_mentions]))
    graph1_ent_mentions = set([a for a in graph1_ent_mentions if a.split('[')[0] in common_sentences])
    graph2_ent_mentions = set([a for a in graph2_ent_mentions if a.split('[')[0] in common_sentences])

    # Exclude ignored_words, for versions 5 and up:
    if not graph2.ignored_indices == None:
        graph1_ent_mentions = set([a for a in graph1_ent_mentions if len(overlap_set(a, graph2.ignored_indices)) == 0])

    if not graph1.ignored_indices == None:
        graph2_ent_mentions = set([a for a in graph2_ent_mentions if len(overlap_set(a, graph1.ignored_indices)) == 0])

    consensual_mentions = graph1_ent_mentions.intersection(graph2_ent_mentions)

    return consensual_mentions, graph1_ent_mentions, graph2_ent_mentions