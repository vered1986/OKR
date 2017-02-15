"""
Author: Vered Shwartz

    Receives two annotated graphs and computes the agreement on the predicate coreference.
    This script returns the following scores:

    1) MUC (Vilain et al., 1995) - a link-based metric.
    2) B-CUBED (Bagga and Baldwin, 1998) - a mention-based metric.
    3) CEAF (Constrained Entity Aligned F-measure) metric (Luo, 2005) - an entity-based metric.
    4) CoNLL F1/ MELA (Denis and Baldridge, 2009) - an average of these three measures.
"""

import numpy as np

from munkres import *
from entity_coref import muc, bcubed, ceaf, pad_to_square


def compute_predicate_coref_agreement(graph1, graph2):
    """
    Receives two annotated graphs and computes the agreement on the predicate coreference:
    1) MUC (Vilain et al., 1995) 2) B-CUBED (Bagga and Baldwin, 1998)
    2) B-CUBED (Bagga and Baldwin, 1998) - a mention-based metric.
    3) CEAF (Luo, 2005) 4) MELA (Denis and Baldridge, 2009)
    :param graph1: the first annotator's graph
    :param graph2: the second annotator's graph
    :return: MUC, B-CUBED, CEAF and MELA scores. Each score is computed twice, each time
    a different annotator is considered as the gold, and the averaged score is returned.
    In addition, it returns the optimal alignment of predicates from graph1 to predicates from graph2.
    """

    # Get predicate mentions
    # TODO: currently there is a problem with predicates that participate in more than one proposition: they
    # are only counted once. We need to find a way to represent them.
    graph1_pred_mentions = [set(map(str, proposition.mentions.values())) for proposition in graph1.propositions.values()]
    graph2_pred_mentions = [set(map(str, proposition.mentions.values())) for proposition in graph2.propositions.values()]

    # Compute twice, each time considering a different annotator as the gold, and return the average among
    # each measure
    muc1, bcubed1, ceaf1 = muc(graph1_pred_mentions, graph2_pred_mentions), \
                           bcubed(graph1_pred_mentions, graph2_pred_mentions), \
                           ceaf(graph1_pred_mentions, graph2_pred_mentions)
    mela1 = np.mean([muc1, bcubed1, ceaf1])

    muc2, bcubed2, ceaf2 = muc(graph2_pred_mentions, graph1_pred_mentions), \
                           bcubed(graph2_pred_mentions, graph1_pred_mentions), \
                           ceaf(graph2_pred_mentions, graph1_pred_mentions)
    mela2 = np.mean([muc2, bcubed2, ceaf2])

    muc_score = np.mean([muc1, muc2])
    bcubed_score = np.mean([bcubed1, bcubed2])
    ceaf_score = np.mean([ceaf1, ceaf2])
    mela_score = np.mean([mela1, mela2])

    # Compute the consensual graphs - find the maximum alignment between predicate clusters and keep only
    # the intersection between the clusters in each graph's aligned clusters
    cost = -np.vstack([np.array([len(s1.intersection(s2)) for s1 in graph1_pred_mentions])
                       for s2 in graph2_pred_mentions])
    m = Munkres()
    cost = pad_to_square(cost)
    indices = m.compute(cost)
    rev_optimal_alignment = { row : col for row, col in indices }
    optimal_alignment = { col : row for row, col in indices }
    id_alignment= { graph1.propositions.keys()[k] : graph2.propositions.keys()[v]
                    for k, v in optimal_alignment.iteritems()
                    if k < len(graph1.propositions) and v < len(graph2.propositions) }

    s1_to_s2 = { graph1.propositions.keys()[i] : s1.intersection(graph2_pred_mentions[optimal_alignment[i]])
                 for i, s1 in enumerate(graph1_pred_mentions)
                 if optimal_alignment[i] < len(graph2_pred_mentions) }

    s2_to_s1 = { graph2.propositions.keys()[i] : s2.intersection(graph1_pred_mentions[rev_optimal_alignment[i]])
                 for i, s2 in enumerate(graph2_pred_mentions)
                 if rev_optimal_alignment[i] < len(graph1_pred_mentions) }

    consensual_graph1 = filter_clusters(graph1, s1_to_s2)
    consensual_graph2 = filter_clusters(graph2, s2_to_s1)

    return muc_score, bcubed_score, ceaf_score, mela_score, consensual_graph1, consensual_graph2, id_alignment


def filter_clusters(graph, consensual_clusters):
    """
    Remove propositions that are not consensual
    :param graph: the original graph
    :param consensual_clusters:
    :return: the graph, containing only the consensual clusters
    """

    consensual_graph = graph.clone()
    removed = []

    for prop_id, prop in consensual_graph.propositions.iteritems():

        if prop_id not in consensual_clusters.keys():
            removed.append(prop_id)
            continue

        # Filter mentions
        prop.mentions = { id : mention for id, mention in prop.mentions.iteritems()
                            if str(mention) in consensual_clusters[prop_id] }

        # Remove them also from the entailment graph
        prop.entailment_graph.mentions_graph = [(m1, m2) for (m1, m2)
                                                  in prop.entailment_graph.mentions_graph
                                                  if m1 in consensual_clusters[prop_id]
                                                  and m2 in consensual_clusters[prop_id]]

        # Remove propositions without mentions
        if len(prop.mentions) == 0:
            removed.append(prop_id)

    # Remove propositions without mentions
    consensual_graph.propositions = { entity_id : entity for entity_id, entity in consensual_graph.propositions.items()
                                  if entity_id not in removed }

    return consensual_graph
