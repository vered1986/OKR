"""
Author: Vered Shwartz

    Receives two annotated graphs and computes the agreement on the entailment graph
    This script returns the following scores:

    1) F1 score for the agreement on the entities graph
    2) F1 score for the agreement on the propositions
"""

import numpy as np

from sklearn.metrics import precision_recall_fscore_support


def compute_entailment_graph_agreement(graph1, graph2):
    """
    Compute the agreement for the entailment graph: entities, arguments and predicates
    :param graph1: the first annotator's graph
    :param graph2: the second annotator's graph
    :return:
    """

    # Compute the agreement for the entity entailment graph, for each entity, and return the average
    # (remove entities with one mention)
    entities_1_gold_f1 = compute_entities_f1(graph1, graph2)
    entities_2_gold_f1 = compute_entities_f1(graph2, graph1)
    entities_f1 = (entities_1_gold_f1 + entities_2_gold_f1) / 2.0

    # Compute the agreement for the predicate entailment graph, for each predicate, and return the average
    # (remove predicates with one mention)
    props_1_gold_f1 = compute_predicate_f1(graph1, graph2)
    props_2_gold_f1 = compute_predicate_f1(graph2, graph1)
    propositions_f1 = (props_1_gold_f1 + props_2_gold_f1) / 2.0

    # TODO: implement
    arguments_f1 = 0.0

    # Compute the consensual graphs: TODO
    consensual_graph1 = graph1
    consensual_graph2 = graph2

    return entities_f1, arguments_f1, propositions_f1, consensual_graph1, consensual_graph2


def compute_entities_f1(gold_graph, pred_graph):
    """
    Compute the agreement for the entity entailment graph, for each entity, and return the average
    :param gold_graph: the first annotator's graph
    :param pred_graph: the second annotator's graph
    :return: the entity edges' mean F1 score
    """

    # Get all the possible edges in the entity entailment graph
    all_edges = {str(entity): set([(str(m1), str(m2))
                                   for m1 in entity.mentions.values()
                                   for m2 in entity.mentions.values() if m1 != m2])
                 for entity in gold_graph.entities.values() if len(entity.mentions) > 1}

    # Get the binary predictions/gold for these edges
    str_entities_gold = { entity : str(entity) for entity in gold_graph.entities.values() }
    entity_entailments_gold = {str_entities_gold[entity]:
                                [1 if (m1, m2) in set(entity.entailment_graph.mentions_graph) else 0
                                 for (m1, m2) in all_edges[str_entities_gold[entity]]]
                            for entity in gold_graph.entities.values() if str_entities_gold[entity] in all_edges.keys()}

    str_entities_pred = { entity : str(entity) for entity in pred_graph.entities.values() }
    entity_entailments_pred = {str_entities_pred[entity]:
                                [1 if (m1, m2) in set(entity.entailment_graph.mentions_graph) else 0
                                 for (m1, m2) in all_edges[str_entities_pred[entity]]]
                            for entity in pred_graph.entities.values() if str_entities_pred[entity] in all_edges.keys()}

    mutual_entities = list(set(entity_entailments_gold.keys()).intersection(entity_entailments_pred.keys()))

    # If both graphs contain no entailments, the score should be one
    f1 = np.mean([precision_recall_fscore_support(entity_entailments_gold[entity], entity_entailments_pred[entity],
                                                  average='binary')[2]
                  if np.sum(entity_entailments_gold[entity]) > 0 or np.sum(entity_entailments_pred[entity]) > 0 else 1.0
                  for entity in mutual_entities])

    return f1


def compute_predicate_f1(gold_graph, pred_graph):
    """
    Compute the agreement for the predicate entailment graph, for each predicate, and return the average
    :param gold_graph: the first annotator's graph
    :param pred_graph: the second annotator's graph
    :return: the predicates' edges mean F1 score
    """

    # Use only explicit mentions with more than one proposition
    str_prop_gold = { prop : str(prop) for prop in gold_graph.propositions.values()
                   if len(set(map(str, prop.mentions.values()))) > 1 }
    str_prop_pred = { prop : str(prop) for prop in pred_graph.propositions.values()
                  if len(set(map(str, prop.mentions.values()))) > 1 }

    # Get all the possible edges in the entity entailment graph
    all_edges = {str(prop) : set([(str(m1), str(m2))
                                   for m1 in prop.mentions.values()
                                   for m2 in prop.mentions.values() if str(m1) != str(m2)])
                 for prop in gold_graph.propositions.values()
                 if len(set(map(str, prop.mentions.values()))) > 1}

    # Get the binary predictions/gold for these edges
    prop_entailments_gold = {str_prop_gold[prop]:
                              [1 if (m1, m2) in set(prop.entailment_graph.mentions_graph) else 0
                               for (m1, m2) in all_edges[str_prop_gold[prop]]]
                          for prop in str_prop_gold.keys()
                          if str_prop_gold[prop] in all_edges.keys()}

    prop_entailments_pred = {str_prop_pred[prop]:
                              [1 if (m1, m2) in set(prop.entailment_graph.mentions_graph) else 0
                               for (m1, m2) in all_edges[str_prop_pred[prop]]]
                          for prop in str_prop_pred.keys()
                          if str_prop_pred[prop] in all_edges.keys()}

    mutual_props = list(set(prop_entailments_gold.keys()).intersection(prop_entailments_pred.keys()))

    # If both graphs contain no entailments, the score should be one
    f1 = np.mean([precision_recall_fscore_support(prop_entailments_gold[entity], prop_entailments_pred[entity],
                                                  average='binary')[2]
                  if np.sum(prop_entailments_gold[entity]) > 0 or np.sum(prop_entailments_pred[entity]) > 0 else 1.0
                  for entity in mutual_props])

    return f1
