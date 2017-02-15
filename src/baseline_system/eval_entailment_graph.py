"""
Author: Vered Shwartz

    Receives a validation set and a test set. Tunes the hyper-parameters of the baseline system on the validation set
    and evaluates performance on the test set.
    Computes the following scores:

    1) Average F1 score for the entities graphs
    2) Average F1 score for the predicates graphs
"""

import sys
sys.path.append('../agreement')

import numpy as np

from entailment_graph import *
from entity_entailment import *
from predicate_entailment import *


def evaluate_entailment(val_graphs, test_graphs):
    """
    Evaluation for the entailment graph: entities and predicates
    :param val_graphs: the gold standard annotations for the validation set
    :param test_graphs: the gold standard annotations for the test set
    :return: the entity graph and predicate graph F1 scores
    """
    entities_f1 = evaluate_entity_entailment(val_graphs, test_graphs)
    propositions_f1 = evaluate_predicate_entailment(val_graphs, test_graphs)
    return entities_f1, propositions_f1


def evaluate_predicate_entailment(val_graphs, test_graphs):
    """
    Evaluation for the predicate entailment graph
    :param val_graphs: the gold standard annotations for the validation set
    :param test_graphs: the gold standard annotations for the test set
    :return: the predicate entailment F1 score
    """

    # Load the resource for predicate entailment
    print 'Loading predicate entailment resource...'
    pred_ent = PredicateEntailmentBaseline('../../resources/predicate_rules_berant.db')

    # Tune the threshold for the validation set
    print 'Tuning the threshold...'
    thresholds = [-10000.0] # [-10000.0, 10000.0] + [-4 + 0.1 * i for i in range(80)]
    f1_scores = []

    for threshold in thresholds:
        pred_ent.set_threshold(threshold)
        curr_f1_scores = []

        for val_graph in val_graphs:
            val_pred = predict_predicate_entailment(pred_ent, val_graph)
            curr_f1_scores.append(compute_predicate_f1(val_graph, val_pred))

        f1 = np.mean(curr_f1_scores)
        print 'threshold = %.3f, f1 = %.3f' % (threshold, f1)
        f1_scores.append(f1)

    best_threshold = thresholds[np.argmax(f1_scores)]

    # Compute Kappa on the test set
    print 'best threshold for predicate entailment: %.3f' % best_threshold
    pred_ent.set_threshold(best_threshold)

    propositions_f1_scores = []
    for test_graph in test_graphs:
        test_pred = predict_predicate_entailment(pred_ent, test_graph)
        propositions_f1_scores.append(compute_predicate_f1(test_graph, test_pred))

    return np.mean(propositions_f1_scores)


def predict_predicate_entailment(pred_ent, gold):
    """
    Creates a new structure which is identical to the gold standard
    except for the entailment graph, in which edges represent rules above threshold
    :param pred_ent: the predicate entailment finder
    :param gold: the gold standard graph
    :return: a new structure which is identical to the gold standard
    except for the entailment graph, in which edges represent rules above threshold
    """
    pred = gold.clone()

    # Copy all the structure from the gold except for the entailment graph, and add edges to the entailment graph
    # if there is a rule above threshold
    for p_id, prop in gold.propositions.iteritems():
        pred.propositions[p_id].entailment_graph.mentions_graph = \
            [(str(m1), str(m2)) for m1 in prop.mentions.values() for m2 in prop.mentions.values()
             if pred_ent.is_entailing(m1.template, m2.template)]

    return pred


def evaluate_entity_entailment(val_graphs, test_graphs):
    """
    Compute the agreement for the entity entailment graph
    :param val_graphs: the gold standard annotations for the validation set
    :param test_graphs: the gold standard annotations for the test set
    :return: the entity entailment F1 score
    """

    # Load the resource for entity entailment
    print 'Loading entity entailment resource...'
    ent_ent = EntityEntailmentBaseline('../../resources/hypenet_unigram.db',
                                       '../../resources/hypenet_ngram.db',
                                       '../../resources/resource_dictionary.db')

    # Tune the threshold for the validation set
    print 'Tuning the threshold...'
    unigram_thresholds = [0.5] # [0.5 + 0.01 * i for i in range(50)]
    ngram_thresholds = [0.5] # [0.5 + 0.01 * i for i in range(50)]
    thresholds = [(u, n) for u in unigram_thresholds for n in ngram_thresholds]
    f1_scores = []

    for (unigram_threshold, ngram_threshold) in thresholds:
        ent_ent.set_unigram_threshold(unigram_threshold)
        ent_ent.set_ngram_threshold(ngram_threshold)
        curr_f1_scores = []

        for val_graph in val_graphs:
            val_pred = predict_entity_entailment(ent_ent, val_graph)
            curr_f1_scores.append(compute_entities_f1(val_graph, val_pred))

        f1 = np.mean(curr_f1_scores)
        print 'unigram threshold = %.3f, ngram_threshold = %.3f, f1 = %.3f' % (unigram_threshold, ngram_threshold, f1)
        f1_scores.append(f1)

    (best_unigram_threshold, best_ngram_threshold) = thresholds[np.argmax(f1_scores)]

    # Compute Kappa on the test set
    print 'best thresholds for entity entailment: unigram = %.3f, ngram = %.3f' % \
          (best_unigram_threshold, best_ngram_threshold)
    ent_ent.set_unigram_threshold(best_unigram_threshold)
    ent_ent.set_ngram_threshold(best_ngram_threshold)

    entity_f1_scores = []

    for test_graph in test_graphs:
        test_pred = predict_entity_entailment(ent_ent, test_graph)
        entity_f1_scores.append(compute_entities_f1(test_graph, test_pred))

    return np.mean(entity_f1_scores)


def predict_entity_entailment(ent_ent, gold):
    """
    Creates a new structure which is identical to the gold standard
    except for the entailment graph, in which edges represent rules above threshold
    :param ent_ent: the entity entailment finder
    :param gold: the gold standard graph
    :return: a new structure which is identical to the gold standard
    except for the entailment graph, in which edges represent rules above threshold
    """
    pred = gold.clone()

    # Copy all the structure from the gold except for the entailment graph, and add edges to the entailment graph
    # if there is a rule above threshold
    for e_id, entity in gold.entities.iteritems():
        pred.entities[e_id].entailment_graph.mentions_graph = \
            [(str(m1), str(m2)) for m1 in entity.mentions.values() for m2 in entity.mentions.values()
             if ent_ent.is_entailing(m1.terms, m2.terms)]

    return pred