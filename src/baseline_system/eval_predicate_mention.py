"""
Receives a test set and evaluates performance of predicate mention.
For proposition extraction we use propositions extracted from PropS (Stanovsky et al., 2016),
where non-restrictive arguments were reduced following (Stanovsky and Dagan, 2016).

Author: Gabi Stanovsky
"""
import sys
sys.path.append('../common')
sys.path.append('../agreement')

import nltk
import logging
import numpy as np

from constants import NULL_VALUE
from filter_propositions import cram_proposition_mentions
from filter_propositions import filter_verbal, filter_non_verbal
from predicate_mention import compute_predicate_mention_agreement
from predicate_mention import compute_predicate_mention_agreement
from okr import PropositionMention, Proposition, load_graphs_from_folder


logging.basicConfig(level = logging.INFO)


def evaluate_predicate_mention(test_graphs, prop_ex, nom_file):
    """
    Calculate the average predicate mention metric on test graphs.
    :param test_graphs: the graphs for the test sets
    :param prop_ex: the proposition extraction object
    :return the average predicate mention metric on the test graphs
    """
    pred_graphs = [predict_predicate_mention(test_graph, prop_ex, nom_file) for test_graph in test_graphs]
    return np.mean([compute_predicate_mention_agreement(test_graph, pred_graph)[0]
                    for test_graph, pred_graph in zip(test_graphs, pred_graphs)])


def evaluate_predicate_mention_verbal(test_graphs, prop_ex):
    """
    Calculate the average predicate mention metric on the verbal propositions in test graphs
    :param test_graphs: the graphs for the test sets
    :param prop_ex: the proposition extraction object
    :return the average predicate mention metric on the verbal propositions in test graphs
    """
    verbal_graphs = map(filter_verbal, test_graphs)
    pred_graphs = [predict_predicate_mention(verbal_graph, prop_ex, apply_non_verbal=False)
                   for verbal_graph in verbal_graphs]
    return np.mean([compute_predicate_mention_agreement(test_graph, pred_graph)[0]
                    for test_graph, pred_graph in zip(verbal_graphs, pred_graphs)])


def evaluate_predicate_mention_non_verbal(test_graphs, prop_ex, nom_file):
    """
    Calculate the average predicate mention metric on the non-verbal propositions in test graphs
    :param test_graphs: the graphs for the test sets
    :param prop_ex: the proposition extraction object
    :return the average predicate mention metric on the non-verbal propositions in test graphs
    """
    non_verbal_graphs = map(filter_non_verbal, test_graphs)
    pred_graphs = [predict_predicate_mention(non_verbal_graph, prop_ex, apply_verbal=False, nom_file=nom_file)
                   for non_verbal_graph in non_verbal_graphs]
    return np.mean([compute_predicate_mention_agreement(test_graph, pred_graph)[0]
                    for test_graph, pred_graph in zip(non_verbal_graphs, pred_graphs)])


def predict_predicate_mention(test_graph, prop_ex, nom_file=None, apply_verbal=True, apply_non_verbal=True):
    """
    Given a test graph, and a nominalizations list, returns an identical graph -- apart from the
    predicted proposition mention.
    :param test_graph: the OKR graph
    :param prop_ex: the proposition extraction object
    :param nom_file: the file containing nominalizations
    :param apply_verbal: whether to apply for verbal predicates
    :param apply_non_verbal: whether to apply for non-verbal predicates
    :return the average predicate mention metric for propositions in test graphs
    """

    pred = test_graph.clone()
    proposition_mentions = []

    if nom_file:

        # Load nomlex
        logging.debug('Loading nomlex')
        nom_lexicon = [line.split('\t')[0] for line in open(nom_file)]
        logging.debug('Nomlex[:10] = {}'.format(nom_lexicon[:10]))

    for sent_id, sent in test_graph.sentences.iteritems():

        logging.debug('Analyzing sentence: "{}"'.format(' '.join(sent)))
        indices_and_terms = []

        if apply_verbal:

            # Extract Verbal - make sure that parser's indices agree with the sentence
            indices_and_terms += filter(lambda (indices, terms): all([(ind < len(sent)) and (sent[ind] == term)
                                                                      for (ind, term) in zip(indices, terms.split(" "))]),
                                       prop_ex.get_extractions(' '.join(sent)))
            indices_and_terms += [([ind], word) for (ind, [word, pos]) in enumerate(nltk.pos_tag(sent))
                                  if pos.startswith('V')]
            logging.debug('Verbal: {}'.format(indices_and_terms))

        if apply_non_verbal:

            # Add nominalizations through lookup
            noms = [([ind], word) for (ind, word) in enumerate(sent) if word.lower() in nom_lexicon]
            logging.debug('Nominalizations: {}'.format(noms))
            indices_and_terms += noms

        # Consolidate by removing duplicates (probably not needed)
        indices_and_terms = [(map(int, indices.split('_')), terms) for (indices, terms) in
                             dict([('_'.join(map(str, indices)), terms)
                                   for (indices, terms) in indices_and_terms]).iteritems()]

        proposition_mentions.extend([create_proposition_mention(sent_id, indices, terms)
                                     for (indices, terms) in indices_and_terms])

    pred.propositions = cram_proposition_mentions(proposition_mentions)
    return pred


def create_proposition_mention(sent_id, indices, terms):
    """
    Instansiate only proposition_mention's fields which are required for predicate mention agreement computation
    :param sent_id: the sentence ID
    :param indices: the indices of tokens for the current proposition
    :param terms: the proposition's terms
    :return a new proposition mention
    """
    return PropositionMention(id=NULL_VALUE,
                              sentence_id=sent_id,
                              indices=indices,  # Indices of words in the template
                              terms=terms,      # Terms in template
                              parent=NULL_VALUE,
                              argument_mentions=NULL_VALUE,
                              is_explicit=True)

