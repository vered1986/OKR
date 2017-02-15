"""
Functions for filtering propositions for analysis purposes -- used both in agreement and baseline
computations.
Author: Gabi Stanovsky
"""
import nltk
import logging

from okr import Proposition
from constants import NULL_VALUE


# Return a graph with only the verbal proposition in the input graph
filter_verbal = lambda(test_graph): \
                filter_proposition_mentions(verbal_filter,
                                            test_graph)

# Return a graph with only the non-verbal propositions in the input graph
filter_non_verbal = lambda(test_graph): \
                    filter_proposition_mentions(lambda sentence, mention: not(verbal_filter(sentence, mention)),
                                                test_graph)

verbal_filter = lambda sentence, mention: (len(mention.indices) == 1) and \
                                            (nltk.pos_tag(sentence)[mention.indices[0]][1].startswith('V'))


def filter_proposition_mentions(filter_func, test_graph):
    """
    Filter proposition mentions in test graph iff filter_func holds
    :param filter_func: the filtering function (from OKR graph to OKR graph)
    :param test_graph: the OKR graph
    """
    ret = test_graph.clone()
    proposition_mentions = []
    logging.debug('Filtering verbal propositions')
    for prop in test_graph.propositions.values():
        for mention in prop.mentions.values():
            sent = test_graph.sentences[mention.sentence_id]
            if filter_func(sent, mention):
                logging.debug('Found {}'.format(mention.terms))
                proposition_mentions.append(mention)

    logging.debug('#Propositions after filter = {}'.format(len(proposition_mentions)))
    ret.propositions = cram_proposition_mentions(proposition_mentions)
    return ret


def cram_proposition_mentions(proposition_mentions):
    """
    "Cram" all of the predicted proposition mentions under the same proposition
    Since it doesn't matter for agreement computation
    :param proposition_mentions the proposition mentions
    """
    return {0 : Proposition(id = NULL_VALUE,
                            name = NULL_VALUE,
                            mentions = dict(zip(range(len(proposition_mentions)),
                                                proposition_mentions)),
                            attributor = NULL_VALUE,
                            terms = NULL_VALUE,
                            entailment_graph = NULL_VALUE)}
