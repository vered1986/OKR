"""
compute_baseline_subtasks
Author: Vered Shwartz

    Receives the validation set and the test set, runs the baseline systems,
    and computes the task-level evaluation metrics:
    1) Entity mentions
    2) Entity coreference
    3) Predicate mentions
    4) Predicate coreference
    5) Argument mention within predicate chains
    6) Entailment graph
"""

import sys

sys.path.append('../common')

from okr import *
from docopt import docopt
from eval_predicate_mention import *
from prop_extraction import prop_extraction
from eval_entity_coref import evaluate_entity_coref
from eval_entailment_graph import evaluate_entailment
from eval_argument_coref import evaluate_argument_coref
from eval_entity_mention import evaluate_entity_mention
from eval_predicate_coref import evaluate_predicate_coref
from eval_argument_mention import evaluate_argument_mention


def main():
    """
    Receives the validation set and the test set, runs the baseline systems,
    and computes the task-level evaluation metrics:
    1) Entity mentions
    2) Entity coreference
    3) Predicate mentions
    4) Predicate coreference
    5) Argument mention within predicate chains
    6) Entailment graph
    """
    args = docopt("""Receives the validation set and the test set, runs the baseline systems,
    and computes the task-level evaluation metrics:
    1) Entity mentions
    2) Entity coreference
    3) Predicate mentions
    4) Predicate coreference
    5) Argument mention within predicate chains
    6) Entailment graph

    Usage:
        compute_baseline_subtasks.py <val_set_folder> <test_set_folder>

        <val_set_folder> = the validation set file
        <test_set_folder> = the test set file
    """)

    val_folder = args['<val_set_folder>']
    test_folder = args['<test_set_folder>']

    # Load the annotation files to OKR objects
    val_graphs = load_graphs_from_folder(val_folder)
    test_graphs = load_graphs_from_folder(test_folder)

    # Run the entity mentions component and evaluate them
    ent_score = evaluate_entity_mention(test_graphs)
    print 'Entity mentions: %.3f' % ent_score

    # Run the entity coreference component and evaluate them
    ent_muc, ent_b_cube, ent_ceaf_c, ent_mela = evaluate_entity_coref(test_graphs)
    print 'Entity coreference: MUC=%.3f, B^3=%.3f, CEAF_C=%.3f, MELA=%.3f' % (ent_muc, ent_b_cube, ent_ceaf_c, ent_mela)

    # Load a common proposition extraction model
    logging.debug('Loading proposition extraction module')
    prop_ex = prop_extraction()

    # Run the predicate mentions component and evaluate them
    pred_score = evaluate_predicate_mention(test_graphs, prop_ex, './nominalizations/nominalizations.reuters.txt')
    print 'Predicate mentions(full): %.3f' % pred_score

    # Split the predicate mentions into verbal and non-verbal and evaluate performance
    pred_verbal_score = evaluate_predicate_mention_verbal(test_graphs, prop_ex)
    print 'Predicate mentions(verbal): %.3f' % pred_verbal_score

    pred_non_verbal_score = evaluate_predicate_mention_non_verbal(test_graphs, prop_ex,
                                                                  './nominalizations/nominalizations.reuters.txt')
    print 'Predicate mentions(non-verbal): %.3f' % pred_non_verbal_score

    # Run the entity predicate component and evaluate them
    macro_score = evaluate_predicate_coref(test_graphs)
    pred_muc, pred_b_cube, pred_ceaf_c, pred_mela = macro_score
    print 'Predicate coreference: MUC=%.3f, B^3=%.3f, CEAF_C=%.3f, MELA=%.3f' % \
          (pred_muc, pred_b_cube, pred_ceaf_c, pred_mela)

    # Run the argument mentions component and evaluate them.
    score = evaluate_argument_mention(test_graphs, 1)
    print 'Argument mentions: %.3f' % score

    # Compute coreference scores for alignment between arguments of the same propositions
    muc, b_cube, ceaf_c, mela = evaluate_argument_coref(test_graphs)
    print 'Argument coreference: MUC=%.3f, B^3=%.3f, CEAF_C=%.3f, MELA=%.3f' % (muc, b_cube, ceaf_c, mela)

    # Run the predicate, entities and arguments entailment components and evaluate them
    entities_f1, propositions_f1 = evaluate_entailment(val_graphs, test_graphs)
    print 'Entailment graph F1: entities=%.3f, propositions=%.3f' % (entities_f1, propositions_f1)


if __name__ == '__main__':
    main()
