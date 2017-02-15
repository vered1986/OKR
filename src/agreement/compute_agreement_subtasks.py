"""
compute_agreement_subtasks
Author: Vered Shwartz

    Receives two annotation files about the same story, each annotated by a different annotator,
    and computes the task-level agreement:
    1) Entity mentions
    2) Entity coreference
    3) Predicate mentions
    4) Predicate coreference
    5) Argument mention within predicate chains
    6) Entailment graph
"""
import os
import sys
sys.path.append('../common')

import numpy as np

from okr import *
from docopt import docopt
from entity_coref import compute_entity_coref_agreement
from entity_mention import compute_entity_mention_agreement
from argument_coref import compute_argument_coref_agreement
from predicate_coref import compute_predicate_coref_agreement
from entailment_graph import compute_entailment_graph_agreement
from argument_mention import compute_argument_mention_agreement
from predicate_mention import compute_predicate_mention_agreement, compute_predicate_mention_agreement_verbal, \
    compute_predicate_mention_agreement_non_verbal


def main():
    """
    Receives two annotation directories, containing graph annotations of the same stories (with identical
    file names among the directories), each annotated by a different annotator, and computes the task-level agreement:
    1) Entity mentions
    2) Entity coreference
    3) Predicate mentions
    4) Predicate coreference
    5) Argument mention within predicate chains
    6) Entailment graph
    """
    args = docopt("""Receives two annotation directories, containing graph annotations of the same stories (with identical
    file names among the directories), each annotated by a different annotator, and computes the task-level agreement:
    1) Entity mentions
    2) Entity coreference
    3) Predicate mentions
    4) Predicate coreference
    5) Argument mention within predicate chains
    6) Entailment graph

    Usage:
        compute_agreement_subtasks.py <annotator1_dir> <annotator2_dir>

        <annotator1_dir> = the directory containing the annotations of the first annotator
        <annotator2_dir> = the directory containing the annotations of the second annotator
    """)

    annotator1_dir = args['<annotator1_dir>']
    annotator2_dir = args['<annotator2_dir>']

    annotator1_files = sorted(os.listdir(annotator1_dir))
    annotator2_files = sorted(os.listdir(annotator2_dir))

    results = []
    for annotator1_file, annotator2_file in zip(annotator1_files, annotator2_files):
        print 'Agreement for %s, %s' % (annotator1_dir + '/' + annotator1_file, annotator2_dir + '/' + annotator2_file)
        results.append(compute_agreement(annotator1_dir + '/' + annotator1_file, annotator2_dir + '/' + annotator2_file))

    average = np.mean(results, axis=0)
    ent_score, ent_muc, ent_b_cube, ent_ceaf_c, ent_mela, \
    pred_score, pred_mention_verbal_score, pred_mention_non_verbal_score, pred_muc, pred_b_cube, pred_ceaf_c, pred_mela, \
    arg_mention_score, arg_muc, arg_b_cube, arg_ceaf_c, arg_mela, entities_f1, propositions_f1 = average.tolist()

    print '\n\nAverage:\n=========\n'
    print 'Entity mentions: %.3f' % ent_score
    print 'Entity coreference: MUC=%.3f, B^3=%.3f, CEAF_C=%.3f, MELA=%.3f' % \
          (ent_muc, ent_b_cube, ent_ceaf_c, ent_mela)
    print 'Predicate mentions: %.3f, verbal: %.3f, non-verbal: %.3f' % \
          (pred_score, pred_mention_verbal_score, pred_mention_non_verbal_score)
    print 'Predicate coreference: MUC=%.3f, B^3=%.3f, CEAF_C=%.3f, MELA=%.3f' % \
          (pred_muc, pred_b_cube, pred_ceaf_c, pred_mela)
    print 'Argument mentions: %.3f' % arg_mention_score
    print 'Argument coreference: MUC=%.3f, B^3=%.3f, CEAF_C=%.3f, MELA=%.3f' % \
          (arg_muc, arg_b_cube, arg_ceaf_c, arg_mela)
    print 'Entailment graph F1: entities=%.3f,  propositions=%.3f' % (entities_f1, propositions_f1)


def compute_agreement(annotator1_file, annotator2_file):
    """
    Receives two annotation files about the same story, each annotated by a different annotator,
    and computes the task-level agreement:
    1) Entity mentions
    2) Entity coreference
    3) Predicate mentions
    4) Predicate coreference
    5) Argument mention within predicate chains
    6) Entailment graph
    :param annotator1_file The path for the first graph
    :param annotator2_file The path for the second graph
    """

    # Load the annotation files to OKR objects
    graph1 = load_graph_from_file(annotator1_file)
    graph2 = load_graph_from_file(annotator2_file)

    # Compute agreement for entity mentions and update the graphs to contain only annotations
    # in which both annotators agreed on the entity mentions
    ent_mention_score, consensual_graph1, consensual_graph2 = compute_entity_mention_agreement(graph1, graph2)
    print 'Entity mentions: %.3f' % ent_mention_score

    # Compute agreement for entity coreference and update the graphs to contain only annotations
    # in which both annotators agreed on the entity clusters
    ent_muc, ent_b_cube, ent_ceaf_c, ent_conll_f1, consensual_graph1, consensual_graph2 = \
        compute_entity_coref_agreement(consensual_graph1, consensual_graph2)
    print 'Entity coreference: MUC=%.3f, B^3=%.3f, CEAF_C=%.3f, MELA=%.3f' % (ent_muc, ent_b_cube, ent_ceaf_c, ent_conll_f1)

    # Compute agreement for predicate mentions and update the graphs to contain only annotations
    # in which both annotators agreed on the predicate mentions
    # For analysis purposes, compute also verbal and non-verbal
    pred_mention_non_verbal_score = compute_predicate_mention_agreement_non_verbal(consensual_graph1, consensual_graph2)

    pred_mention_verbal_score = compute_predicate_mention_agreement_verbal(consensual_graph1, consensual_graph2)

    pred_mention_score, consensual_graph1, consensual_graph2 = compute_predicate_mention_agreement(consensual_graph1,
                                                                                      consensual_graph2)

    print 'Predicate mentions: %.3f, verbal: %.3f, non-verbal: %.3f' % (pred_mention_score,
                                                                        pred_mention_verbal_score,
                                                                        pred_mention_non_verbal_score)

    # Compute agreement for predicate coreference and update the graphs to contain only annotations
    # in which both annotators agreed on the predicate clusters
    pred_muc, pred_b_cube, pred_ceaf_c, pred_conll_f1, consensual_graph1, consensual_graph2,optimal_alignment = \
        compute_predicate_coref_agreement(consensual_graph1, consensual_graph2)
    print 'Predicate coreference: MUC=%.3f, B^3=%.3f, CEAF_C=%.3f, MELA=%.3f' % (pred_muc, pred_b_cube, pred_ceaf_c, pred_conll_f1)

    # Compute agreement for argument mention within predicate chains and update the graphs to contain only annotations
    # in which both annotators agreed on the argument mentions
    arg_mention_score, consensual_graph1, consensual_graph2= compute_argument_mention_agreement(consensual_graph1,
                                                                                                consensual_graph2)
    print 'Argument mentions: %.3f' % arg_mention_score
	
    #Compute coreference scores for alignement between arguments of the same propositions:
    arg_muc, arg_b_cube, arg_ceaf_c, arg_conll_f1, consensual_graph1, consensual_graph2 = \
        compute_argument_coref_agreement(consensual_graph1, consensual_graph2,optimal_alignment)
    print 'Argument coreference: MUC=%.3f, B^3=%.3f, CEAF_C=%.3f, MELA=%.3f' % (arg_muc, arg_b_cube, arg_ceaf_c, arg_conll_f1)

    # Compute agreement for the entailment graph and update the graphs to contain only annotations
    # in which both annotators agreed on the edges (propositions, arguments and entities)
    entities_f1, arguments_kappa, propositions_f1, consensual_graph1, consensual_graph2 = \
        compute_entailment_graph_agreement(consensual_graph1, consensual_graph2)
    print 'Entailment graph F1: entities=%.3f, propositions=%.3f' % (entities_f1, propositions_f1)

    return [ent_mention_score, ent_muc, ent_b_cube, ent_ceaf_c, ent_conll_f1,
            pred_mention_score, pred_mention_verbal_score, pred_mention_non_verbal_score,
            pred_muc, pred_b_cube, pred_ceaf_c, pred_conll_f1,
            arg_mention_score, arg_muc, arg_b_cube, arg_ceaf_c, arg_conll_f1,
            entities_f1,  propositions_f1]


if __name__ == '__main__':
    main()
