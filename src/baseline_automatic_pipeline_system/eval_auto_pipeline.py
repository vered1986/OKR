"""
Author: Shany Barhom

Evaluation for automatic pipeline .
Run it from base OKR directory.

Usage: eval_auto_pipeline --input=INPUT_FOLDER --gold=GOLD_STANDARD_FOLDER

INPUT_FOLDER: the folder that contains the input files (text files)
GOLD_STANDARD_FOLDER: the folder that contains the corresponding gold standard files (XML files)

Usage example: eval_auto_pipeline --input=data/baseline/test_input --gold=data/baseline/test

NOTE - the name of an input file should be the same name as the
corresponding gold file (or at least start with the same name as the gold file)

"""

import os
import sys
import numpy as np
import copy
from docopt import docopt

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

from okr import *
from parse_okr_info import get_raw_sentences_from_file,auto_pipeline_okr_info
from eval_entity_coref import eval_entity_coref_between_two_graphs
from eval_predicate_coref import eval_predicate_coref_between_two_graphs
from eval_predicate_mention import evaluate_predicate_mention_between_two_graphs


def eval_auto_pipeline(input_folder,gold_folder):
    """
    Evaluate entity coreference, predicate coreference, entity and predicate extraction
    for auto-created graphs and their gold graphs
    :param input_folder: Path to the input files folder (text files)
    :param gold_folder: Path to the gold graphs folder (xml files)
    :return:
    """

    input_files = [f for f in os.listdir(input_folder)]
    gold_files = [f for f in os.listdir(gold_folder)]

    # TODO: Add automatic matching between input files and gold files

    sorted_input = sorted(input_files)
    sorted_gold = sorted(gold_files)

    predicate_coref_scores = []
    entity_coref_scores = []
    predicate_mention_scores = []
    entity_mention_scores = []


    for input_file,gold_file in zip(sorted_input,sorted_gold):
        print 'Evaluate input file '+ input_file + ' with gold file ' + gold_file
        gold_graph = load_graph_from_file(os.path.join(gold_folder, gold_file))
        # for prediction, take only sentences occurring in the gold (in propositions or entities
        annotated_sentences = set([str(m.sentence_id)
                                   for prop in gold_graph.propositions.values()
                                   for m in prop.mentions.values()] +
                                  [str(m.sentence_id)
                                   for entity in gold_graph.entities.values()
                                   for m in entity.mentions.values()])

        predicted_sentences = get_raw_sentences_from_file(os.path.join(input_folder, input_file))
        sentences = {sentence_id : sentence
                     for sentence_id, sentence in predicted_sentences.iteritems()
                     if sentence_id in annotated_sentences}
        okr_info = auto_pipeline_okr_info(sentences)
        predicted_graph = OKR(**copy.deepcopy(okr_info))

        # Evaluate
        pred_mention_score = evaluate_predicate_mention_between_two_graphs(predicted_graph,gold_graph)
        ent_mention_score = evaluate_entity_mention_between_two_graphs(predicted_graph,gold_graph)
        curr_predicate_scores, _ = eval_predicate_coref_between_two_graphs(predicted_graph, gold_graph)
        curr_entity_scores = eval_entity_coref_between_two_graphs(predicted_graph, gold_graph)

        predicate_mention_scores.append(pred_mention_score)
        entity_mention_scores.append(ent_mention_score)
        predicate_coref_scores.append(curr_predicate_scores)
        entity_coref_scores.append(curr_entity_scores)

    predicate_scores = np.mean(predicate_coref_scores, axis=0).tolist()
    entity_scores = np.mean(entity_coref_scores, axis=0).tolist()
    predicate_mention_score = np.mean(predicate_mention_scores, axis=0).tolist()
    entity_mention_score = np.mean(entity_mention_scores, axis=0).tolist()

    print 'Predicate mentions(F1) = %.3f Recall = %.3f Precision = %.3f  ' % (predicate_mention_score[0],predicate_mention_score[1],predicate_mention_score[2])

    print 'Entity mentions(F1): %.3f Recall = %.3f Precision = %.3f' % tuple(entity_mention_score[:3])

    pred_muc, pred_b_cube, pred_ceaf_c, pred_mela = predicate_scores
    print 'Predicate coreference: MUC=%.3f, B^3=%.3f, CEAF_C=%.3f, MELA=%.3f' % \
          (pred_muc, pred_b_cube, pred_ceaf_c, pred_mela)

    entity_muc, entity_b_cube, entity_ceaf_c, entity_mela = entity_scores
    print 'Entity coreference: MUC=%.3f, B^3=%.3f, CEAF_C=%.3f, MELA=%.3f' % \
          (entity_muc, entity_b_cube, entity_ceaf_c, entity_mela)



def evaluate_entity_mention_between_two_graphs(pred_graph,gold_graph):
    """
    Receives a predicted graph (after the automatic pipeline) and a gold graph and
    compute the f1,recall and precision scores for the entity mentions extraction
    :param predicted_graph: Auto-created graph
    :param gold_graph: the gold standard graph
    :return: numpy array of the f1,recall and precision scores
    """
    graph2_ent_mentions = set.union(*[set(map(str, entity.mentions.values()))
                                      for entity in pred_graph.entities.values()])
    graph1_ent_mentions = set.union(*[set(map(str, entity.mentions.values()))
                                      for entity in gold_graph.entities.values()])
    consensual_mentions = graph1_ent_mentions.intersection(graph2_ent_mentions)

    recall = len(consensual_mentions) * 1.0 / len(graph1_ent_mentions)
    precision = len(consensual_mentions) * 1.0 / len(graph2_ent_mentions)

    f1 = 2.00 * (recall * precision) / (recall + precision) if (recall + precision) > 0.0 else 0.0
    return np.array([f1,recall,precision])



if __name__ == "__main__":

    # Parse arguments
    args = docopt(__doc__)
    input_folder = args["--input"]
    gold_folder = args["--gold"]

    print 'Running Evaluation...'
    eval_auto_pipeline(input_folder=input_folder, gold_folder=gold_folder)