"""
Author: Rachel Wities

    Receives two annotated graphs and computes the agreement on the argument coreference (within proposition).
    This script returns the following scores:

    1) MUC (Vilain et al., 1995) - a link-based metric.
    2) B-CUBED (Bagga and Baldwin, 1998) - a mention-based metric.
    3) CEAF (Constrained Entity Aligned F-measure) metric (Luo, 2005) - an entity-based metric.
    4) CoNLL F1/ MELA (Denis and Baldridge, 2009) - an average of these three measures.
"""

from entity_coref import *


def compute_argument_coref_agreement(graph1, graph2, optimal_pred_alignment):
    """
    Receives two annotated graphs and computes the agreement on the entity coreference:
    1) MUC (Vilain et al., 1995) 2) B-CUBED (Bagga and Baldwin, 1998)
    2) B-CUBED (Bagga and Baldwin, 1998) - a mention-based metric.
    3) CEAF (Luo, 2005) 4) MELA (Denis and Baldridge, 2009)
    :param graph1: the first annotator's graph
    :param graph2: the second annotator's graph
    :param optimal_pred_alignment: the optimal alignment of predicates from graph1 to predicates from graph2,
    computed by previous stages.
    :return: MUC, B-CUBED, CEAF and MELA scores. Each score is computed twice, each time
    a different annotator is considered as the gold, and the averaged score is returned.
    """

    # Get argument mentions
    graph1_arg_mentions_dicts = { p_id : [{ arg_id : str(arg) for arg_id, arg in mention.argument_mentions.iteritems() }
                                            for mention in p.mentions.values()]
                                  for p_id, p in graph1.propositions.iteritems() }

    graph2_arg_mentions_dicts = { p_id : [{ arg_id : str(arg) for arg_id, arg in mention.argument_mentions.iteritems() }
                                            for mention in p.mentions.values()]
                                  for p_id, p in graph2.propositions.iteritems() }

    # Clusters of arguments per proposition
    graph1_arg_mentions = { p_id : [set([mention_dict[str(arg_num)]
                                         for mention_dict in mention_lst if str(arg_num) in mention_dict])
                                    for arg_num in range(0, 10)]
                            for p_id, mention_lst in graph1_arg_mentions_dicts.iteritems()}

    graph2_arg_mentions = { p_id : [set([mention_dict[str(arg_num)]
                                         for mention_dict in mention_lst if str(arg_num) in mention_dict])
                                    for arg_num in range(0, 10)]
                            for p_id, mention_lst in graph2_arg_mentions_dicts.iteritems()}

    # Remove empty arguments
    graph1_arg_mentions = {k: [s for s in v if len(s) > 0] for k, v in graph1_arg_mentions.iteritems()}
    graph2_arg_mentions = {k: [s for s in v if len(s) > 0] for k, v in graph2_arg_mentions.iteritems()}

    # within each proposition, compute coreference scores:
    # Compute twice, each time considering a different annotator as the gold, and return the average among each measure
    muc_scores = []
    bcubed_scores = []
    ceaf_scores = []
    mela_scores = []

    for prop1, prop2 in optimal_pred_alignment.iteritems():

        if prop1 not in graph1_arg_mentions:
            continue

        if prop2 not in graph2_arg_mentions:
            continue

        # No arguments
        if len(graph1_arg_mentions[prop1]) == 0 or len(graph2_arg_mentions[prop2]) == 0:
            continue

        muc1, bcubed1, ceaf1 = muc(graph1_arg_mentions[prop1], graph2_arg_mentions[prop2]), \
                               bcubed(graph1_arg_mentions[prop1], graph2_arg_mentions[prop2]), \
                               ceaf(graph1_arg_mentions[prop1], graph2_arg_mentions[prop2])
        mela1 = np.mean([muc1, bcubed1, ceaf1])

        muc2, bcubed2, ceaf2 = muc(graph2_arg_mentions[prop2], graph1_arg_mentions[prop1]), \
                               bcubed(graph2_arg_mentions[prop2], graph1_arg_mentions[prop1]), \
                               ceaf(graph2_arg_mentions[prop2], graph1_arg_mentions[prop1])
        mela2 = np.mean([muc2, bcubed2, ceaf2])

        muc_scores.append(np.mean([muc1, muc2]))
        bcubed_scores.append(np.mean([bcubed1, bcubed2]))
        ceaf_scores.append(np.mean([ceaf1, ceaf2]))
        mela_scores.append(np.mean([mela1, mela2]))

    muc_score = np.mean(muc_scores)
    bcubed_score = np.mean(bcubed_scores)
    ceaf_score = np.mean(ceaf_scores)
    mela_score = np.mean(mela_scores)

    # TODO: Compute the consensual graphs (it wasn't important for the next step)
    consensual_graph1 = graph1
    consensual_graph2 = graph2

    return muc_score, bcubed_score, ceaf_score, mela_score, consensual_graph1, consensual_graph2
