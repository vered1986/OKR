"""
This file will contain the coreference algorithms for the automatic pipeline system.
Author: shanybar
"""

import clustering_common

#TODO create new algorithm for entity-coreference instead of relying on clustering_common baseline
def cluster_entity_mentions(mention_list):
    """
    currently using clustering_common.
    :param mention_list: the mentions to cluster
    :return: clusters of mentions
    """
    import eval_entity_coref
    return clustering_common.cluster_mentions(mention_list, eval_entity_coref.score)

#TODO create new algorithm for proposition-coreference instead of relying on clustering_common baseline
def cluster_proposition_mentions(mention_list, entities_clustering):
    """
    currently using clustering_common.
    :param mention_list: the mentions to cluster
    :param entities_clustering: a clustering of entity-mentions to coreference chains. 
        format: list of lists of tuples, each tuple stands for an entity-mention - (mention-id, terms)
    :return: clusters of mentions
    """
    import eval_predicate_coref
    return clustering_common.cluster_mentions(mention_list, eval_predicate_coref.score)
