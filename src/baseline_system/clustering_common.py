"""
Utility script for clustering functions
"""


def cluster_mentions(mention_list, score):
    """
    Cluster the predicate mentions in a greedy way: assign each predicate to the first
    cluster with similarity score > 0.5. If no such cluster exists, start a new one.
    :param mention_list: the mentions to cluster
    :param score: the score function that receives a mention and a cluster and returns a score
    :return: clusters of mentions
    """
    clusters = []

    for mention in mention_list:
        found_cluster = False
        for cluster in clusters:
            if score(mention, cluster) > 0.5:
                cluster.add(mention)
                found_cluster = True
                break

        if not found_cluster:
            clusters.append(set([mention]))

    return clusters
