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

def cluster_mentions_with_max_cluster(mention_list, score):
    """
    Cluster the predicate mentions in a greedy way: assign each predicate to the
    cluster with the max similarity score, such that similarity score > 0.2. If no such cluster exists, start a new one.
    :param mention_list: the mentions to cluster
    :param score: the score function that receives a mention and a cluster and returns a score
    :return: clusters of mentions
    """
    clusters = []

    for mention in mention_list:
        found_cluster = False
        max_score = 0
        max_cluster = None
        for cluster in clusters:
            cluster_score = score(mention, cluster)
            if cluster_score > 0.2:
                if cluster_score > max_score:
                    max_score = cluster_score
                    max_cluster = cluster
                    found_cluster = True

        if found_cluster:
            max_cluster.add(mention)
        else:
            clusters.append(set([mention]))

    return clusters