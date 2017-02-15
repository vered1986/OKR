"""
Author: Rachel Wities

    Utility methods for mention agreement.
"""

def str_to_set(str_mention):
    """
    input: mention in str format ("sentence_id[indices_ids]")
    output: set of 1-index strs ("sentence_id[index_id]","sentence_id[index_id]","sentence_id[index_id]")
    """
    sent = str_mention.split('[')[0]
    indices = str_mention[str_mention.index('[') + 1 : str_mention.index(']')].split(',')
    return set([sent + '[' + index + ']' for index in indices])


def overlap_set(str_mention1, set1):
    """
    Receives a mention and a set of mentions and returns whether the mention is in the set
    :param str_mention1: a mention
    :param set1: a set of mentions
    :return: whether the mention is in the set
    """
    return str_to_set(str_mention1).intersection(set1)