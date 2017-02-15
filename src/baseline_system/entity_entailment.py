import bsddb

from num2words import num2words

"""
A class for determining whether one entity entails the other.

We use three resources:

    1) Dictionary created by training HypeNET on a subset of two datasets: BLESS (Baroni and Lenci, 2011)
    and K&H+N (Necsulescu et al., 2015), using only hypernyms from these datasets (and setting other relations to False).
    Then, as candidates, we used (x, y) term-pairs that occurred together in Wikipedia (thus have connecting dependency
    paths), for which y is one of the 10 most similar words to x in GloVe.
    We classified them using the model and included those that were classified as hypernyms (around 20k term-pairs).

    2) Dictionary with bigrams and trigrams. We extracted the 100k most common bigrams and trigrams in Wikipedia
    (excluding stop words). We used the paragram vectors (Wieting et al., 2015) that encode an ngram as the sum of
    its word embeddings. For each x in the frequent ngrams, we got the 10 most similar words y1,...,y10
    (e.g. for "united states" it will retrieve the words with similar vector to w_v(united)+w_v(states)).
    We filtered out the candidates to use only those that co-occur in Wikipedia. We trained HypeNET on the HypeNET dataset
    using the same paragram vectors to classify term-pairs, representing an ngram as the vector sum of its word embeddings
    (e.g. w_v(united states) = w_v(united) + w_v(states). We filtered out trivial pairs in which y is a word in x
    (the resource contains around 3.5k pairs).

    3) Dictionary with hypernyms from knowledge resources. We used Wikidata, DBPedia and Yago.
    We took (x, y) pairs that were related via one of the hypernymy-indicating relations (manually selected using LinKeR).
    In addition, we filtered out terms that contained weird signs, keeping only meaningful words (contains around 3m
    term-pairs).

Author: Vered Shwartz
"""


class EntityEntailmentBaseline:
    """
    A class for determining whether one entity entails the other.
    """

    def __init__(self, hypenet_unigrams_file, hypenet_ngrams_file, resources_file):

        # Load the resource file
        self.hypenet_unigrams = load_resource_with_score(hypenet_unigrams_file)
        self.hypenet_ngrams = load_resource_with_score(hypenet_ngrams_file)
        self.resource_entailments = load_resource(resources_file)

        # Set thresholds to default
        self.unigram_threshold = 0.5
        self.ngram_threshold = 0.5

    def set_unigram_threshold(self, threshold):
        """
        Set the threshold above which entities are considered entailing.
        :param threshold: the threshold above which entities are considered entailing.
        """
        self.unigram_threshold = threshold

    def set_ngram_threshold(self, threshold):
        """
        Set the threshold above which entities are considered entailing.
        :param threshold: the threshold above which entities are considered entailing.
        """
        self.ngram_threshold = threshold

    def is_entailing(self, entity1, entity2):
        """
        Check whether the first entity entails the second
        :param entity1: the first entity
        :param entity2: the second entity
        """

        # Entailment between the full phrases
        if self.full_entailment(entity1, entity2):
            return True

        ent1_words, ent2_words = entity1.split(' '), entity2.split(' ')

        # Convert numbers to words
        ent1_words = [num2words(int(w)).replace('-', ' ') if w.isdigit() else w for w in ent1_words]
        ent2_words = [num2words(int(w)).replace('-', ' ') if w.isdigit() else w for w in ent2_words]

        # Partial entailment with equivalence, e.g. 'two girls' -> 'two kids':
        if len(ent1_words) > 1 and len(ent2_words) > 1 and \
            (ent1_words[:-1] == ent2_words[:-1] and self.full_entailment(ent1_words[-1], ent2_words[-1]) or \
            ent1_words[1:] == ent2_words[1:] and self.full_entailment(ent1_words[0], ent2_words[0])):
            return True

        # Person name (simplified co-reference)
        if ((self.full_entailment(entity1, 'person') or self.full_entailment(entity1, 'human') and entity2 in entity1) or
            (self.full_entailment(entity2, 'person') or self.full_entailment(entity2, 'human') and entity1 in entity2)):
            return True

        return False

    def full_entailment(self, entity1, entity2):
        """
        Entailment between the full phrases
        :param entity1: the first entity
        :param entity2: the second entity
        :return:
        """
        return (entity1 in self.resource_entailments and entity2 in self.resource_entailments[entity1]) or \
           (entity1 in self.hypenet_unigrams and entity2 in self.hypenet_unigrams[entity1]
            and float(self.hypenet_unigrams[entity1][entity2]) >= self.unigram_threshold) or \
           (entity1 in self.hypenet_ngrams and entity2 in self.hypenet_ngrams[entity1]
            and float(self.hypenet_ngrams[entity1][entity2]) >= self.ngram_threshold)


def load_resource_with_score(resource_file):
    """
    Load a resource file that contains score for each rule
    :param resource_file: the resource file
    :return: a dictionary from word to dictionary of word to score
    """
    db = bsddb.btopen(resource_file, 'r')
    resource = { x : { item.split(':')[0] : float(item.split(':')[1]) for item in ys.split('##') }
                 for x, ys in db.iteritems() }
    return resource


def load_resource(resource_file):
    """
    Load a resource file that doesn't contain score
    :param resource_file: the resource file
    :return: a dictionary from word to set of words
    """
    db = bsddb.btopen(resource_file)
    resource = { x : set(ys.split('##')) for x, ys in db.iteritems() }
    return resource