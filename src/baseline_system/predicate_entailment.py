import re
import bsddb

from spacy.en import English


"""
A class for determining whether one predicate template entails the other.

    We use a state-of-the-art resource of entailment rules between natural language predicates, containing millions of
    entailment rules, described in the paper:
    Efficient Tree-based Approximation for Entailment Graph Learning. Jonathan Berant, Ido Dagan, Meni Adler, and
    Jacob Goldberger. ACL 2012.
    We tune a threshold on the validation set to separate between entailing and non-entailing rules.

Author: Vered Shwartz
"""


class PredicateEntailmentBaseline:
    """
    A class for determining whether one predicate template entails the other.
    Using a state-of-the-art resource of entailment rules between natural language predicates, containing millions of
    entailment rules, described in the paper:
    Efficient Tree-based Approximation for Entailment Graph Learning. Jonathan Berant, Ido Dagan, Meni Adler, and
    Jacob Goldberger. ACL 2012.
    We tune a threshold on the validation set to separate between entailing and non-entailing rules.
    """

    def __init__(self, resource_file):

        # Load the resource file
        self.entailment_rules = bsddb.btopen(resource_file, 'r')

        # Set threshold to default as recommended
        self.threshold = 0.0

        self.nlp = English()

    def set_threshold(self, threshold):
        """
        Set the threshold above which predicates are considered entailing.
        :param threshold: the threshold above which predicates are considered entailing.
        """
        self.threshold = threshold

    def is_entailing(self, pred1, pred2):
        """
        Check whether the first predicate entails the second predicate
        :param pred1: the first predicate
        :param pred2: the second predicate
        """

        # Find the first two arguments from each predicate, and the matching arguments between predicates
        args1 = re.findall(r'(\[a[0-9]+\])', pred1)
        args2 = re.findall(r'(\[a[0-9]+\])', pred2)
        shared_args = [a for a in args1 if a in args2] # intersection that keeps order

        # Support only binary predicated
        if len(shared_args) < 2:
            return False

        pred1_rule, pred2_rule = pred1.replace(shared_args[0], '@X@').replace(shared_args[1], '@Y@'), \
                                 pred2.replace(shared_args[0], '@X@').replace(shared_args[1], '@Y@')

        # Remove preceding arguments
        start, end = min(pred1_rule.index('@X@'), pred1_rule.index('@Y@')), \
                     max(pred1_rule.index('@X@'), pred1_rule.index('@Y@'))
        pred1_rule = pred1_rule[start:end + 3]

        start, end = min(pred2_rule.index('@X@'), pred2_rule.index('@Y@')), \
                     max(pred2_rule.index('@X@'), pred2_rule.index('@Y@'))
        pred2_rule = pred2_rule[start:end + 3]

        # Lemmatize the predicate templates
        pred1_rule = str(' '.join([token.lemma_.lower().strip() for token in self.nlp(unicode(pred1_rule))]))
        pred2_rule = str(' '.join([token.lemma_.lower().strip() for token in self.nlp(unicode(pred2_rule))]))

        rule = '###'.join((pred1_rule, pred2_rule))
        rule = rule.replace('@x@', 'X').replace('@y@', 'Y')

        # If there is still another argument in the middle, discard the predicates
        if re.findall(r'(\[a[0-9]+\])', pred1_rule) or re.findall(r'(\[a[0-9]+\])', pred2_rule):
            return False

        other_rule = rule.replace('@X@', '@Z@').replace('@Y@', 'X').replace('@Z@', 'Y')

        return (rule in self.entailment_rules and float(self.entailment_rules[rule]) >= self.threshold) or \
               (other_rule in self.entailment_rules and float(self.entailment_rules[other_rule]) >= self.threshold)
