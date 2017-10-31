"""
This script is for manual assessment of the automatic pipeline. 
it contains valuable functions to visualize, explore and assess our okr_info output from parse_okr_info. 
available to import and use in interactive mode.
"""
import os, sys
for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

import okr
import pickle


def get_all_prop_mentions_terms_from_okr(okr_graph):
    return list(set.union(*[set(map(lambda pm:terms_of_prop_mention(pm, okr_graph), prop.mentions.values()))
                            for prop in okr_graph.propositions.values()]))

def get_all_prop_mentions_from_okr(okr_graph):
    return [mention for prop in okr_graph.propositions.values() for mention in prop.mentions.values() ]

def terms_of_prop_mention(prop_mention, okr_graph):
    if prop_mention.is_explicit:
        return prop_mention.terms
    else:
        rr = lambda arg: argument_mention_repr(arg, okr_graph)
        return '|'.join([rr(arg) for arg in prop_mention.argument_mentions.values()])

def implicit_percentage(all_proposition_mentions):
    if type(all_proposition_mentions) is dict:
        all_proposition_mentions = all_proposition_mentions.values()
    all_mentions = len(all_proposition_mentions)
    implicits = len([m for m in all_proposition_mentions if m["Head"]["Lemma"] == "IMPLICIT"])
    return float(implicits) / all_mentions


def concept_repr(concept):
    if "terms" in concept:
        return concept["terms"]
    elif 'Bare predicate' in concept:
        return concept['Bare predicate'][0]
    else:
        return "??"


def view_prop_clusters_terms(propositions):
    for p in propositions:
        print [m[1] for m in p]


def view_prop_clusters_full(propositions):
    """ print and return a textual representation of the clustering - each cluster in separte line
    :param propositions: clustering - list of lists 
    :return: list of lists of textual representation of each prop mention
    """
    props = []
    for p in propositions:
        full_prop_cluster = []
        for m in p:
            s = m[2]["Template"]
            arg_map = {arg_id :concept_repr(arg) for arg_id, arg in m[2]["Arguments"].iteritems()}
            full_prop=s.format(**arg_map)
            full_prop_cluster.append(full_prop)
        props.append(full_prop_cluster)
        print ";\t".join(full_prop_cluster)
    return props


# on okr_v1 object
def argument_mention_repr(argumentMention, okr_v1):
    if argumentMention.mention_type == 0:   # arg is entity
        parent_mention = okr_v1.entities[argumentMention.parent_id].mentions[argumentMention.parent_mention_id]
    else:
        parent_mention = okr_v1.propositions[argumentMention.parent_id].mentions[argumentMention.parent_mention_id]
    return parent_mention.terms


def view_okr_prop_clusters(okr_v1):
    props = []
    for pid,p in okr_v1.propositions.iteritems():
        full_prop_cluster = []
        for mid, m in p.mentions.iteritems():
            s = m.template
            arg_map = {arg_id :argument_mention_repr(arg, okr_v1) for arg_id, arg in m.argument_mentions.iteritems()}
            full_prop=s.format(**arg_map)
            full_prop_cluster.append(full_prop)
        props.append(full_prop_cluster)
        print ";\t".join(full_prop_cluster)
    return props

def present_prop_extraction(pred, gold):    # okr_v1 objects
    gold_mentions = get_all_prop_mentions_from_okr(gold)
    pred_mentions = get_all_prop_mentions_from_okr(pred)
    sentences = set([m.sentence_id for m in gold_mentions])
    for sent in sentences:
        gold_sent_prop_terms = [terms_of_prop_mention(m, gold) for m in gold_mentions if m.sentence_id == sent]
        pred_sent_prop_terms = [terms_of_prop_mention(m, pred) for m in pred_mentions if m.sentence_id == str(sent)]
        print sent, gold_sent_prop_terms, pred_sent_prop_terms

def present_gold(gold, sentences):
    gold_mentions = get_all_prop_mentions_from_okr(gold)
    for sent in sorted(sentences.keys()):
        mm = [m for m in gold_mentions if str(m.sentence_id)==str(sent)]
        for m in mm:
            print sent, sentences[sent], "\t#", m.template, "|", terms_of_prop_mention(m, gold), str(m)

