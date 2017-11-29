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
    """ Return a unified list of all prop-mentions terms in the graph. """
    return list(set.union(*[set(map(lambda pm:terms_of_prop_mention(pm, okr_graph), prop.mentions.values()))
                            for prop in okr_graph.propositions.values()]))

def get_all_prop_mentions_from_okr(okr_graph):
    """ Return all PropositionMentions in the graph. """
    return sorted([mention for prop in okr_graph.propositions.values() for mention in prop.mentions.values() ],
                  key=lambda m:int(m.sentence_id))

def get_all_entity_mentions_from_okr(okr_graph):
    """ Return all EntityMentions in the graph. """
    return sorted([mention for ent in okr_graph.entities.values() for mention in ent.mentions.values() ],
                  key=lambda m:int(m.sentence_id))

def terms_of_prop_mention(prop_mention, okr_graph):
    """ Return a representation of terms of prop-mention - template for explicit, concatenated args for implicit. """
    if prop_mention.is_explicit:
        return prop_mention.template
    else:
        rr = lambda arg: argument_mention_repr(arg, okr_graph)
        return '|'.join([rr(arg) for arg in prop_mention.argument_mentions.values()])

def implicit_percentage(okr_v1):
    """ return #num-of-implicit-prop-mention / #num-of-prop-mentions. """
    all_proposition_mentions = get_all_prop_mentions_from_okr(okr_v1)
    all_mentions = len(all_proposition_mentions)
    implicits = len([m for m in all_proposition_mentions if not m.is_explicit])
    return float(implicits) / all_mentions


def view_prop_clusters_full(propositions):
    """ print and return a textual representation of the clustering - each cluster in separte line
    :param propositions: clustering - list of lists 
    :return: list of lists of textual representation of each prop mention
    """

    def concept_repr(concept):
        if "terms" in concept:
            return concept["terms"]
        elif 'Bare predicate' in concept:
            return concept['Bare predicate'][0]
        else:
            return "??"

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
    """ Return terms (str) of argument mention (retrieve from parent EntityMention or PropositionMention) """
    if argumentMention.mention_type == 0:   # arg is entity
        parent_mention = okr_v1.entities[argumentMention.parent_id].mentions[argumentMention.parent_mention_id]
    else:
        parent_mention = okr_v1.propositions[argumentMention.parent_id].mentions[argumentMention.parent_mention_id]
    return parent_mention.terms


def view_okr_prop_clusters(okr_v1):
    """ print and return a textual representation of the clustering, based on OKR object """
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

def view_and_compare_prop_extraction(pred, gold):    # okr_v1 objects
    """ print a comparison of predicted and gold proposition-mentions extraction"""
    gold_mentions = get_all_prop_mentions_from_okr(gold)
    pred_mentions = get_all_prop_mentions_from_okr(pred)
    sentences = set([m.sentence_id for m in gold_mentions])
    for sent in sentences:
        gold_sent_prop_terms = [terms_of_prop_mention(m, gold) for m in gold_mentions if m.sentence_id == sent]
        pred_sent_prop_terms = [terms_of_prop_mention(m, pred) for m in pred_mentions if m.sentence_id == str(sent)]
        print sent, gold_sent_prop_terms, pred_sent_prop_terms


def view_implicits(okr_v1, sentences):
    """ print all implicit proposition mentions in graph. """
    for m in get_all_prop_mentions_from_okr(okr_v1):
        if m.is_explicit or str(m.sentence_id) not in sentences:
            continue
        sent_id=str(m.sentence_id)
        print sent_id, sentences[sent_id], "#\n  Terms:",terms_of_prop_mention(m, okr_v1)

def view_explicits(okr_v1, sentences):
    """ print all explicit proposition mentions in graph. """
    for m in get_all_prop_mentions_from_okr(okr_v1):
        if not m.is_explicit or str(m.sentence_id) not in sentences:
            continue
        sent_id=str(m.sentence_id)
        print sent_id, sentences[sent_id], "#\n  Terms:",terms_of_prop_mention(m, okr_v1), "  Template: ", m.template

def get_all_props_of_sentence(graph, sent_id):
    """ return al proposition-mentions of a certain sentence. """
    return [m for m in get_all_prop_mentions_from_okr(graph) if m.sentence_id==sent_id ]

def get_all_entities_of_sentence(graph, sent_id):
    """ return al entity-mentions of a certain sentence. """
    return [m for m in get_all_entity_mentions_from_okr(graph) if m.sentence_id==sent_id ]

def view_nesting_props(graph, sentences):
    """ Print all nesting proposition mentions - prop-mentions that one of their argument is a proposition"""
    mentions = get_all_prop_mentions_from_okr(graph)
    for m in mentions:
        prop_args = [a for a in m.argument_mentions.values() if a.mention_type==1]
        orig_arg_mentions = [graph.propositions[a.parent_id].mentions[a.parent_mention_id] for a in prop_args]
        if prop_args:
            for arg in orig_arg_mentions:
                print m.sentence_id, sentences[str(m.sentence_id)]
                print terms_of_prop_mention(m, graph) + " ; " + terms_of_prop_mention(arg, graph)
                print "\n"

def find_in_sentences(graph, string_to_find):
    """ return a list with IDs of all sentence in which string_to_find occur """
    return [id for id, sent in graph.sentences.iteritems() if string_to_find in ' '.join(sent)]

def single_sentence_extraction(gold_graph, sentent_id):
    """ return a representation of Entity and Proposition extraction of a sentence from a gold graph """
    prop_mentions = get_all_props_of_sentence(gold_graph, sentent_id)
    entity_mentions = get_all_entities_of_sentence(gold_graph, sentent_id)
    return {"Propositions" : [terms_of_prop_mention(m, gold_graph) for m in prop_mentions],
            "Entities" : [m.terms for m in entity_mentions],
            "Sentence" : ' '.join(gold_graph.sentences[sentent_id])}
