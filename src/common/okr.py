"""
A class for the OKR graph structure
Author: Rachel Wities and Vered Shwartz
"""
import os
import copy
import logging
import itertools
import xml.etree.ElementTree as ET

from constants import *


class OKR:
    """
    A class for the OKR graph structure
    """
    def __init__(self, name, sentences, ignored_indices, tweet_ids, entities, propositions):

        self.name = name  # XML file name
        self.sentences = sentences  # Dictionary of sentence ID (starts from 1) to tokenized sentence
        self.ignored_indices = ignored_indices  # set of words to ignore, in format sentence_id[index_id]
        self.tweet_ids = tweet_ids  # Dictionary of sentence ID to tweet ID
        self.entities = entities  # Dictionary of entity ID to Entity object
        self.propositions = propositions  # Dictionary of proposition id to Proposition object

        # Set arguments original indices and name
        for prop in self.propositions.values():
            for prop_mention in prop.mentions.values():
                for argument in prop_mention.argument_mentions.values():
                    set_parent_indices(argument, self)

        # Set template for predicate mentions and use it to create mention entailment graph
        for p_id, prop in self.propositions.iteritems():
            for m_id, prop_mention in prop.mentions.iteritems():
                set_template(prop_mention, self.entities, self.propositions)

            prop.entailment_graph.mentions_graph = from_term_id_to_mention_id(prop.entailment_graph.graph,
                                                                              prop.mentions, MentionType.Proposition)
            prop.entailment_graph.contradictions_mention_graph = from_term_id_to_mention_id(
                prop.entailment_graph.contradictions_graph, prop.mentions, MentionType.Proposition)

        # Create dictionaries to get mentions by their string ID
        self.prop_mentions_by_key = {str(mention): mention
                                     for prop in self.propositions.values() for mention in prop.mentions.values()}

        self.ent_mentions_by_key = {str(mention): mention
                                    for ent in self.entities.values() for mention in ent.mentions.values()}

    def clone(self):
        """
        Returns a deep copy of the graph
        """
        return copy.deepcopy(self)

    def get_sentence_by_id(self, sent_id_str):
        """
        Receives a sentence ID and returns a sentence
        :param sent_id_str The sentence ID
        :return the sentence
        """
        sent_id = int(sent_id_str.split('[')[0])
        sentence = self.sentences[sent_id]
        indices = [int(index) for index in sent_id_str.split('[')[1][:-1].split(',')]
        indices_new = [int(index) for index in sent_id_str.split('[')[1][:-1].split(',') if int(index) < len(sentence)]
        if not len(indices) == len(indices_new):
            logging.warning('Error in the length of sentence id %s' % sent_id_str)

        return ' '.join([sentence[i] for i in indices_new])


class AbstractNode:
    """
    A class for either a proposition or an entity in the graph
    """

    def __init__(self, id, name, mentions, terms, entailment_graph):
        self.id = id
        self.name = name
        self.mentions = mentions
        self.terms = terms
        self.entailment_graph = entailment_graph

    def __str__(self):
        """
        Use this as a unique id for a node which is comparable among graphs
        """
        return '#'.join(sorted(list(set(map(str, self.mentions.values())))))


class Entity(AbstractNode):
    """
    A class for an entity in the graph
    """

    def __init__(self, id, name, mentions, terms, entailment_graph):
        AbstractNode.__init__(self, id, name, mentions, terms, entailment_graph)


class Proposition(AbstractNode):
    """
    A class for a proposition in the graph
    """

    def __init__(self, id, name, mentions, attributor, terms, entailment_graph):
        AbstractNode.__init__(self, id, name, mentions, terms, entailment_graph)
        self.attributor = attributor


class Mention:
    """
    An abstract class for a mention in the graph
    """

    def __init__(self, id, sentence_id, indices, terms, parent):
        self.id = id
        self.sentence_id = sentence_id
        self.indices = indices
        self.terms = terms
        self.parent = parent

    def __str__(self):
        """
        Use this as a unique id for a mention which is comparable among graphs
        """
        return str(self.sentence_id) + str(self.indices)


class Entailment_graph:
    """
    A class representing the entailment graph (for propositions, entities or arguments)
    """

    def __init__(self, graph, mentions_graph, contradictions_graph, contradictions_mention_graph):
        self.graph = graph  # graph of terms
        self.mentions_graph = mentions_graph  # graph of mention IDs (each term is connected to one or more mention IDs)
        self.contradictions_graph = contradictions_graph  # graph of contradictions (terms)
        self.contradictions_mention_graph = contradictions_mention_graph  # graph of contradictions (mention IDs)


class EntityMention(Mention):
    """
    A class for an entity mention in the graph
    """

    def __init__(self, id, sentence_id, indices, terms, parent):
        Mention.__init__(self, id, sentence_id, indices, terms, parent)


class PropositionMention(Mention):
    """
    A class for a proposition mention in the graph
    """

    def __init__(self, id, sentence_id, indices, terms, parent, argument_mentions, is_explicit):
        Mention.__init__(self, id, sentence_id, indices, terms, parent)
        self.argument_mentions = argument_mentions
        self.template = None  # template with argument IDs
        self.is_explicit = is_explicit

    def __str__(self):
        """
        Use this as a unique id for a mention which is comparable among graphs
        override inherited function in order to implement str for implicit mentions and remove prepositions
        """

        # Implicit proposition
        if self.indices == [-1]:
            new_indices = [item for sublist in [arg.parent_indices[1] for arg in self.argument_mentions.values()] for
                           item in sublist]
            new_indices.sort()
            return str(self.sentence_id) + str(new_indices)

        # TODO: Rachel - replace with POS looking for nouns and verbs
        terms_lst = self.terms.split()
        verb_noun_indices = [self.indices[i] for i in range(0, len(self.indices) - 1) if terms_lst[i] not in STOP_WORDS]

        # Predicate with noun or a verb
        if len(verb_noun_indices) > 0:
            return str(self.sentence_id) + str(verb_noun_indices)

        return str(self.sentence_id) + str(self.indices)


class ArgumentMention:
    """
    A class for an argument mention in the graph
    """

    def __init__(self, id, desc, mention_type, parent_id, parent_mention_id):
        self.id = id
        self.desc = desc
        self.mention_type = mention_type
        self.parent_id = parent_id
        self.parent_mention_id = parent_mention_id

        # These fields are set when the graph loading is done
        self.parent_indices = None
        self.parent_name = None

    def __str__(self):
        """
        Use this as a unique id for a mention which is comparable among graphs
        Returns unique ID for argument mention only
        """
        if self.parent_indices == None:
            return 'NONE'
        return str(self.parent_indices[0]) + str(self.parent_indices[1])

    def str_p(self, proposition_mention):
        """
        Use this as a unique id for a mention which is comparable among graphs
        Returns unique ID of proposition_mention + argument_mention
        """
        return str(proposition_mention) + '_' + str(self)


def load_graphs_from_folder(input_folder):
    """
    Load OKR files from a given folder
    :param input_folder: the folder path
    :return: a list of OKR objects
    """
    return [load_graph_from_file(input_folder + "/" + f) for f in os.listdir(input_folder)]


def load_graph_from_file(input_file):
    """
    Loads an OKR object from an xml file
    :param input_file: the xml file
    :return: an OKR object
    """
    mention_types = {'Entity': MentionType.Entity, 'Proposition': MentionType.Proposition}

    # Load the xml to a tree object
    tree = ET.parse(input_file)

    # Load the sentences
    root = tree.getroot()
    sentences_node = root.find('sentences')[1:]

    # Handle different versions - old version:
    if sentences_node[0].find('str') != None:
        sentences = { int(sentence.find('id').text): sentence.find('str').text.split() for sentence in sentences_node }
        ignored_indices = None
        tweet_ids = {}

    # New version
    else:
        sentences = { int(sentence.find('id').text) : [token.find('str').text for token in sentence.find('tokens')]
                     for sentence in sentences_node }
        ignored_indices = set(
            [sentence.find('id').text + '[' + token.find('id').text + ']' for sentence in sentences_node
             for token in sentence.find('tokens') if token.find('isIrrelevant').text == 'true'])
        tweet_ids = {int(sentence.find('id').text): sentence.find('name').text for sentence in sentences_node}

    # Load the entities
    entities_node = root.find('typeManagers').findall('typeManager')[1].find('types')
    entities = {}
    for entity in entities_node:

        # Entity mentions
        mentions = {int(mention[0].text):  # mention id
                        EntityMention(int(mention[0].text),  # mention id
                                      int(mention[1].text),  # sentence id
                                      [int(index[0].text) for index in mention[3]],  # mention indices
                                      ' '.join([index[1].text.lower() for index in mention[3]]),  # mention terms
                                      int(entity[0].text)  # parent
                                      ) for mention in entity.find('mentions')}
        # Check for empty mentions
        empty_mentions = [(mention.parent, m_id) for m_id, mention in mentions.iteritems() if len(mention.indices) == 0]
        if len(empty_mentions) > 0:
            logging.warning('Empty mentions in entity %s' % entity[0].text)

        # Entity entailment graph
        entailment_info = entity[3]
        ent_terms = entailment_info[0]
        term_dic = {int(term[0].text): term[1].text.lower() for term in ent_terms}
        graph = []
        contradictions_graph = []
        ent_connections = entailment_info[1]
        for connection in ent_connections:

            # The second entails the first or they are equal
            if connection[0].text == '1' or connection[0].text == '0':
                graph.append((term_dic[int(connection[2].text)], term_dic[int(connection[1].text)]))

            # The first entails the second or they are equal
            if connection[0].text == '2' or connection[0].text == '0':
                graph.append((term_dic[int(connection[1].text)], term_dic[int(connection[2].text)]))

            # Contradiction
            if connection[0].text == '3':
                contradictions_graph.append((term_dic[int(connection[1].text)], term_dic[int(connection[2].text)]))

        # Create the transitive closure of the entailment graph
        final_graph = transitive_closure(graph)
        mentions_graph = from_term_id_to_mention_id(final_graph, mentions, MentionType.Entity)
        contradictions_mentions_graph = from_term_id_to_mention_id(contradictions_graph, mentions, MentionType.Entity)

        entity_entailment = Entailment_graph(final_graph, mentions_graph, contradictions_graph,
                                             contradictions_mentions_graph)

        # Entity terms
        terms = set([mention.terms for mention in mentions.values()])

        entities[int(entity[0].text)] = Entity(int(entity[0].text),  # id
                                               entity[1].text,  # name
                                               mentions,  # entity mentions
                                               terms,  # entity terms
                                               entity_entailment)  # entity entailment graph

    # Load the propositions
    propositions_node = root.find('typeManagers').findall('typeManager')[0].find('types')
    propositions = {}

    for proposition in propositions_node:

        # Proposition mentions
        mentions = {int(mention.find('id').text):  # mention id
                        PropositionMention(int(mention.find('id').text),  # mention id
                                           int(mention.find('sentenceId').text),  # sentence id
                                           [int(index.find('ind').text) for index in mention.find('tokens')], # mention indices

                                           # mention terms
                                           ' '.join([index.find('word').text.lower() for index in mention.find('tokens')]),

                                           int(proposition[0].text),  # parent

                                           # Argument mentions
                                           {arg[0].text: ArgumentMention(arg[0].text,  # argument id
                                                                         arg[1].text,  # argument description
                                                                         mention_types[arg[2][0][0].text],
                                                                         # mention type (entity/proposition)
                                                                         int(arg[2][0][1].text),
                                                                         # entity/proposition id
                                                                         int(arg[2][0][2].text))

                                            # entity/proposition mention id
                                            for arg in mention.find('args')},
                                           mention.find('isExplicit').text == 'true'  # is explicit
                                           )

                    for mention in proposition.find('mentions')}

        # Check for empty mentions
        empty_mentions = [(mention.parent, m_id) for m_id, mention in mentions.iteritems() if len(mention.indices) == 0]
        if len(empty_mentions) > 0:
            logging.warning('Empty mentions in proposition %s' % proposition[0].text)

        if len(mentions) == 0:
            logging.warning('Proposition with no mentions: %s' % proposition[0].text)

        terms = set([mention.terms for mention in mentions.values()])

        # Proposition entailment graph
        explicit_mentions = [mention for mention in mentions.values() if mention.is_explicit]

        # Don't create an entailment graph for all implicit propositions
        if len(explicit_mentions) > 0:

            entailment_info = proposition[4]
            prop_terms = entailment_info[0]
            term_dic = { int(term[0].text) : term[1].text.lower() for term in prop_terms }
            graph = []
            contradictions_graph = []
            prop_connections = entailment_info[1]

            for connection in prop_connections:

                # The second entails the first or they are equal
                if connection[0].text == '1' or connection[0].text == '0':
                    graph.append((term_dic[int(connection[2].text)], term_dic[int(connection[1].text)]))

                # The first entails the second or they are equal
                if connection[0].text == '2' or connection[0].text == '0':
                    graph.append((term_dic[int(connection[1].text)], term_dic[int(connection[2].text)]))

                # Contradiction
                if connection[0].text == '3':
                    contradictions_graph.append((term_dic[int(connection[1].text)], term_dic[int(connection[2].text)]))

            # Create the transitive closure of the entailment graph
            final_graph = transitive_closure(graph)
            proposition_entailment = Entailment_graph(final_graph, None, contradictions_graph, None)

        else:
            proposition_entailment = Entailment_graph([], None, [], None)

        propositions[int(proposition[0].text)] = Proposition(int(proposition[0].text),  # id
                                                             proposition[1].text,  # name
                                                             mentions,  # proposition mentions
                                                             proposition[2].text,  # attributor
                                                             terms, proposition_entailment # predicate entailment graph
                                                             )

    okr = OKR(input_file, sentences, ignored_indices, tweet_ids, entities, propositions)

    return okr


def transitive_closure(graph):
    """
    Compute the transitive closure of the graph
    :param graph: a graph (list of directed pairs)
    :return: the transitive closure of the graph
    """
    closure = set(graph)
    while True:
        new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)

        closure_until_now = closure | new_relations

        if closure_until_now == closure:
            break

        closure = closure_until_now
    closure_no_doubles = [(x, y) for (x, y) in closure if not x == y]
    return closure_no_doubles


def set_template(prop_mention, entities, propositions):
    """
    Receives a proposition mention and returns a template, e.g. [A1] intercepted [A2]
    :param prop_mention: the proposition mention
    :param entities: the graph entities
    :param propositions: the graph propositions
    :return: the proposition template
    """
    indices, terms, argument_mentions = prop_mention.indices, prop_mention.terms, prop_mention.argument_mentions
    words_list = [(x, y) for x, y in zip(indices, terms.split())]

    words_list += [(entities[v.parent_id].mentions[v.parent_mention_id].indices[0], '[a' + str(int(k) + 1) + ']')
                   for k, v in argument_mentions.iteritems()
                   if v.mention_type == 0]

    words_list += [(propositions[v.parent_id].mentions[v.parent_mention_id].indices[0], '[a' + str(int(k) + 1) + ']')
                   for k, v in argument_mentions.iteritems()
                   if v.mention_type == 1]

    words_list.sort(key=lambda x: x[0])
    prop_mention.template = ' '.join([x[1] for x in words_list])


def set_parent_indices(arg, graph):
    """
    Copy the information about the argument "parent" - an entity or a proposition, to the argument itself
    :param arg: the argument mention
    :param graph: the OKR graph
    """
    orig_element = None
    name = ''

    # The argument is an entity
    if arg.mention_type == MentionType.Entity:

        # Parent mention ID is missing in the graph
        if arg.parent_mention_id not in graph.entities[arg.parent_id].mentions:
            warning = 'Error: missing parent ID %s.%s [%s] for entity argument %s [%s]' % \
                  (arg.parent_id, arg.parent_mention_id, graph.entities[arg.parent_id].terms, arg.id, arg.desc)
            logging.warn(warning)

        else:
            orig_element = graph.entities[arg.parent_id].mentions[arg.parent_mention_id]
            name = graph.entities[arg.parent_id].name

    # The argument is a proposition
    if arg.mention_type == MentionType.Proposition:

        # Parent mention ID is missing in the graph
        if arg.parent_mention_id not in graph.propositions[arg.parent_id].mentions:
            warning = 'Error: missing parent ID %s.%s [%s] for proposition argument %s [%s]' % \
                  (arg.parent_id, arg.parent_mention_id, graph.propositions[arg.parent_id].terms, arg.id, arg.desc)
            logging.warn(warning)

        else:
            orig_element = graph.propositions[arg.parent_id].mentions[arg.parent_mention_id]
            name = '*P' + str(arg.parent_id)

    arg.parent_indices = (orig_element.sentence_id, orig_element.indices)
    arg.parent_name = name


def from_term_id_to_mention_id(graph, mentions, mention_type):
    """
    Receives an entailment graph with IDs and returns an entailment graph with the
    actual terms (for entities) or predicate templates (for propositions).
    :param graph: an entailment graph for an entity/predicate with unique terms
    :param mentions: all mentions of one entity or proposition
    :param mention_type: mention type (proposition/entity)
    :return: an entailment graph of all mentions that match the terms
    """
    new_graph = []

    for (m1, m2) in graph:

        # Entity entailment - match by exact terms
        if mention_type == MentionType.Entity:
            m1_lst = [str(mention) for mention in mentions.values() if mention.terms == m1]
            m2_lst = [str(mention) for mention in mentions.values() if mention.terms == m2]

        # Proposition entailment - match by predicate template
        else:
            m1_lst = [str(mention) for mention in mentions.values() if mention.template == m1]
            m2_lst = [str(mention) for mention in mentions.values() if mention.template == m2]

        new_graph += list(itertools.product(m1_lst, m2_lst))

    return new_graph