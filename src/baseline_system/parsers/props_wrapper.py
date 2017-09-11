""" Usage:
   props_wrapper --in=INPUT_FILE --out=OUTPUT_FILE

Author: Gabi Stanovsky

    Abstraction over the PropS parser.
    If ran with the interactive flag also starts a shell expecting raw sentences.
    Parses sentences from input file (not tokenized), and writes an output json to the output file.
"""

import logging
import json

from docopt import docopt
from pprint import pprint, pformat
from collections import defaultdict
from operator import itemgetter

from props.applications.run import parseSentences, load_berkeley
from props.graph_representation.graph_wrapper import ignore_labels

class PropSWrapper:
    """
    Class to give access to PropS variables.
    and the perform some preprocessing, where needed.
    Works in a parse-then-ask paradigm, where sentences are first parsed,
    and then certain inquires on them are supported.
    """
    def __init__(self, get_implicits, get_zero_args, get_conj):
        """
        Inits the underlying Berkeley parser and other relevant initializations.
        Populate the entities and predicates list of this sentence.
        :param get_zero_args - Boolean controlling whether zero-argument predicates are
                               returned.
        :param get_implicits - Boolean controlling whether implicit predicates are
                                returned.
        :param get_conj - Boolean controlling whether conjunction predicates are
                                returned.
        """
        self.get_implicits = get_implicits
        self.get_zero_args = get_zero_args
        self.get_conj = get_conj
        load_berkeley(tokenize = False)

    def _init_internal_state(self):
        """
        Initialize all internal status towards new parse.
        Should be called internally before parsing
        """
        self.pred_counter = 0
        self.ent_counter = 0
        self.dep_tree = None
        self.sentence = ''
        self.entities = {}
        self.predicates = {}
        # A dictionary mapping from token indices to element symbols
        # (either predicates or entitites)
        self.tok_ind_to_symbol = {}

    def get_okr(self):
        """
        Get this sentence OKR in json format
        """
        return {"Sentence": self.sentence,
                "Entities": self.entities,
                "Predicates": self.predicates}

    def get_element_symbol(self, tok_ind, symbol_generator):
        """
        Get a unique predicate or entity symbol. Creates it if it doesn't exists
        and associate with a token index.
        :param tok_ind - int, the token for which to obtain the symbol
        :param symbol_generator - func, to call in case the tok_ind is not associated
                                  with a symbol
        """
        if tok_ind not in self.tok_ind_to_symbol:
            # generate symbol in case it doesn't exist
            self.tok_ind_to_symbol[tok_ind] = symbol_generator()
        return self.tok_ind_to_symbol[tok_ind]

    def parse(self, sent):
        """
        Parse a raw sentence - shouldn't return a value, but properly change the internal status.
        :param sent - string, raw tokenized sentence (split by single spaces)
        """
        # Init internal state
        self._init_internal_state()

        # Get PropS graph for this sentence
        # (ignore the textual tree representation)
        self.graph, _ = parseSentences(sent)[0]

        # Get the dependency tree
        self.dep_tree = self.graph.dep_tree.values()

        # Get the tokenized sentence
        self.sentence = self.get_sentence()

        # Populate entities and predicates
        self.parse_okr()

    def parse_okr(self):
        """
        Populate the entities and predicates list of this sentence.
        """
        # For each predicate, add its nested propositions to the OKR
        self.predicate_nodes = self.get_predicates()
        for pred in self.predicate_nodes:
            self.parse_predicate(pred)

        # Split entities according to OKR notations
        self.entities = self._split_entities()

    # A static symbol representing the string and
    # index associated with implicit predicates
    IMPLICIT_SYMBOL = ("IMPLICIT", tuple([-1]))

    def create_implicit_proposition(self, *arg_symbols):
        """
        Returns an implicit proposition over the given argument symbols.
        """
        return {"Bare predicate": PropSWrapper.IMPLICIT_SYMBOL,
                "Template": " ".join(arg_symbols), # The template ommits the symbol
                "Head":{
                    "Surface": PropSWrapper.IMPLICIT_SYMBOL[0],
                    "Lemma": PropSWrapper.IMPLICIT_SYMBOL[0],
                    "POS": PropSWrapper.IMPLICIT_SYMBOL[0]
                },
                "Arguments": arg_symbols
            }

    def _split_entities(self):
        """
        After getting longer PropS entities composed of NPs - further split
        them according to the OKR notion of single word entities.
        """
        ret = {}

        # Get mapping from all symbol to the word index of their head
        ent_symbol_to_head_ind = dict([(v, k) for (k, v) in self.tok_ind_to_symbol.iteritems()])

        # Iterate over entities and split where necessary
        for ent_head_symbol, (ent_str, ent_indices) in self.entities.iteritems():
            for word, ind in zip(ent_str.split(" "),
                                 ent_indices):
                cur_dep_node = self.dep_tree[ind + 1]

                if ind + 1 ==  ent_symbol_to_head_ind[ent_head_symbol]:
                    # Replace this entity with its head
                    ret[ent_head_symbol] = (word, tuple([ind]))
                else:
                    if  (cur_dep_node.parent_relation in ['det']):
                        # Dont include determiners or prepositions
                        continue

                    if (cur_dep_node.parent_relation in ['prep']) and \
                         (len(cur_dep_node.children) == 1) and \
                         (cur_dep_node.children[0].id - 1 in ent_indices):
                        # This is a preposition whose sole child is in this span
                        # -> Add the preposition as predicate
                        logging.debug("found prep: {} {}".format(cur_dep_node.children[0].id - 1,
                                                                 ent_indices))
                        prep_child = cur_dep_node.children[0]
                        prep_child_symbol = self.get_element_symbol(prep_child.id,
                                                                    self._gensym_ent)
                        ret[prep_child_symbol] = (prep_child.word,
                                                  tuple([prep_child.id - 1]))

                        prep_symbol = self.get_element_symbol(ind + 1,
                                                              self._gensym_pred)

                        self.predicates[prep_symbol] = {"Bare predicate": (word,
                                                                           tuple([ind])),
                                                        "Template": " ".join([ent_head_symbol,
                                                                              word,
                                                                              prep_child_symbol]),
                                                        "Head":{
                                                            "Surface": word,
                                                            "Lemma": word,
                                                            "POS": "IN",
                                                        },
                                                        "Arguments": [ent_head_symbol,
                                                                      prep_child_symbol]
                                                                  }

                    elif not ((cur_dep_node.parent_relation == 'pobj') and \
                              (cur_dep_node.parent.id - 1 in ent_indices)):
                        # If not a preposition, then add an implicit relation to the head
                        # Start by creating new symbols for this word
                        new_ent_symbol = self.get_element_symbol(ind + 1,
                                                                 self._gensym_ent)
                        ret[new_ent_symbol] = (word, tuple([ind]))

                        # Then add an implicit relation
                        self.predicates[self._gensym_pred()] = self.create_implicit_proposition(new_ent_symbol,
                                                                                                ent_head_symbol)
        return ret

    def get_sentence(self):
        """
        Returns the tokenized sentence stored in this instance
        @return - string, space separated sentence
        """
        return " ".join([node.word
                          for node in sorted(self.dep_tree,
                                             key = lambda node: node.id)[1:]])  # Skip over ROOT node

    def get_predicates(self):
        """
        Get this graph's predicate nodes.
        """
        # define filters as lambdas - all should return True if node should be filtered
        zero_arg_filter = lambda node: (not self.get_zero_args) and (len(node.neighbors()) == 0)
        implicit_filter = lambda node: (not self.get_implicits) and node.is_implicit()
        conj_filter = lambda node: (not self.get_conj) and node.isConj()

        # Concat all filters
        is_valid_pred = lambda node: node.isPredicate and \
                        all([not func(node)
                             for func in [zero_arg_filter,
                                          implicit_filter,
                                          conj_filter]])
        return [node
                for node in self.graph.nodes()
                if is_valid_pred(node)]

    def props_node_to_string(self, node):
        """
        Returns a string represnting the given node.
        :param node - PropS node
        """
        return " ".join([word.word
                         for word
                         in sorted(node.text,
                                   key = lambda word: word.index)])


    def is_props_dependent(self, pred_node, word_ind):
        """
        Checks whether a pred a word index represents a node which is
        dependent on a predicate node in the PropS graph
        :param pred_node - PropS node
        :param word_ind - int
        """
        all_dependent_indexes = [word.index
                                 for props_rel, props_nodes in pred_node.neighbors().iteritems()
                                 for props_node in props_nodes
                                 for word in props_node.text
        ]
        return word_ind in all_dependent_indexes

    def get_dep_node(self, predicate_node):
        """
        Get the corresponding dep node for a PropS node
        :param predicate_node - PropsNode
        """
        matching_dep_nodes = [node
                              for node in self.dep_tree
                              if node.id in [w.index for w in predicate_node.text]]
        logging.debug("matching predicate node: {}".format(predicate_node))

        # Return the top most node in the dependency tree
        return min(matching_dep_nodes,
                   key = lambda dep_node: self.get_dep_height(dep_node))


    def get_dep_height(self, dep_node):
        """
        Returns the height of a given dependency node
        I.e., the number of nodes between it and the ROOT.
        """
        return (1 + self.get_dep_height(dep_node.parent)) \
            if (dep_node.parent is not None) \
               else 0

    def get_mwp(self, predicate_node):
        """
        Returns the multiword predicate rooted in the given node.
        In form of a list of dep nodes, to record the word index and the
        :param predicate_node - PropS node, from which to extract the predicate
        """
        # Approach:
        # Identify nodes in dep tree which are related with to the predicate
        # with one of PropS' ignore labels + preposition label

        assert(predicate_node.isPredicate)

        # Get the corresponding dep tree node
        dep_node = self.get_dep_node(predicate_node)

        # Returns a list of dep nodes which aren't dependent in the PropS graph
        # and that are auxiliaries, according to PropS
        return [dep_node] + [dep_child
                             for dep_child in dep_node.get_children()
                             if (dep_child.parent_relation in PropSWrapper.AUX_LABELS) \
                             and not (self.is_props_dependent(predicate_node,
                                                              dep_child.id))]

    def get_node_ind(self, node):
        """
        Return the minimal index of either a props or dep node, hopefully it's unique.
        Bridges over inconsistencies between PropS and dependency names for their indices.
        """
        try:
            # First, try to treat this node as a PropS node
            return min([w.index for w in node.text])

        except:
            # If fails, assume it's a dep node
            return node.id

    def get_props_neighbors(self, node):
        """
        Returns a flat list of neighbors.
        @param node - Props node.
        """
        return [neighbor
                for neighbors_list in node.neighbors().values()
                for neighbor in neighbors_list]


    def parse_predicate(self, predicate_node):
        """
        Given a predicate node, populates the entities and predicates nested under it in the PropS graph.
        :param predicate_node - PropS node, from which to extract the predicate
        """
        assert(predicate_node.isPredicate)
        dep_tree = self.get_dep_node(predicate_node)

        # Get the full bare predicate and generate a symbol for it
        bare_predicate = self.get_mwp(predicate_node)
        bare_predicate_str = " ".join([node.word for node in sorted(bare_predicate,
                                                                    key = lambda node: node.id)])
        bare_predicate_indices = [node.id - 1 for node in sorted(bare_predicate,
                                                             key = lambda node: node.id)]
        predicate_symbol = self.get_element_symbol(self.get_node_ind(predicate_node),
                                                   self._gensym_pred)
        # Create template
        ## Collect items participating in it from predicates and arguments
        predicate_items = [(node.id, node.word)
                           for node in bare_predicate]

        ## Get arguments which are predicates on their own
        dep_preds = [node
                     for node in self.get_props_neighbors(predicate_node)
                     if node in self.predicate_nodes]

        ## Get entity arguments
        dep_entities = [node
                        for node in self.get_props_neighbors(predicate_node)
                        if (node not in self.predicate_nodes)]

        # Concat, sort, and get the words forming the template
        # Element placeholders appear with curly brackets, for replacement with format
        all_template_elements = predicate_items + \
                                [(self.get_node_ind(node),
                                  '{{{}}}'.format(self.get_element_symbol(self.get_node_ind(node),
                                                                          self._gensym_pred)))
                                  for node in dep_preds] + \
                                [(self.get_node_ind(node),
                                  '{{{}}}'.format(self.get_element_symbol(self.get_node_ind(node),
                                                                          self._gensym_ent)))
                                  for node in dep_entities]

        logging.debug(all_template_elements)

        template = " ".join(map(itemgetter(1),
                                sorted(all_template_elements,
                                       key = lambda (ind, word): ind))) 

        # Store in this sentence's OKR
        self.predicates[predicate_symbol] = {"Bare predicate": (bare_predicate_str,
                                                                tuple(bare_predicate_indices)),
                                             "Template": template,
                                             "Head":{
                                                 "Surface": (dep_tree.word,
                                                             [dep_tree.id - 1]),
                                                 "Lemma": predicate_node.features.get('Lemma', ''),
                                                 "POS": dep_tree.pos,
                                             },
                                             "Arguments":[self.get_element_symbol(self.get_node_ind(node),
                                                                                  self._gensym_ent)
                                                          for node in dep_entities] + \
                                             [self.get_element_symbol(self.get_node_ind(node),
                                                                      self._gensym_pred)
                                              for node in dep_preds]
        }

        # Add entities by iterating over dep_entities and getting the entire subtree text
        for node in dep_entities:
            ent_symbol = self.get_element_symbol(self.get_node_ind(node),
                                                 self._gensym_ent)
            try:
                sorted_entity = sorted(set(node.original_text),
                                       key = lambda n: n.index)
            except:
                sorted_entity = sorted(set(node.str),
                                       key = lambda n: n.index)

            self.entities[ent_symbol] = (" ".join([w.word
                                                   for w in sorted_entity]),
                                         tuple([w.index - 1
                                                for w in sorted_entity]))

        #TODO: Known bug in conjunctions - John wanted to take the book from Bob and give it to Mary

    def _gensym_pred(self):
        """
        Generate a unique predicate symbol name.
        (Should be called from get_element_symbol)
        """
        self.pred_counter += 1
        return "P{}".format(self.pred_counter)

    def _gensym_ent(self):
        """
        Generate a unique entity symbol name.
        (Should be called from get_element_symbol)
        """
        self.ent_counter += 1
        return "A{}".format(self.ent_counter)

    # Constants
    # Add a few labels to PropS' auxiliaries
    AUX_LABELS = ["det", "neg", "aux",
                  "auxpass", "prep", "cc",
                  "conj"]

if __name__ == "__main__":
    """
    Simple unit tests and examples of usage
    """
    logging.basicConfig(level = logging.INFO)

    # Parse arguments
    args = docopt(__doc__)
    input_fn = args["--in"]
    output_fn = args["--out"]

    # Example of usage:

    # 1. Initialize PropSWrapper
    pw = PropSWrapper(get_implicits = False,
                      get_zero_args = False,
                      get_conj = False)

    okrs = []
    for line in open(input_fn):
        sent = line.strip()

        if not sent or sent.startswith('#'):
            # Ignore commented lines
            continue

        # 2. Parse sentence
        pw.parse(sent)

        # 3. Get OKR Json object
        okr = pw.get_okr()
        okrs.append(okr)
        logging.info(pformat(okr))

    # Dump json
    with open(output_fn, 'w') as fout:
        json.dump(okrs, fout)
