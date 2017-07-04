""" Usage:
   props_wrapper

Author: Gabi Stanovsky

    Abstraction over the PropS parser.
    If ran with the interactive flag also starts a shell expecting raw sentences.
"""

from docopt import docopt
import logging
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
    def __init__(self):
        """
        Inits the underlying Berkeley parser and other relevant initializations.
        """
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

    def get_sentence(self):
        # Returns the tokenized sentence stored in this instance
        # @return - string, space separated sentence
        return " ".join([node.word
                          for node in sorted(self.dep_tree,
                                             key = lambda node: node.id)[1:]])  # Skip over ROOT node

    def get_predicates(self,
                       get_implicits,
                       get_zero_args,
                       get_conj,
    ):
        """
        Get this graph's predicate nodes.
        :param get_zero_args - Boolean controlling whether zero-argument predicates are
                               returned.
        :param get_implicites - Boolean controlling whether implicit predicates are
                                returned.
        :param get_conj - Boolean controlling whether conjunction predicates are
                                returned.
        """
        # define filters as lambdas - all should return True if node should be filtered
        zero_arg_filter = lambda node: (not get_zero_args) and (len(node.neighbors()) == 0)
        implicit_filter = lambda node: (not get_implicits) and node.is_implicit()
        conj_filter = lambda node: (not get_conj) and node.isConj()

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
        assert len(matching_dep_nodes) == 1,\
            "Problems matching {}; nodes matched were:{}; dep tree: {}".format(predicate_node.text[0].index,
                                                                               matching_dep_nodes,
                                                                               self.dep_tree
            )
        return matching_dep_nodes[0]

    def get_mwp(self,predicate_node):
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


    def get_template(self, predicate_node):
        """
        Given a predicate node - returns the predicate and arguments involved
        in this predicate
        :param predicate_node - PropS node, from which to extract the predicate
        """
        assert(predicate_node.isPredicate)
        dep_tree = self.get_dep_node(predicate_node)

        # Get the full bare predicate and generate a symbol for it
        bare_predicate = self.get_mwp(predicate_node)
        bare_predicate_str = " ".join([node.word for node in sorted(bare_predicate,
                                                                    key = lambda node: node.id)])
        predicate_symbol = self.get_element_symbol(self.get_node_ind(predicate_node),
                                                   self._gensym_pred)
        # Create template
        ## Collect items participating it from predicates and arguments
        predicate_items = [(node.id, node.word)
                           for node in bare_predicate]

        ## Get arguments which are predicates on their own
        dep_preds = [node
                     for node in self.get_props_neighbors(predicate_node)
                     if node.isPredicate]

        ## Get entity arguments
        dep_entities = [node
                        for node in self.get_props_neighbors(predicate_node)
                        if (not node.isPredicate)]

        # Concat, sort, and get the words forming the template
        all_template_elements = predicate_items + \
                                [(self.get_node_ind(node),
                                  self.get_element_symbol(self.get_node_ind(node),
                                                          self._gensym_pred))
                                  for node in dep_preds] + \
                                [(self.get_node_ind(node),
                                  self.get_element_symbol(self.get_node_ind(node),
                                                          self._gensym_ent))
                                  for node in dep_entities]

        logging.debug(all_template_elements)

        template = " ".join(map(itemgetter(1),
                                sorted(all_template_elements,
                                       key = lambda (ind, word): ind))) 

        # Store in this sentence's OKR
        self.predicates[predicate_symbol] = {"Bare predicate": bare_predicate_str,
                                             "Template": template}

        # # For each argument - parse according to whether it's a predicate or an argument
        # for rel, nodes in predicate_node.neighbors().iteritems():


    @staticmethod
    def get_node_original_text(node):
        """
        Given a node from the graph, returns a string concatenating all the words in its
        original string field.
        :param node - a node in the graph.
        """
        return " ".join(" ".join([str(w.word)
                                  for w in
                                  sorted(n.original_text,
                                         key = lambda w: w.index)]))
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

def main(pw, sent):
    """
    Run a batch of test commands, prints to screen, may return some variables
    for interactive inspection.
    @param pw - PropSWrapper instance
    @param sent - str, a raw sentence on which to run the commands.
    """
    pw.parse(sent)
    preds = pw.get_predicates(get_implicits = False,
                              get_zero_args = False,
                              get_conj = False)

    return pw.get_template(preds[0])

if __name__ == "__main__":
    """
    Simple unit tests
    """
    logging.basicConfig(level = logging.DEBUG)

    # Parse arguments
    args = docopt(__doc__)

    pw = PropSWrapper()
    sent = "The Syrian plane landed in Moscow."
    main(pw, sent)
    logging.debug(pformat(pw.get_okr()))
