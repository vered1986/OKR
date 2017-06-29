"""
Author: Gabi Stanovsky

    Abstraction over the PropS parser.
"""

import logging
from pprint import pprint, pformat

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
        # A dictionary mapping from token indices to element symbols
        # (either predicates or entitites)
        self.tok_ind_to_symbol = {}

    def get_element_symbol(self, tok_ind, symbol_generator):
        """
        Get a unique predicate or entity symbol. Creates it if it doesn't exists
        and associate with a token index.
        :param tok_ind - int, the token for which to obtain the symbol
        :param symbol_generator - func, to call in case the tok_ind is not associated
                                  with a symbol
        """
        if tok_ind in self.tok_ind_to_symbol:
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
                                 for word in props_node.text]
        return word_ind in all_dependent_indexes

    def get_dep_node(self, predicate_node):
        """
        Get the corresponding dep node for a PropS node
        :param predicate_node - PropsNode
        """
        matching_dep_nodes = [node
                              for node in predicate_node.isPredicate[0].nodes()
                              if node.id in [w.index for w in predicate_node.text]]
        
        assert(len(matching_dep_nodes) == 1)
        return matching_dep_nodes[0]

    def get_mwp(self,predicate_node):
        """
        Returns the multiword predicate rooted in the given node.
        In form of a list of dep nodes, to record the word index and the
        :param predicate_node - PropS node, from which to extract the predicate
        """
        # Approach:
        # Identify nodes in dep tree which are related with to the predicate
        # with one of PropS' ignore labels

        assert(predicate_node.isPredicate)
        # Get the corresponding dep tree node
        dep_node = self.get_dep_node(predicate_node)

        # Returns a list of dep nodes which aren't dependent in the PropS graph
        return [dep_node] + [dep_child
                             for dep_child in dep_node.get_children()
                             if dep_child.parent_relation in ignore_labels]


    def get_template(self, predicate_node):
        """
        Returns a multi-word predicate as a list of Word objects
        :param predicate_node - A predicate node instance
        """
        assert(predicate_node.isPredicate)
        dep_tree = predicate_node.isPredicate[0]


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
    def gensym_pred(self):
        """
        Generate a unique predicate symbol name
        """
        pred_counter += 1
        return "P{}".format(self.pred_counter)

    def gensym_ent(self):
        """
        Generate a unique entity symbol name
        """
        ent_counter += 1
        return "E{}".format(self.ent_counter)



if __name__ == "__main__":
    """
    Simple unit tests
    """
    logging.basicConfig(level = logging.DEBUG)
    pw = PropSWrapper()
    pw.parse("John, the new ambassador, wanted to take the box from Mary and give it to Bob ")
    preds = pw.get_predicates(get_implicits = False,
                              get_zero_args = False,
                              get_conj = False)
    logging.info(pformat(pw.graph))
    mwps = [(p, pw.get_mwp(p))
            for p in preds]
    mwps_str = [(pw.props_node_to_string(pred),
                 " ".join([n.word
                          for n in sorted(mwp,
                                          key = lambda n: n.id)]))
                 for pred, mwp in mwps]
    logging.info(pformat(mwps_str))

