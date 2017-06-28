"""
Author: Gabi Stanovsky

    Abstraction over the PropS parser.
"""

import logging
from pprint import pformat

from props.applications.run import parseSentences, load_berkeley

class PropSWrapper:
    """
    Class to give access to PropS variables.
    and the perform some preprocessing, where needed.
    Works in a parse-then-ask paradigm, where sentences are first parsed,
    and then certain inquires on them are supported.
    """
    def __init__(self):
        """
        Inits the underlying Berkeley parser.
        """
        load_berkeley(tokenize = False)

    def parse(self, sent):
        """
        Parse a raw sentence - shouldn't return a value, but properly change the internal status.
        :param sent - string, raw tokenized sentence (split by single spaces)
        """
        # Get PropS graph for this sentence
        # (ignore the textual tree representation)
        self.graph, _ = parseSentences(sent)[0]

if __name__ == "__main__":
    """
    Simple unit tests
    """
    logging.basicConfig(level = logging.DEBUG)
    pw = PropSWrapper()
    pw.parse("John wanted to fly")
    logging.info(pformat(pw.graph))
