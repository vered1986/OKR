"""Usage:
   okr_for_mds --tweets=TWEETS_FILE --meta=TWEET_METADATA_FILE --out=OUTPUT_FILE

Author: Ayal Klein

TWEETS_FILE is a Tab-delimited file with: tweet-id <TAB> tweet-string
TWEET_METADATA_FILE is a Tab-delimited file with: tweet-id <TAB> author <TAB> author-id <TAB> timestamp
    each line is a tweet record.

the output is a json, following 
"""

import sys, logging
from docopt import docopt
sys.path.append("../common")
sys.path.append("../baseline_system")

# main
if __name__ == "__main__":
    # general settings
    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    args = docopt(__doc__)
    tweets_fn = args["--tweets"]
    metadata_fn = args["--meta"]
    output_fn = args["--out"]

