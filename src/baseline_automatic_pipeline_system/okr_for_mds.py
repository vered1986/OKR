"""Usage:
   okr_for_mds --tweets=TWEETS_FILE --meta=TWEET_METADATA_FILE --out=OUTPUT_FILE

Author: Ayal Klein

TWEETS_FILE is a Tab-delimited file with: tweet-id <TAB> tweet-string
TWEET_METADATA_FILE is a Tab-delimited file with: tweet-id <TAB> author <TAB> author-id <TAB> timestamp
    each line is a tweet record.

the output is a json, following the mds_input_schema.json json-schema.
"""

import sys, logging
from docopt import docopt
sys.path.append("../common")
sys.path.append("../baseline_system")

def get_tweets_from_files(tweets_fn, metadata_fn):
    """ Get tweets information from two input files.
    @:returns: a dict { tweet-id : tweet-info }.
    tweet_info is a dict with "string", "id", "timestamp", and "author" attributes.
    """
    import csv
    # retrieve tweets_strings = { tweet-id : tweet-string } from tweets file
    tweets_strings_list = list(csv.reader(open(tweets_fn, 'rb'), delimiter='\t'))
    tweets_strings = dict([r for r in tweets_strings_list if len(r)==2 and r[0][0] != "#"])
    # retrieve tweets_metadata = { tweet-id : {"author":author, "timestamp":timestamp} }
    tweets_metadata_list = list(csv.reader(open(metadata_fn, 'rb'), delimiter='\t'))
    # every record in tweets_metadata_list is a (tweet-id, author, author-id, timestamp) tuple
    tweets_metadata = { r[0] : {"author":r[1], "timestamp":r[3]} for r in tweets_metadata_list if len(r) == 4}

    # combine information to one dictionary
    tweets_full_info = {}
    for tweet_id in tweets_strings:
        assert tweet_id in tweets_metadata, "tweet %s is missing metadata" % tweet_id
        tweets_full_info[tweet_id] = { "id": tweet_id, "string": tweets_strings[tweet_id] }
        tweets_full_info[tweet_id].update(tweets_metadata[tweet_id])
    return tweets_full_info

# main
if __name__ == "__main__":
    # general settings
    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    args = docopt(__doc__)
    tweets_fn = args["--tweets"]
    metadata_fn = args["--meta"]
    output_fn = args["--out"]

    # read tweets from input files
    tweets = get_tweets_from_files(tweets_fn, metadata_fn)
