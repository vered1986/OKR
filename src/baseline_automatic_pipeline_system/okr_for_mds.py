"""Usage:
   okr_for_mds --tweets=TWEETS_FILE --meta=TWEET_METADATA_FILE --out=OUTPUT_FILE

Example:
    python src/baseline_automatic_pipeline_system/okr_for_mds.py --out=boy_scouts.json \
        --tweets=examples/tweets/boy_scouts_tweets --meta=examples/tweets/tweets_metadata

TWEETS_FILE is a Tab-delimited file with: tweet-id <TAB> tweet-string
TWEET_METADATA_FILE is a Tab-delimited file with: tweet-id <TAB> author <TAB> author-id <TAB> timestamp
    each line is a tweet record.
OUTPUT_FILE would be the input json file for multi-document summarization

run it from base OKR directory.

Author: Ayal Klein

Process okr information for Multi Document Summarization.
the output is a json, following the mds_input_schema.json json-schema.
"""

import sys, os, logging
from docopt import docopt
from collections import defaultdict

# add all src sub-directories to path
for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

from dict_utils import clean_dict, rename_attribute

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
        #assert tweet_id in tweets_metadata, "tweet %s is missing metadata" % tweet_id
        if tweet_id in tweets_metadata:
            tweets_full_info[tweet_id] = { "id": tweet_id, "string": tweets_strings[tweet_id] }
            tweets_full_info[tweet_id].update(tweets_metadata[tweet_id])
        else:
            logging.warn("tweet %s is missing metadata, ignoring it" % tweet_id)
    return tweets_full_info


def class_object_to_json_convertable(obj):
    """ Return a json-representable object of obj. 
    * replace class-objects with thier __dict__ property.
    * call recursively on dictionary values and list\tuple elements.
    """
    primitives = [bool, int, float, str, unicode, list, tuple, dict]
    if type(obj) in primitives:
        # for dict and list - recursively run converter on elements\values
        if type(obj) in [list, tuple]:
            return [class_object_to_json_convertable(e) for e in obj]
        if type(obj) is dict:
            return {k: class_object_to_json_convertable(v) for k,v in obj.items() if v is not None}
        else:
            return obj
    # for complex class' objects
    else:
        return class_object_to_json_convertable(obj.__dict__)


def unify_mentions(mentions_dict):
    """ Unify mentions with same term. 
    return list of [ {"term": <term>, "sources": [source-tweet-ids] ]. 
    """
    unique_terms = set([mention["terms"] for mention in mentions_dict.values() ])
    return [{ "term" : term,
              # sentence_IDs of all mentions using this term
              "sources" : list(set([ mention['sentence_id']
                                      for mention in mentions_dict.values()
                                      if mention["terms"] == term ]))
            }
            for term in unique_terms ]


def prepare_proposition_predicates_and_arguments(mentions):
    """
    Prepare the proposition data to mds export (for a single Proposition).
    :param mentions: a dict { prop_mention_id : prop_mention } of the Proposition
    :return: (proposition["predicates"], proposition["arguments"])
    Plan:
    1. Transform templates to use aligned arguments - new arguments symbols ("slots"),  
    representing a semantic role relative to the predicate. one slot may refer to multiple concepts across various
    proposition-mentions. 
    The mapping process  between original argument values (which are concepts - known Entity\Proposition)
    to the new "consolidated" arguments slots is called Argument Alignment. 
    """
    # TODO - modify according to change in parse_okr_info
    templates = {mention_id : mention["template"] for mention_id, mention in mentions.iteritems()}
    sources_of_template = defaultdict(list) # a mapping from modified-template to a list of source tweet-ids
    # data-structures for the "arguments" json attribute:
    # a mapping from argument-slot (=arg-id) to a set of all concepts that take this slot
    slot_to_concept = defaultdict(set)
    # a mapping from (slot, concept-id) to list of sources of prop-mentions in which the concept is the value of the slot
    slot_concept_to_sources = defaultdict(list)
    for prop_mention_id, template in templates.iteritems():
        prop_mention = mentions[prop_mention_id]
        # iterate all args and replace old symbol (concept_id) with new symbol (slot)
        for arg_id, arg in prop_mention["argument_mentions"].iteritems():
            concept_id = arg["parent_id"]
            slot = arg_id
            # maintain mapping of slot-concept relation
            slot_to_concept[slot].add(concept_id)
            # maintain sources of slot-concept relation
            slot_concept_to_sources[(slot, concept_id)].append(prop_mention["sentence_id"])
        # add this prop-mention's tweet-id to sources of this modified_template
        sources_of_template[template].append(mentions[prop_mention_id]["sentence_id"])

    # prepare "predicates" attribute for schema format
    templates_att = [ {"template": template,
                      "sources": sources_of_template[template] }
                     for template in templates.values()]
    predicates_att = {"templates": templates_att}

    # prepare "arguments" attribute for schema format
    slots = set(slot_to_concept.keys())
    arguments_att = []
    for slot in slots:
        argument_json_object = {}
        argument_json_object["id"] = slot
        argument_json_object["values"] = [{"concept-id": concept,
                                           "sources": slot_concept_to_sources[(slot, concept)]}
                                          for concept in slot_to_concept[slot]]
        # TODO improve labeling of aligned arguments
        # baseline: take label of a random argument-mention
        relevant_arg_mentions = [ arg_mention
                                  for prop_mention in mentions.values()
                                  for arg_id, arg_mention in prop_mention["argument_mentions"].iteritems()
                                  if arg_id == slot]
        representative_argument_mention = relevant_arg_mentions[0]
        argument_json_object["label"] = representative_argument_mention["desc"]

        # add this argument value (=concept) to list of this slot
        arguments_att.append(argument_json_object)

    return predicates_att, arguments_att



def prepare_okr_info_to_export(okr_info):
    """
    :param okr_info containing only primitive python types (i.e. json-convertable)  
    :return: okr_info in a format suitable for mds (following the mds_input_schema.json) 
    """
    # remove top-level unnecessary attributes
    clean_dict(okr_info, ["tweets", "entities", "propositions"])
    # prepare entities
    for entity in okr_info["entities"].values():
        rename_attribute(entity, old_key_name="name", new_key_name="alias")
        clean_dict(entity, ['mentions', 'alias', 'id'])
        # prepare entity -> mentions
        entity["mentions"] = unify_mentions(entity["mentions"])    # convert to list
    # prepare propositions
    for proposition in okr_info["propositions"].values():
        rename_attribute(proposition, old_key_name="name", new_key_name="alias")
        proposition["attribution"] = None   # TODO parse attribution
        # prepare proposition -> predicates and proposition -> arguments out of proposition -> mentions
        proposition["predicates"], proposition["arguments"] = \
            prepare_proposition_predicates_and_arguments(proposition["mentions"])

        clean_dict(proposition, ['arguments', 'predicates', 'alias', 'id', 'attribution'])

    return okr_info


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
    tweets_strings = {tweet_id : tweet["string"] for tweet_id, tweet in tweets.iteritems() }
    # retrieve okr info - explicitly call all pipeline stages, for debugging
    import parse_okr_info as prs
    parsed_sentences = prs.parse_single_sentences(tweets_strings)
    all_entity_mentions, all_proposition_mentions = prs.get_mention_lists(parsed_sentences)
    entities = prs.cluster_entities(all_entity_mentions)
    propositions = prs.cluster_propositions(all_proposition_mentions)
    okr_info = prs.generate_okr_info(tweets_strings, all_entity_mentions, all_proposition_mentions, entities, propositions)

    okr_info["tweets"] = tweets     # tweets are aligned with mds requirements
    okr_json = prepare_okr_info_to_export(class_object_to_json_convertable(okr_info))

    # export okr json to file and log
    import json
    with open(output_fn, "w") as fout:
        json.dump(okr_json, fout, sort_keys=True, indent=3)
    logging.debug(okr_json)
