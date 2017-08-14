"""Usage:
   okr_for_mds --tweets=TWEETS_FILE --meta=TWEET_METADATA_FILE --out=OUTPUT_FILE

TWEETS_FILE is a Tab-delimited file with: tweet-id <TAB> tweet-string
TWEET_METADATA_FILE is a Tab-delimited file with: tweet-id <TAB> author <TAB> author-id <TAB> timestamp
    each line is a tweet record.

Author: Ayal Klein

run it from base OKR directory.

Process okr information for Multi Document Summarization.
the output is a json, following the mds_input_schema.json json-schema.
"""

import sys, os, logging
from docopt import docopt
from collections import defaultdict

# add all src sub-directories to path
for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

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


def clean_dict(dictionary, desired_attributes):
    """ Remove all attributes of dictionary beside those in desired_attributes. """
    remove_list = (att for att in dictionary.keys() if att not in desired_attributes)
    for att in remove_list:
        del dictionary[att]


def rename_attribute(dictionary, old_key_name, new_key_name):
    dictionary[new_key_name] = dictionary.pop(old_key_name)


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


def argument_alignment(prop_mentions):
    """
    :return: list of lists, a grouping of (prop-mention-id, arg-mention-id) to arguments slots. 
    """

    # Currently, the argument alignment is simply grouping the argument referring to same concept.
    concepts = set( [arg_mention["parent_id"]
                     for prop_mention in prop_mentions.values()
                     for arg_mention in prop_mention['argument_mentions'].values() ] )

    concept_to_slot_index = { concept : index for index, concept in enumerate(concepts) }
    grouping = defaultdict(list)
    for prop_mention_id, prop_mention in prop_mentions.items():
        for arg_mention_id, arg_mention in prop_mention['argument_mentions'].items():
            slot_index = concept_to_slot_index[arg_mention["parent_id"]]
            grouping[slot_index].append( (prop_mention_id, arg_mention_id) )

    return grouping.values()


def prepare_proposition_predicates_and_arguments(mentions):

    """
    Prepare the proposition data to mds export.
    :param mentions: a dict { prop_mention_id : prop_mention }
    :return: (proposition["predicates"], proposition["arguments"])
    Plan:
    1. Transform templates to use aligned arguments - new arguments symbols ("slots"),  
    representing a semantic role relative to the predicate. one slot may refer to multiple concepts across various
    proposition-mentions. 
    The mapping process  between original argument values (which are concepts - known Entity\Proposition)
    to the new "consolidated" arguments slots is called Argument Alignment.
    
    """

    argument_slots_grouping = argument_alignment(mentions)
    # mapping from (prop_mention_id, arg_mention_id) to a symbol of (proposition-level) argument-slot
    ids_to_slot = { arg_ids : "a." + str(slot_index)
                    for slot_index, slot_group in enumerate(argument_slots_grouping, start=1)
                    for arg_ids in slot_group }

    templates = {m_id : m["template"] for m_id, m in mentions.items()}
    # transform templates - replace concept-id with general (proposition-level) argument-slot
    modified_templates = set()
    sources_of_template = defaultdict(list) # a mapping from modified-template to a list of source tweet-ids
    # data-structures for the "arguments" json attribute:
    # a mapping from argument-slot (symbol) to a set of all concepts that take this slot
    slot_to_concept = defaultdict(set)
    # a mapping from (slot, concept-id) to list of sources of prop-mentions in which the concept is the value of the slot
    slot_concept_to_sources = defaultdict(list)
    for prop_mention_id, template in templates.items():
        prop_mention = mentions[prop_mention_id]
        modified_template = template
        # iterate all args and replace old symbol (concept_id) with new symbol (slot)
        for arg_id, arg in prop_mention["argument_mentions"].items():
            concept_id = arg["parent_id"]
            slot = ids_to_slot[(prop_mention_id, arg_id)]
            modified_template = modified_template.replace("{"+concept_id+"}", "{"+slot+"}")
            # maintain mapping of slot-concept relation
            slot_to_concept[slot].add(concept_id)
            # maintain sources slot-concept relation
            slot_concept_to_sources[(slot, concept_id)].append(prop_mention["sentence_id"])
        # add new template to set
        modified_templates.add(modified_template)
        # add this prop-mention's tweet-id to sources of this modified_template
        sources_of_template[modified_template].append(mentions[prop_mention_id]["sentence_id"])

    # prepare "predicates" attribute for schema format
    templates_att = [ {"template": new_template,
                      "sources": sources_of_template[new_template] }
                     for new_template in modified_templates]
    predicates_att = {"templates": templates_att}

    # prepare "arguments" attribute for schema format
    slots = ids_to_slot.values()
    arguments_att = []
    for slot in slots:
        argument_json_object = {}
        argument_json_object["id"] = slot
        argument_json_object["values"] = [{"concept-id": concept,
                                           "sources": slot_concept_to_sources[(slot, concept)]}
                                          for concept in slot_to_concept[slot]]
        # TODO improve labeling of aligned arguments
        # baseline: take label of a random argument-mention
        ids_of_corresponding_argument_mentions = [arg_ids for arg_ids, s in ids_to_slot.items() if s==slot]
        representative_argument_mention_ids = ids_of_corresponding_argument_mentions[0]
        rep_mention_id, rep_arg_id = representative_argument_mention_ids
        representative_argument_mention = mentions[rep_mention_id]["argument_mentions"][rep_arg_id]
        argument_json_object["label"] = representative_argument_mention["desc"]

        # add this argument value (=concept) to list of this slot
        arguments_att.append(argument_json_object)

    return predicates_att, arguments_att



def prepare_okr_info_to_export(okr_info):
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
    tweets_strings = {tweet_id : tweet["string"] for tweet_id, tweet in tweets.items() }
    # retrieve okr info
    from parse_okr_info import auto_pipeline_okr_info
    okr_info = auto_pipeline_okr_info(tweets_strings)
    okr_info["tweets"] = tweets     # tweets are aligned with mds requirements
    okr_json = prepare_okr_info_to_export(class_object_to_json_convertable(okr_info))

    # export okr json to file and log
    import json
    json.dump(okr_json, open(output_fn, "w"))
    logging.info(okr_json)
