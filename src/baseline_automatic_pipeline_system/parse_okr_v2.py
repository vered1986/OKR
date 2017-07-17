"""Usage:
   parse_okr_v2 --in=INPUT_FILE --out=OUTPUT_FILE

Author: Ayal Klein

baseline for automatic pipeline for okr-v2.
run it from baseline_automatic_pipeline_system directory.

steps:
1. get list of sentences (tweets).
2. parse each sentence to get single-sentence semantic representation
3. combine: create combined lists of EntityMentions and of PropositionMentions (=predicate template is a proposition-mention)
4. use baseline coref system to cluster EntityMentions to Entities
5. use baseline coref system to cluster PropositionMentions to Propositions
6. create okr object based on the pipeline results
7. evaluate automatic-pipline results
8. generate output json

"""

import sys, logging
from docopt import docopt
sys.path.append("../common")
sys.path.append("../baseline_system")

from parsers.props_wrapper import PropSWrapper
import okr

# TODO handle input (tweets raw data)
# instead of step 1 - meantime suppose its just a simple text file of line-separated sentences, given in tweets_file
tweets_file = "../../examples/raw_sentences"
def get_raw_sentences_from_file(input_fn):
    """ Get list of raw sentence out of file. ignore comments. """
    raw_sentences = []
    for line in open(input_fn):
        sent = line.strip()
        if not sent or sent.startswith('#'):
            # Ignore commented lines
            continue
        raw_sentences.append(sent)

    # make it a dict {id : sentence}, id starting from 1
    sentences_dict = {"S"+str(id) : sentence for id, sentence in enumerate(raw_sentences, start=1)}
    return sentences_dict

# step 2
def parse_single_sentences(raw_sentences):
    """ return a list of parsed single-sentence representation. uses props as parser."""
    pw = PropSWrapper(get_implicits=False,
                      get_zero_args=False,
                      get_conj=False)
    parsed_sentences = {}
    for sent_id, sent in raw_sentences.items():
        pw.parse(sent)
        parsed_sent = pw.get_okr()
        parsed_sentences[sent_id] = parsed_sent
    return parsed_sentences

#step 3
def get_mention_lists(parsed_sentences):
    """ return combined lists of EntityMentions and of PropositionMentions. """

    all_entity_mentions = {}        # { global-unique-id : entity-mention-info-dict }
    all_proposition_mentions = {}   # { global-unique-id : proposition-mention-info-dict }
                                    # global-unique-id is composed from sent_id + "_" + key(=id-symbol at props_wrapper)

    for sent_id, parsed_sent in parsed_sentences.items():
        # *** entities ***
        # mentions_keys_terms_indices = [(key, unicode(terms), indices, sent_id) for key, (terms, indices) in parsed_sent["Entities"].items()]
        # # each mention is a (key, id, terms) triplet
        # sentence_entity_mentions = [(mention_info[0], id)+ mention_info[1:] for id, mention_info in zip(mentions_ids, mentions_keys_terms_indices)]
        # # add all entity-mentions of sentence to global list
        # all_entity_mentions.extend(sentence_entity_mentions)

        sentence_entity_mentions = { sent_id + "_" + key :
                                         { "terms":          unicode(terms),
                                           "indices":       indices,
                                           "sentence_id":   sent_id }
                                     for key, (terms, indices) in parsed_sent["Entities"].items() }
        all_entity_mentions.update(sentence_entity_mentions)

        # *** propositions ***
        sentence_proposition_mentions = { str(sent_id) + "_" + key :
                                              dict( {"sentence_id" : sent_id},
                                                    **prop_ment_info )
                                          for key, prop_ment_info in parsed_sent["Predicates"].items() }
        all_proposition_mentions.update(sentence_proposition_mentions)

    return all_entity_mentions, all_proposition_mentions

# step 4
def cluster_entities(all_entity_mentions):
    import eval_entity_coref as entity_coref
    from clustering_common import cluster_mentions
    """
    must cluster hashable items (uses set()) - thus giving only the unique id as first element.
    only second element (terms) is being used for clustering.
    """
    mentions_for_clustering = [ (mention_id, mention_info["terms"])
                                for mention_id, mention_info in all_entity_mentions.items()]
    clusters = cluster_mentions(mentions_for_clustering, entity_coref.score)
    return clusters

#step 5
def cluster_propositions(all_proposition_mentions):
    import eval_predicate_coref as predicate_coref
    from clustering_common import cluster_mentions
    """
    must cluster hashable items (uses set()) - thus giving only the unique id as first element.
    only second element (head-lemma) is being used for clustering.
    """
    mentions_for_clustering = [(mention_id, mention_info["Head"]["Lemma"])
                               for mention_id, mention_info in all_proposition_mentions.items()]
    clusters = cluster_mentions(mentions_for_clustering, predicate_coref.score)
    return clusters

"""
step 6
create okr_info dict (dict with all graph info) based on entity & proposition clustering
    - when generating the EntityMention & PropositionMention objects, maintain a global mapping 
    between global_mention_id to mention-object. 
    - at ArgumentMentions generation, use global-mention-IDs as parent-mention-id. don't assign parent-id.
        The Reason- the Proposition IDs are only created in this first traverse.
    - as a second step, for each PropositionMention: 
        * traverse all ArgumentMentions, and replace parent-mention-id and parent-id through the global mapping.
        * modify the template - replace original single-sentence template by changing the single-sentence symbols to their cluster (Entity\Proposition) ID.
"""
def generate_okr_info(sentences, all_entity_mentions, all_proposition_mentions, entities, propositions):
    from constants import MentionType
    okr_info = {}   # initialize all keyword arguments here for okr object initialization
    okr_info["name"] = "default_name"   #TODO how (or whether) should we decide this?
    okr_info["sentences"] = sentences
    okr_info["ignored_indices"] = None  # TODO does this have meaning in non-annotated okr?
    okr_info["tweet_ids"] = None    # TODO necessary?

    # when generating the EntityMention & PropositionMention objects, maintain a global mapping
    # between global_mention_id (sent_id + "_" + mention_symbol) to mention-object
    global_mention_id_to_mention_object = {}

    # generate Entities
    okr_info["entities"] = {}
    for entity_id, entity in enumerate(entities, start=1):
        entity_id = "E" + str(entity_id)
        entity_mentions = {}
        for new_mention_id, (mention_global_id, _) in enumerate(entity, start=1):
            mention_info = all_entity_mentions[mention_global_id]
            mention_object = okr.EntityMention(id=new_mention_id,
                                               sentence_id=mention_info["sentence_id"],
                                               indices=mention_info["indices"],
                                               terms=mention_info["terms"],
                                               parent=entity_id)

            # add this EntityMention object to Entity.mentions dict
            entity_mentions[new_mention_id] = mention_object
            # maintain global mapping
            global_mention_id_to_mention_object[mention_global_id] = mention_object

        # add this entity as Entity object to okr.entities
        all_entity_terms = [ mention.terms for mention in entity_mentions.values() ]
        okr_info["entities"][entity_id] = okr.Entity(id=entity_id,
                                                     name=all_entity_terms[0],
                                                     mentions=entity_mentions,
                                                     terms=all_entity_terms,
                                                     entailment_graph=None)

    # generate Propositions
    okr_info["propositions"] = {}
    for prop_id, prop in enumerate(propositions, start=1):
        prop_id = "P" + str(prop_id)
        prop_mentions = {}
        for new_mention_id, (mention_global_id, _) in enumerate(prop, start=1):

            # retrieve original prop-mention information by the unique id
            mention = all_proposition_mentions[mention_global_id]

            def generate_argument_mentions():
                # if no args - return empty dict
                if not mention["Arguments"]:
                    return {}

                sentence_id = mention["sentence_id"]

                # generate ArgumentMentions
                prefix_to_arg_type={"P":MentionType.Proposition, "A":MentionType.Entity}
                argument_mentions = {}
                for arg_id, arg_symbol in enumerate(mention["Arguments"], start=1):
                    mention_type = prefix_to_arg_type[arg_symbol[0]]
                    parent_mention_id = str(sentence_id) + "_" + arg_symbol
                    # note that can't assign parent_id, because id of proposition is not yet determined. do that afterwards.
                    argument_mentions[arg_id] = okr.ArgumentMention(id=arg_id,
                                                                    desc="",
                                                                    mention_type=mention_type,
                                                                    # TODO these will be modified afterward
                                                                    parent_id=None,
                                                                    parent_mention_id=parent_mention_id)
                return argument_mentions

            argument_mentions = generate_argument_mentions()
            # generate PropositionMention object
            mention_object = okr.PropositionMention(id=new_mention_id,
                                                    sentence_id=mention["sentence_id"],
                                                    indices=mention["Bare predicate"][1],
                                                    terms=mention["Bare predicate"][0],
                                                    parent=prop_id,
                                                    argument_mentions=argument_mentions,
                                                    is_explicit=True)

            # TODO this will also be modified afterwards
            mention_object.template = mention["Template"]

            # add this PropositionMention object to Proposition.mentions dict
            prop_mentions[new_mention_id] = mention_object
            # maintain global mapping
            global_mention_id_to_mention_object[mention_global_id] = mention_object

        # add this prop as Proposition object to okr.propositions
        all_prop_terms = [ mention.terms for mention in prop_mentions.values() ]
        okr_info["propositions"][prop_id] = okr.Proposition(id=prop_id,
                                                            name=all_prop_terms[0],
                                                            mentions=prop_mentions,
                                                            attributor=None,  # TODO neccessary?
                                                            terms=all_prop_terms,
                                                            entailment_graph=None)

    # modify PropositionMention - parent_mention_id, parent_id, and template
    for prop_id, prop in okr_info["propositions"].items():
        for prop_mention_id, prop_mention in prop.mentions.items():
            # modify arguments
            for argument_mention_id, argument_mention in prop_mention.argument_mentions.items():
                mention_global_id = argument_mention.parent_mention_id
                sent_id, symbol = mention_global_id.split("_")
                # modify parent_mention_id
                arg_orig_mention_object = global_mention_id_to_mention_object[mention_global_id]
                argument_mention.parent_mention_id = arg_orig_mention_object.id
                argument_mention.parent_id = arg_orig_mention_object.parent

                # replace symbol of argument in template with id of its parent (Entity/Proposition)
                prop_mention.template = prop_mention.template.replace("{"+symbol+"}", "{"+argument_mention.parent_id+"}")

    return okr_info


# main
if __name__ == "__main__":
    # general settings
    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    args = docopt(__doc__)
    input_fn = args["--in"]
    output_fn = args["--out"]

    #automatic pipeline
    sentences = get_raw_sentences_from_file(input_fn)
    parsed_sentences = parse_single_sentences(sentences)
    all_entity_mentions, all_proposition_mentions = get_mention_lists(parsed_sentences)
    entities = cluster_entities(all_entity_mentions)
    propositions = cluster_propositions(all_proposition_mentions)
    okr_info = generate_okr_info(sentences, all_entity_mentions, all_proposition_mentions, entities, propositions)
    okr_v1 = okr.OKR(**okr_info)

    # log eventual results
    ## did we cluster any mentions?
    logging.debug("entities with more than one mention:")
    logging.debug([ entity.terms for entity in okr_v1.entities.values() if len(entity.mentions)>1 ])
    logging.debug("propositions with more than one mention:")
    logging.debug([prop.terms for prop in okr_v1.propositions.values() if len(prop.mentions) > 1])

    # export output
    import pickle, json
    pickle.dump(okr_v1, open(output_fn, "w"))

