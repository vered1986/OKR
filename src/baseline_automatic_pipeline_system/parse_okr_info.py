"""Usage:
   parse_okr_info --in=INPUT_FILE --out=OUTPUT_FILE

Author: Ayal Klein

baseline for automatic pipeline for okr-v2.
run it from base OKR directory.

steps:
1. get list of sentences (tweets).
2. parse each sentence to get single-sentence semantic representation
3. combine: create combined lists of EntityMentions and of PropositionMentions (=predicate template is a proposition-mention)
4. use baseline coref system to cluster EntityMentions to Entities
5. use baseline coref system to cluster PropositionMentions to Propositions
6. create okr object based on the pipeline results

"""

import sys, logging, copy, os
from docopt import docopt
from collections import defaultdict
# add all src sub-directories to path
for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

from parsers.props_wrapper import PropSWrapper
from dict_utils import rename_attribute
import okr

props_wrapper = PropSWrapper(get_implicits=False,
                  get_zero_args=False,
                  get_conj=False)

def rename_attribute(dictionary, old_key_name, new_key_name):
    dictionary[new_key_name] = dictionary.pop(old_key_name)

# step 1 - meantime suppose its just a simple text file of line-separated sentences
def get_raw_sentences_from_file(input_fn):
    """ Get dict of { sentence_id : raw_sentence } out of file. ignore comments """
    raw_sentences = [ line.strip()
                      for line in open(input_fn, "r")
                      if line.strip() and not line.startswith('#') and not line.strip().isdigit() and not line.startswith('<name>')]

    # make it a dict {id : sentence}, id starting from 1
    sentences_dict = {str(sent_id) : sentence for sent_id, sentence in enumerate(raw_sentences, start=1)}
    return sentences_dict

# step 2
def parse_single_sentences(raw_sentences):
    """ Return a dict of { sentence_id : parsed single-sentence representation } . uses props as parser.
    @:arg raw_sentences: a { sentence_id : raw_sentence } dict. 
    """
    parsed_sentences = {}
    for sent_id, sent in raw_sentences.iteritems():
        try:
            props_wrapper.parse(sent)
            parsed_sentences[sent_id] = props_wrapper.get_okr()
        except Exception as e:
            logging.error("failed to parse sentence: " + sent)

    return parsed_sentences

#step 3
def get_mention_lists(parsed_sentences):
    """ Return combined lists of EntityMentions and of PropositionMentions (of all sentences). """

    all_entity_mentions = {}        # would be: { global-unique-id : entity-mention-info-dict }
    all_proposition_mentions = {}   # would be: { global-unique-id : proposition-mention-info-dict }
                                    # global-unique-id is composed from sent_id + "_" + key(=id-symbol at props_wrapper)

    for sent_id, parsed_sent in parsed_sentences.iteritems():
        # *** entities ***
        sentence_entity_mentions = { sent_id + "_" + key :
                                         { "terms":          unicode(terms),
                                           "indices":       indices,
                                           "sentence_id":   sent_id }
                                     for key, (terms, indices) in parsed_sent["Entities"].iteritems() }
        all_entity_mentions.update(sentence_entity_mentions)

        # *** propositions ***
        sentence_proposition_mentions = { sent_id + "_" + key :
                                              dict( {"sentence_id" : sent_id},
                                                    **prop_ment_info )
                                          for key, prop_ment_info in parsed_sent["Predicates"].iteritems() }
        all_proposition_mentions.update(sentence_proposition_mentions)

    return all_entity_mentions, all_proposition_mentions

# step 4
def cluster_entities(all_entity_mentions):
    """ Cluster entity-mentions to entities, Using baseline coreference functions.  """
    import eval_entity_coref as entity_coref
    from clustering_common import cluster_mentions
    """
    must cluster hashable items (uses set()) - thus giving only the unique id as first element.
    only second element (terms) is being used for clustering.
    """
    mentions_for_clustering = [ (mention_id, mention_info["terms"])
                                for mention_id, mention_info in all_entity_mentions.iteritems()]
    clusters = cluster_mentions(mentions_for_clustering, entity_coref.score)
    return clusters

#step 5
def cluster_propositions(all_proposition_mentions, all_entity_mentions):
    """ Cluster proposition-mentions to propositions, Using baseline coreference functions.  """
    """
    for clustering, each mention is a (mention-unique-id, mention-head-lemma, mention-full-info) tuple.
    mention-head-lemma is a string, and is given for backward compatibitly. 
    mention-full-info is a dict containing all the info about the proposition-mention as given by props_wrapper,
    with a modification of the "Arguments" field, which would be a dict mapping template-symbols (e.g. "A1" or "P2") 
    to their mention records.
    """
    # get all entities and propositions into all_concepts
    all_concepts = all_proposition_mentions.copy()
    all_concepts.update(all_entity_mentions)
    # prepare mentions for clustering
    mentions_for_clustering = []
    for mention_id, mention_info in all_proposition_mentions.iteritems():
        head_lemma = mention_info["Head"]["Lemma"]
        """
        get all relevant concepts - all concepts which their symbol's prefix is the sentence-id.
        This is because the template is using the single-sentence symbol - Ai for entities ans Pi for propositions.
        The mention id is of the form <sentence-id>_<single-sentence-symbol>
        """
        sentence_id = mention_info["sentence_id"]
        relevent_concepts = { m_id.split("_")[1] : m_info
                              for m_id, m_info in all_concepts.iteritems()
                              if m_id.split("_")[0] == sentence_id }
        # construct a mapping between template-symbols (e.g. "A1" or "P2") to their mention records
        arguments_map = { arg : relevent_concepts[arg] for arg in mention_info["Arguments"] }
        mention_info.update({"Arguments": arguments_map})
        mentions_for_clustering.append( (mention_id, head_lemma, mention_info) )

    # cluster the list of mentions to coreference chains
    import eval_predicate_coref as predicate_coref
    from clustering_common import cluster_mentions
    clusters = cluster_mentions(mentions_for_clustering, predicate_coref.score)
    return clusters

# helper function for step 6
def generate_argument_mentions(prop_mention):
    """ Generate argument_mentions dict { arg_id : ArgumentMention} for a PropositionMention. """
    from constants import MentionType
    # if no args - return empty dict
    if not prop_mention["Arguments"]:
        return {}

    sentence_id = prop_mention["sentence_id"]

    # generate ArgumentMentions
    prefix_to_arg_type = {"P": MentionType.Proposition, "A": MentionType.Entity}
    argument_mentions = {}
    for arg_id, arg_symbol in enumerate(prop_mention["Arguments"], start=1):
        mention_type = prefix_to_arg_type[arg_symbol[0]]
        parent_mention_id = str(sentence_id) + "_" + arg_symbol
        # note that can't assign parent_id, because id of proposition is not yet determined. do that afterwards.
        argument_mentions[arg_id] = okr.ArgumentMention(id=arg_id,
                                                        desc="",
                                                        mention_type=mention_type,
                                                        # these will be modified afterward
                                                        parent_id=None,
                                                        parent_mention_id=parent_mention_id)
    return argument_mentions

#step 6
def generate_okr_info(sentences, all_entity_mentions, all_proposition_mentions, entities, propositions):
    """ Create okr_info dict (dict with all graph info) based on entity & proposition clustering.
    1. when generating the EntityMention & PropositionMention objects, maintain a global mapping 
    between global_mention_id to mention-object. 
    2. at ArgumentMentions generation, use global-mention-IDs as parent-mention-id. don't assign parent-id.
        The Reason- the Proposition IDs are only created in this first traverse.
    3. as a following step, for each PropositionMention: 
        * traverse all ArgumentMentions, and replace parent-mention-id and parent-id through the global mapping.
        * modify the template - replace original single-sentence template by changing the single-sentence symbols to their cluster (Entity\Proposition) ID.
    4. Argument Alignment - the alignment of arguments of different PropositionMentions (of same Proposition) should
        be eventually expressed by their arg-id. However, In the generation of the argument mentions, such alignment 
        does not exist; the alignment can be done only after each ArgumentMention is mapped to a concept (step 3. above).
        Therefore, a subsequent step is required, in which:
            4.a. first, do Argument Alignment - decide on Proposition-level argument-slots, that refer to a same role 
                relative to the predicate. Map each argument-mention to an argument-slot.  
            4.b. After that, iterate all ArgumentMentions and change their arg-id to their argument-slot symbol. 
    """
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
        entity_id = "E." + str(entity_id)
        entity_mentions = {}
        for new_mention_id, (mention_global_id, _) in enumerate(entity, start=1):
            mention_info = all_entity_mentions[mention_global_id]
            mention_object = okr.EntityMention(id=new_mention_id,
                                               sentence_id=mention_info["sentence_id"],
                                               indices=list(mention_info["indices"]),
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
        prop_id = "P." + str(prop_id)
        prop_mentions = {}
        for new_mention_id, (mention_global_id, _,_) in enumerate(prop, start=1):

            # retrieve original prop-mention information by the unique id
            mention = all_proposition_mentions[mention_global_id]

            # generate PropositionMention object
            bare_predicate = mention["Bare predicate"]
            mention_object = okr.PropositionMention(id=new_mention_id,
                                                    sentence_id=mention["sentence_id"],
                                                    indices=list(bare_predicate[1]),
                                                    terms=bare_predicate[0],
                                                    parent=prop_id,
                                                    argument_mentions=generate_argument_mentions(mention),
                                                    is_explicit=not bare_predicate == PropSWrapper.IMPLICIT_SYMBOL)

            # this will also be modified afterwards
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
                                                            attributor=None,  # TODO extract
                                                            terms=all_prop_terms,
                                                            entailment_graph=None)

    # 3. modify PropositionMention - parent_mention_id, parent_id, and template
    for prop_id, prop in okr_info["propositions"].iteritems():
        for prop_mention_id, prop_mention in prop.mentions.iteritems():
            # modify arguments
            for argument_mention_id, argument_mention in prop_mention.argument_mentions.iteritems():
                mention_global_id = argument_mention.parent_mention_id
                # modify parent_mention_id
                arg_orig_mention_object = global_mention_id_to_mention_object[mention_global_id]
                argument_mention.parent_mention_id = arg_orig_mention_object.id
                argument_mention.parent_id = arg_orig_mention_object.parent
                # replace symbol of argument in template with id of its parent (Entity/Proposition)
                sent_id, symbol = mention_global_id.split("_")
                prop_mention.template = prop_mention.template.replace("{"+symbol+"}", "{"+argument_mention.parent_id+"}")

    # 4. Argument Alignment step
    for prop_id, prop in okr_info["propositions"].iteritems():
        argument_slots_grouping = argument_alignment(prop.mentions)
        replace_arg_mentions_to_arg_slots(prop.mentions, argument_slots_grouping)

    return okr_info

def replace_arg_mentions_to_arg_slots(mentions, slots_grouping):
    """ Transform arg-identifiers (in arg-dicts and templates) to use aligned arguments (slots). 
    The new arguments symbols ("slots") are representing a semantic role relative to the predicate. 
    one slot may refer to multiple concepts across various proposition-mentions. 
    The mapping process between original argument values (which are concepts - known Entity\Proposition)
    to the new "consolidated" arguments slots is called Argument Alignment. 
    :param mentions: a dict { prop_mention_id : PropositionMention } of the Proposition
    :param slots_grouping: list of lists, a grouping of (prop-mention-id, arg-mention-id) to arguments slots. 
    """
    # mapping from (prop_mention_id, arg_mention_id) to a symbol of (proposition-level) argument-slot
    ids_to_slot = { arg_ids : str(slot_index)
                    for slot_index, slot_group in enumerate(slots_grouping, start=1)
                    for arg_ids in slot_group }

    for prop_mention_id, prop_mention in mentions.iteritems():
        modified_template = prop_mention.template
        # iterate all args and replace old symbol (concept_id) with new symbol (slot)
        for arg_id, arg in prop_mention.argument_mentions.iteritems():
            concept_id = arg.parent_id
            slot = ids_to_slot[(prop_mention_id, arg_id)]
            # modify template to use argument-slot instead of concept-id
            modified_template = modified_template.replace("{"+concept_id+"}", "{"+slot+"}")
            # remove traces of pseudo-concept-id, which are there only for handling duplication
            arg.parent_id = concept_id.split("_")[0]
            # change arg-id to argument-slot
            arg.id = slot
        # replace arg_dict from {original-arg-id : arg} to {argument-slot : arg}
        new_arg_dict = {arg.id : arg for arg in prop_mention.argument_mentions.values()}
        prop_mention.argument_mentions = new_arg_dict

        # TODO decide how to replace the template args symbols to slots. verify mds-output regarding.
        # replace existing template (that uses concept-ids) with modified template (that uses argument-slots)
        prop_mention.template = modified_template


def argument_alignment(prop_mentions):
    """
    Group the arguments-mentions of a Proposition, to different argument-slots. 
    different argument-slots are representing different semantic roles relative to the predicate. 
    :return: list of lists, a grouping of (prop-mention-id, arg-mention-id) to arguments slots. 
    """

    # Currently, the argument alignment is simply grouping the argument referring to same concept.
    concepts = set( [arg_mention.parent_id
                     for prop_mention in prop_mentions.values()
                     for arg_mention in prop_mention.argument_mentions.values() ] )

    concept_to_slot_index = { concept : index for index, concept in enumerate(concepts) }
    number_of_slots = len(concept_to_slot_index)
    grouping = defaultdict(list)
    for prop_mention_id, prop_mention in prop_mentions.iteritems():
        # save the concepts encountered in this prop-mention (to find duplications)
        encountered_concepts = []
        for arg_mention_id, arg_mention in prop_mention.argument_mentions.iteritems():
            referred_concept = arg_mention.parent_id
            """
            Special treatment for cases where a template is containing two arguments referring to the same concept.
            This case, both arg-mentions will be aligned to same argument-slot. This is Problematic (and forbidden),
            since by definition, different argument-slots refer to different semantic roles, so an argument-slot cannot
            occur twice in a template.
            """
            if referred_concept in encountered_concepts:
                # duplication - special (new) slot necessary for arg
                grouping[number_of_slots].append( (prop_mention_id, arg_mention_id) )
                number_of_slots += 1    # increment number_of_slots to index a new slot
                """
                change the argument's concept-id at the mention's template - 
                to differentiate between the occurrences of the same concept-id in the template, we must
                change the concept-id used for the arg to an "pseudo-id". This way, we could map the args 
                to different slots.
                """
                pseudo_id = referred_concept + "_a." + str(number_of_slots)
                prop_mention.template = prop_mention.template.replace("{"+referred_concept+"}",
                                                                            "{"+pseudo_id+"}",
                                                                            1)  # replace only first occurrence
                arg_mention.parent_id = pseudo_id

                logging.info("duplication handled: mention {} of prop {}, concept repeating is {}".format(
                             prop_mention_id, prop_mention.parent, referred_concept))
            else:
                # no duplications - append this arg-mention to the slot corresponding to referred concept
                encountered_concepts.append(referred_concept)
                slot_index = concept_to_slot_index[referred_concept]
                grouping[slot_index].append( (prop_mention_id, arg_mention_id) )

    return grouping.values()


# all together (after parsing input files)
def auto_pipeline_okr_info(sentences):
    """ Get okr_info dictionary from raw-sentences.
    :param sentences: a dict of { sentence_id : raw_sentence }
    :return: okr_info, dictionary for initializing okr graphs.
    okr_v1 can be initialized using:
        okr.OKR(**okr_info)
    """
    parsed_sentences = parse_single_sentences(sentences)
    all_entity_mentions, all_proposition_mentions = get_mention_lists(parsed_sentences)
    entities = cluster_entities(all_entity_mentions)
    propositions = cluster_propositions(all_proposition_mentions, all_entity_mentions)
    okr_info = generate_okr_info(sentences, all_entity_mentions, all_proposition_mentions, entities, propositions)
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

    """
    The following section is equivalent to:
        okr_info = auto_pipeline_okr_info(sentences)
    We are using the full pipeline explicitly, for debug purposes.
    """
    parsed_sentences = parse_single_sentences(sentences)
    all_entity_mentions, all_proposition_mentions = get_mention_lists(parsed_sentences)
    # coreference
    entities = cluster_entities(all_entity_mentions)
    propositions = cluster_propositions(all_proposition_mentions, all_entity_mentions)
    # consolidating info
    okr_info = generate_okr_info(sentences, all_entity_mentions, all_proposition_mentions, entities, propositions)

    # using copy because OKR CTor changes the template of PropositionMentions of propositions attribute
    okr_v1 = okr.OKR(**copy.deepcopy(okr_info))

    # log eventual results
    ## did we cluster any mentions?
    logging.debug("entities with more than one mention:")
    logging.debug([ entity.terms for entity in okr_v1.entities.values() if len(entity.mentions)>1 ])
    logging.debug("propositions with more than one mention:")
    logging.debug([prop.terms for prop in okr_v1.propositions.values() if len(prop.mentions) > 1])

    # export output
    import pickle, json
    pickle.dump(okr_v1, open(output_fn, "w"))

