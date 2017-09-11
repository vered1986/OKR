# OKR: A Consolidated <b>O</b>pen <b>K</b>nowledge <b>R</b>epresentation for Multiple Texts

This is the code used in the paper:

<b>"A Consolidated Open Knowledge Representation for Multiple Texts"</b><br/>
Rachel Wities, Vered Shwartz, Gabriel Stanovsky, Meni Adler, Ori Shapira, Shyam Upadhyay, Dan Roth, Eugenio Martinez Camara, Iryna Gurevych and Ido Dagan. LSDSem 2017. [link](???) (TBD).

The dataset developed for the paper can be found [here](http://u.cs.biu.ac.il/~nlp/resources/downloads/twitter-events/) (TBD).

***

<b>Prerequisites:</b>
* Python 2.7
* numpy
* [bsddb](https://docs.python.org/2/library/bsddb.html)
* spacy
* stop-words

<b>Quick Start:</b>

The repository contains the following directories:
* src - the source files - used to load the OKR graph (common), compute inter-annotator agreement (agreement), and automatically construct the OKR object (baseline_system).
* resources - used by the baseline system.
* data - the annotation files used to compute the inter-annotator agreement (agreement) and the development and test set used in the baseline system (baseline).

## Running the baseline system:

From src/baseline_system: `python compute_baseline_subtasks.py  ../../data/baseline/dev ../../data/baseline/test`

In the entity mentions components, the F1 score we originaly reoprted was 0.58. We managed to raise it to 0.61 by changing spacy tokenization. If you want the original code that returns the original 0.58 score, set GET_ORIGINAL_SCORE to True in line 22 in eval_entity_mention.py.

The entailment component requires resources. The entity entailment resource files are found in the resources directory. The predicate entailment file is much larger, and we therefore provide the [script](resources/create_predicate_entailment_resource.py) to build it from the original resource (reverb_local_clsf_all.txt from [here](http://u.cs.biu.ac.il/~nlp/resources/downloads/predicative-entailment-rules-learned-using-local-and-global-algorithms/)).

## Detailed description of the OKR object:
TBD
