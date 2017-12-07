
Up-To-Date results (5.12.17):
Evaluation output for the test set:

Predicate mentions(F1) = 0.309 Recall = 0.325 Precision = 0.300
Entity mentions(F1): 0.479 Recall = 0.519 Precision = 0.456
Predicate coreference: MUC=0.228, B^3=0.600, CEAF_C=0.176, MELA=0.335
Entity coreference: MUC=0.273, B^3=0.672, CEAF_C=0.362, MELA=0.436 

(23.11.17):
Evaluation output for the test set:

Predicate mentions(F1) = 0.321 Recall = 0.265 Precision = 0.418
Entity mentions(F1): 0.448 Recall = 0.420 Precision = 0.494
Predicate coreference: MUC=0.197, B^3=0.622, CEAF_C=0.180, MELA=0.333
Entity coreference: MUC=0.173, B^3=0.614, CEAF_C=0.300, MELA=0.362


-----------------------------------------------------------------------------
Evaluation output for the development set:

Predicate mentions(F1): 0.221
Entity mentions(F1): 0.259
Predicate coreference: MUC=0.062, B^3=0.571, CEAF_C=0.185, MELA=0.273
Entity coreference: MUC=0.074, B^3=0.563, CEAF_C=0.203, MELA=0.280

Evaluation output for the test set:

Predicate mentions(F1): 0.312
Entity mentions(F1): 0.277
Predicate coreference: MUC=0.159, B^3=0.601, CEAF_C=0.192, MELA=0.317
Entity coreference: MUC=0.055, B^3=0.568, CEAF_C=0.246, MELA=0.290


Evaluation output for the big events set:

Predicate mentions(F1): 0.218
Entity mentions(F1): 0.171
Predicate coreference: MUC=0.017, B^3=0.499, CEAF_C=0.147, MELA=0.221
Entity coreference: MUC=0.036, B^3=0.493, CEAF_C=0.143, MELA=0.224


-----------------------------------------------------------------------------
Changes in predicate coreference algorithm:

argument score - Using argument coreference information.
lexical score - same lemmatized dependency head

Evaluation output for the test set:

Predicate mentions(F1): 0.238 (include implicit)
Entity mentions(F1): 0.420
Predicate coreference: MUC=0.155, B^3=0.602, CEAF_C=0.203, MELA=0.320
Entity coreference: MUC=0.249, B^3=0.714, CEAF_C=0.303, MELA=0.422

------------------------------------------------------------------------------
Changes in predicate coreference algorithm:

argument score - Using argument coreference information.
lexical score - have some Wordnet overlap or pass fuzzy string matching (using all the words in the predicates)

Evaluation output for the test set:

Predicate mentions(F1): 0.238 (include implicit)
Entity mentions(F1): 0.420
Predicate coreference: MUC=0.133, B^3=0.528, CEAF_C=0.184, MELA=0.282
Entity coreference: MUC=0.249, B^3=0.714, CEAF_C=0.303, MELA=0.422

--------------------------------------------------------------------------------

Changes in predicate coreference algorithm:

argument score - Using argument coreference information.
lexical score - have some Wordnet overlap or pass fuzzy string matching (using lemmatized dependency heads of predicates)

Evaluation output for the test set:

Predicate mentions(F1): 0.238 (include implicit)
Entity mentions(F1): 0.420
Predicate coreference: MUC=0.173, B^3=0.580, CEAF_C=0.185, MELA=0.313
Entity coreference: MUC=0.249, B^3=0.714, CEAF_C=0.303, MELA=0.422