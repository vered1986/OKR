import bsddb
import codecs

from docopt import docopt


def main():
    """
    Converts the Berant et al. resource to a bsddb dictionary
    """
    args = docopt("""Converts the Berant et al. resource to a bsddb dictionary

    Usage:
        create_predicate_entailment_resource.py <in_res_file> <out_res_file>

        <in_res_file> = the input resource file (reverb_local_clsf_all.txt)
        <out_res_file> = the output bsddb file
    """)

    in_res_file = args['<in_res_file>']
    out_res_file = args['<out_res_file>']

    entailment_rules = bsddb.btopen(out_res_file, 'c')

    for (lhs, rhs, score) in load_resource(in_res_file):
        entailment_rules[lhs + '###' + rhs] = str(score)

    entailment_rules.sync()


def load_resource(res_file):
    """
    Loads the Berant et al. resource and returns a list of rules (lhs, rhs, score)
    :param res_file the resource file
    :return a list of rules (lhs, rhs, score)
    """
    rules = []
    with open(res_file) as f_in:
        for line in f_in:
            lhs, rhs, score = line.strip().split('\t')
            lhs, rhs = format_predicate(lhs), format_predicate(rhs)
            rules.append((lhs, rhs, score))

    return rules


def format_predicate(pred):
    """
    Receives a predicate in the original format (from the Berant et al. resource)
    and returns a format suitable for the entailment component in the baseline system.
    :param pred the predicate in the original format
    :return the predicate in a format suitable for the entailment component in the baseline system
    """
    is_reversed = False

    if pred.endswith('@R'):
        pred = rule[:-2]
        is_reversed = True

    first_arg = 'X' if not is_reversed else 'Y'
    second_arg = 'Y' if not is_reversed else 'X'
    pred = first_arg + ' ' + pred + ' ' + second_arg

    return pred


if __name__ == '__main__':
    main()