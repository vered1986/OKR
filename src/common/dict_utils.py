

def clean_dict(dictionary, desired_attributes):
    """ Remove all attributes of dictionary beside those in desired_attributes. """
    remove_list = (att for att in dictionary.keys() if att not in desired_attributes)
    for att in remove_list:
        del dictionary[att]


def rename_attribute(dictionary, old_key_name, new_key_name):
    dictionary[new_key_name] = dictionary.pop(old_key_name)

