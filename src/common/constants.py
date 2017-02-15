import stop_words

NULL_VALUE = 0
STOP_WORDS = stop_words.get_stop_words('en')


class MentionType:
    """
    Enum for mention type
    """
    Entity = 0
    Proposition = 1
