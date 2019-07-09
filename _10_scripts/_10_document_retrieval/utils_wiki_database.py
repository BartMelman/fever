import unicodedata

def normalise_text(text):
    text = text.replace("_", " ")
    text = text.replace("-LRB-","(",)\
    .replace("-RRB-", ")")\
    .replace("-COLON-",":")\
    .replace("-LSB-", "[")\
    .replace("-RSB-", "]")
    return unicodedata.normalize('NFD', text)