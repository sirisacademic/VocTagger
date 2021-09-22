import nltk
import string
import unicodedata
import re
import pandas as pd
import pkgutil

def utf8_normalize(text) :
    """Normalizes a text to UTF-8 encoding, while bringing it to lowercase.

    Parameters:
        - text(str): the text to encode

    Returns:
        - cleantext(str): the encoded text
    """
    # brings text to lowercase, casts to UTF-8 encoding
    return unicodedata.normalize('NFKD', text.lower()).encode('ASCII', 'ignore').decode("utf-8")

def remove_html_tags(text) :
    """Removes HTML tags from a text.

    Parameters:
        - text(str): the text with HTML tags

    Returns:
        - cleantext(str): the text without HTML tags
    """
    # clean html tags and html symbols,i.e. &quot, &acute,...
    cleanr = re.compile('<.*?>|\"|\'\'')
    return re.sub(cleanr, '', text)

# get a set of regular expression that will allow us to clean up sentences from
# parentheses, commas, quotes, and the like
content = pkgutil.get_data(__name__, 'data/pattern_char_sub.txt').decode()
content = [l.strip() for l in content]
pattern_begin = content[0].split("\t")[1:]
pattern_end = content[1].split("\t")[1:]
pattern_middle = content[2].split("\t")[1:]
pattern_begin = r"(^("+"|".join(["("+re.escape(t)+")" for t in pattern_begin]) +"))"
pattern_end = r"(("+"|".join(["("+re.escape(t)+")" for t in pattern_end]) +")$)"
pattern_middle  = r"("+"|".join(["("+re.escape(t)+")" for t in pattern_middle]) +")"
pattern_chars_sub = re.compile(r"%s|%s|%s" %(pattern_begin, pattern_end, pattern_middle))
pat_space = re.compile(r'(\w+\.\\t.\w+)|(\w+\\t\w+)|(\w+.\\n\w+)|(\w+\\n\w+)|( , )')

def clean_sentence(sentence) :
    """Cleans a sentence from parentheses, quotation marks, punctuations, etc.

    Parameters:
        - sentence(list): list of strings that contains the tokens in a sentence

    Returns:
        - new_sentence(list): a list of strings that contains the cleaned up
        sentence
    """
    new_sentence=[]
    for w in sentence:
        w = pattern_chars_sub.sub("",w)
        w = pat_space.sub(" ",w)
        w = w.replace("- ","-")
        if len(w)>0:
            new_sentence.append(w)
    return new_sentence

def bag_of_words(taggedSents, punctuation=string.punctuation):
    wordPath = ""
    chunkPath = ""

    # for each sentence
    for s in taggedSents:
        # get rid of punctuation
        s1 = [(a, b) for a, b in s if not (a in punctuation)]

        # add edge between consecutive words
        for i, w1 in enumerate (s1):
            if w1[1]=="NP" and len(w1[0].split(" "))>1:

                wordPath+=w1[0].replace(" ","-")+" "
            else:
                wordPath+=w1[0]+" "
    return wordPath
