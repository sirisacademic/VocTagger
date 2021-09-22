import string, re
from collections import defaultdict
import pandas as pd
import numpy as np
import unicodedata
import pkg_resources
from .nlp_utils import utf8_normalize, remove_html_tags, clean_sentence
from spacy.tokens import Doc
from spacy.language import Language
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA,\
        ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
import pdb

class LanguageModel(object) :

    def __init__(self, lang='en', extra_stopwords=None) :
        """Class initialization."""
        self.lang = lang

        # register custom attributes to the document, in order to store properties
        Doc.set_extension("chunks", default=None, force=True)
        Doc.set_extension("chunk_roots", default=None, force=True)
        Doc.set_extension("tokens", default=None, force=True)

        # initialize the NLP tools specific for the language
        if lang == 'en' :
            import en_core_web_lg as spacy_model
            self.nlp = spacy_model.load(disable="ner")

            # british spelling normalizer
            stream = pkg_resources.resource_stream(__name__, 'data/english_spelling.txt')
            self.norm_table = \
               pd.read_csv(stream, sep="\t").\
                 drop_duplicates('american').\
                 set_index('american').uk.to_dict()

            # set default stopwords
            self.stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)

        elif lang == 'fr' :
            self.nlp = spacy.load('fr_core_news_sm', disable="ner")
            self.norm_table = {}

            # set default stopwords
            self.stopwords = list(spacy.lang.fr.stop_words.STOP_WORDS)


        # add extra stopwords
        if isinstance(extra_stopwords, list) and len(extra_stopwords):
            for w in extra_stopwords:
                self.nlp.vocab[w.lower()].is_stop = True
                self.nlp.vocab[w.title()].is_stop = True
                self.nlp.vocab[w.upper()].is_stop = True
            self.stopwords.extend(extra_stopwords)

        # first thing: add the Preprocessing component to the pipeline
        self.nlp.add_pipe("preprocesser", first=True)

        # add defaults to sentencizer
        self.nlp.add_pipe('sentencizer', after='preprocesser')

    def tokenize(self, text,
                 separate_sentences=True,
                 lemmatize_text=True,
                 killer_character='©',
                 exclude_pos = ('PUNCT', 'SPACE'),
                 stopwords=()
                ) :
        """Tokenize and lemmatize a text.
        
        Parameters:
            text (str): the text to analyse.
            (optional) separate_sentences (bool): if True, the resulting text
            will be separated into lists of lists, containing the tokens for
            each sentence. If False, the result will be a single list containing
            the tokens of the text.
            (optional) lemmatize_text (bool): If True, the tokens are lemmatised.
            (optional) killer_character (str): defaults to copyright symbol. If
            this character is detected in a sentence, eliminate the sentence.
            (optional) exclude_pos (tuple): tuple containing the list of POS
            tags that should be eliminated from the text. Defaults to ('PUNCT',
            'SPACE')
        
        Returns:
            sentences (str): a list or list of lists (depending on 
            whether `separate_sentences` is True or False - containing the tokens 
            that were extracted from the text.
        """
        # we check here if the pipeline already contains the
        # "sentence_tokenizer" component. If so, we run the pipeline and return
        # the tokens. If not, we add the component, run the pipe, remove the
        # component, and then return the tokens
        # pdb.set_trace()
        if self.nlp.has_pipe("sentence_tokenizer") :
            doc = self.nlp(text)
        else :
            # build the configuration dictionary only if the sentence_tokenizer
            # pipeline component is not already in the pipeline
            config = {"separate_sentences" : separate_sentences,
                      "lemmatize_text" : lemmatize_text,
                      "killer_character" : killer_character,
                      "exclude_pos" : exclude_pos,
                      "norm_table" : self.norm_table,
                      "stopwords" : stopwords}

            self.nlp.add_pipe("sentence_tokenizer",config=config)
            doc = self.nlp(text)
            self.nlp.remove_pipe("sentence_tokenizer")

        # return result
        return doc._.tokens

    def noun_chunks(self, text) :
        """Retrieve the noun chunks from a text.

        Parameters:
            - doc(spacy.Doc): the doc from which to extract the noun chunks

        Returns:
            - chunks(list): the list of chunks detected
        """
        doc = self.nlp(text)
        return doc_noun_chunks(doc)._.chunks

    def run_nlp(self, df,
                text_id='EID', text_column='fulltext', n_cores=4) :
        # first get a Series object from the data frame, with the text and the
        # identifier for each text
        df = df.set_index(text_id)
        texts_series = df[text_column]
        text_ids = texts_series.index.to_list()
        texts = texts_series.apply(str.lower).values

        # then launch the processing pipeline
        # pdb.set_trace()
        results = list(self.nlp.pipe(texts, n_process=n_cores, batch_size=200))

        # now reassemble IDs and texts and return
        return pd.Series({text_ids[i] : results[i] for i in range(len(texts))})

class SentenceTokenizer :
    def __init__(self, separate_sentences:bool,
                       lemmatize_text:bool,
                       killer_character:str,
                       exclude_pos:tuple,
                       norm_table:dict,
                       stopwords:list):

        # init all the class parameters
        self.separate_sentences = separate_sentences
        self.lemmatize_text = lemmatize_text
        self.killer_character = killer_character
        self.exclude_pos = exclude_pos
        self.norm_table = norm_table
        self.stopwords = stopwords
        # pdb.set_trace()
    
    def __call__(self, doc:Doc) -> Doc:
        # iterate over all the sentences in the document
        sentences = []
        for sent in doc.sents :

            # add a check on whether the `killer_character` was detected in
            # the sentence
            if self.killer_character in sent.text :
                continue

            # iterate over the words in the sentence
            this_sentence = []
            for word in sent :
                
                # check if the word is in the list of stopwords, and if so, skip
                # it
                if word.text in self.stopwords :
                    continue
                
                # check that the POS tag of the word is not
                # included in the POS tags that we want to exclude
                if not word.pos_ in self.exclude_pos :

                    # now extract the word in its requested form: if
                    # `lemmatize_text` is true, then we extract the lemma.
                    # Otherwise it is the plain text
                    if self.lemmatize_text:
                        clean_word = word.lemma_
                    else :
                        clean_word = word.text

                    # cast to lowercase
                    clean_word = clean_word.lower()

                    # finally, we apply spelling rules
                    if self.norm_table is not None :
                        clean_word = self.norm_table.get(clean_word, clean_word)

                    # now behave differently if we want the sentences to be
                    # separated or not
                    if self.separate_sentences :
                        this_sentence.append(clean_word)
                    else :
                        sentences.append(clean_word)

            # at the end of the iteration over a single sentence, check
            # again if we want the sentences to be separated
            if self.separate_sentences :

                # and also check that the sentence is not empty
                if len(this_sentence) > 0 :
                    sentences.append(this_sentence)

        # set value of the custom attribute `tokens` to store the result
        doc._.tokens = sentences

        # return the doc, so that it can be used in multiprocessing
        return doc

class Preprocesser :
    def __init__(self, nlp: Language) :
        self.nlp = nlp

    def __call__(self, doc: Doc) -> Doc:
        # here we extract the text of the doc
        text = doc.text

        # cast to lowercase every word, if the word is not completely uppercase
        working_text = ' '.join([\
                w.lower() if not (w.isupper()) else w \
            for w in text.split()\
            ])

        # here we create a doc that is based around the new text that we created
        return self.nlp.make_doc(working_text)

@Language.factory("preprocesser")
def make_preprocesser(nlp, name) :
    return Preprocesser(nlp)

@Language.factory("sentence_tokenizer",
                  default_config={"separate_sentences":True,
                                  "lemmatize_text":True,
                                  "killer_character":'©',
                                  "exclude_pos":('PUNCT', 'SPACE'),
                                  "norm_table":{},
                                  "stopwords":()})
def create_sentence_tokenizer(nlp, name,
                       separate_sentences:bool,
                       lemmatize_text:bool,
                       killer_character:str,
                       exclude_pos:tuple,
                       norm_table:dict,
                       stopwords:list):
    return SentenceTokenizer(separate_sentences,
                             lemmatize_text, killer_character, exclude_pos,
                             norm_table, stopwords)

@Language.component("noun_chunker")
def doc_noun_chunks(doc) :
    """Helper function to extract noun chunks. Used in NLP pipelines.
    """
    # init the output structure
    chunks = []
    chunk_roots = []
    

    # iterate over all the chunks that SpaCy identifies. Remember that in
    # the noun chunks there will also be punctuation, articles, pronouns,
    # etc.
    for chunk in doc.noun_chunks:

        # init the chunk
        this_noun_chunk = []
        nostop_chunk_count = 0
        
        # iterate over tokens in the chunk
        for token in chunk :

            # check that the POS tag of the token is not useless
            if token.pos_ not in ('PUNCT', 'SPACE', 'DET', 'PRON', 'ADV',
                                  'CCONJ') :
                # this_noun_chunk.append((token, token.tag_, token.pos_))
                this_noun_chunk.append(token.lemma_.lower())
                # track stopwords
                if not token.is_stop :
                    nostop_chunk_count += 1

        # some chunks will be left empty. If not, add them to the output
        # data structure. Also, check whether the number of non-stopword tokens
        # in the chunk is more than 50% of the total number of words in the
        # chunk
        if len(this_noun_chunk) > 0 and \
           nostop_chunk_count >= len(this_noun_chunk)/2:

            ## add the chunk
            chunks.append('_'.join (this_noun_chunk))

            # if len (this_noun_chunk) > 1 :

            ## and also add the chunk: chunk_root as chunk_root
            ## this can be used to retain only the most frequent
            ## chunks and yield the root for the less frequent ones
            chunk_roots.append ( {'_'.join(this_noun_chunk) : chunk.root.lemma_.lower() })
            # chunks.append(chunk.root.lemma_.lower())

    # now add the chunks to our custom attribute of the document. This is the
    # way to be able to add this function to a processing pipeline
    doc._.chunks = chunks
    doc._.chunk_roots = chunk_roots
    
    return doc
