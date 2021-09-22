from .nlp_utils import utf8_normalize, remove_html_tags, clean_sentence
from .languageModels import LanguageModel

import sys, os, re
import pandas as pd
import collections
import numpy as np
import itertools
from collections import defaultdict
from spacy.tokens import Doc
from spacy.language import Language
import pdb



def isHit(idx, threshold, interword_threshold) :
    """
    this function evaluates whether a given set of places of occurrence is a hit
    of a particular keyword. Here, `idx` can either be a single-columned object,
    or a multi-column object. In the first case it is certainly a hit. In the second
    case, harness the `itertools.product` function to obtain the tuples containing
    the positions of the occurrences of the keyword in the text. The mutual difference
    between these positions is then calculated and tested against the given threshold.
    The mutual distance has to be smaller than a value that corresponds to each
    word that composes the keyword to be separated from the next with threshold
    words in between.
    """
    # l is the length of the composite keyword. The threshold represents the
    # maximum distance at which two words that compose the keyword may be
    # located.
    l = len(idx.values)
    if l==1 :
        return True
    else :
        max_dist = (l-1) * threshold
        for c in itertools.product(*list(idx.values)) :
            
            if interword_threshold:
                c = np.diff(np.sort(c))
                
                is_hit = max(c) <= threshold
                
            else:
                m, M = min(c), max(c)

                is_hit = (M-m) <= max_dist

            if is_hit :
                return True
    return False

def invertVoc(x, voc_to_ID, tokenizedVocabulary) :
    for word in tokenizedVocabulary[x.keyword] :
        voc_to_ID[word].append(x.name)
        
def makeD(sentence, vocabulary) :
    d = defaultdict(list)
    for w, word in enumerate(sentence) :
        if word in vocabulary :
            d[word].append(w)
    return pd.Series(d)

class VocabularyTagger :
    def __init__(self, nlp:Language,
                 json_voc:str, json_tokenized_voc:str,
                 threshold:int, sentence_extras:bool,
                 interword_threshold:bool
                ) :
        # set internal parameters
        self.threshold = threshold
        self.sentence_extras = sentence_extras
        self.interword_threshold = interword_threshold

        # transform back the json objects from string to pandas DataFrame
        self.Voc = pd.read_json(json_voc, dtype={'ID':str}).set_index('ID')
        self.tokenizedVocabulary = \
                pd.read_json(json_tokenized_voc, orient='index').\
                stack().reset_index().groupby('level_0')[0].apply(list)

        # this object contains the unique set of tokens that appear in the
        # vocabulary
        self.vocabulary = tuple(set([element for list_ in self.tokenizedVocabulary
                                       for element in list_]))
        
        # get the list of IDs that are associated to each of the words in this
        # vocabulary
        self.voc_to_ID = defaultdict(list)
        useless = self.Voc.apply(invertVoc, axis=1, args=(self.voc_to_ID,
                                                     self.tokenizedVocabulary))


    def __call__(self, doc:Doc) -> Doc:
        sentences = doc._.tokens
        text = doc.text

        # cycle on the sentences
        tags = []
        for sentence, sentence_text in zip(sentences, doc.sents) :
            # make D: a vocabulary that contains the positions in the sentence
            # of the tokens in the vocabulary that are found in the current
            # sentence
            D = makeD(sentence, self.vocabulary)

            # get the list of detected words
            detectedWords = D.keys()

            # now cycle on all the possible keywords of the vocabulary that
            # contain that particular token
            IDs = []
            for word in detectedWords :
                IDs.extend(self.voc_to_ID[word])

            # examine all the candidate IDs
            for ID in set(IDs) :

                # for the current candidate, get the full keyword that contains
                # the token
                topic = self.Voc.loc[ID].keyword
                tokens = set(self.tokenizedVocabulary[topic])

                # if the detected words don't contain all the tokens, skip
                if not tokens.issubset(detectedWords) :
                    continue

                # finally, we should evaluate whether we have a potential hit
                if isHit(D[tokens], self.threshold, self.interword_threshold) :

                    # if we have a potential hit, go fetch the extra keywords
                    extras = self.Voc.loc[ID].extra

                    # sometimes the vocabulary contains a 'NaN' in the `extras`
                    # field, or the extras field is empty
                    if pd.isna(extras) or extras=='':
                        tags.append(ID)
                        continue

                    # if we're here, extras is not empty, and we search for the
                    # extras by using a regular expression. First, the regular
                    # expression must be compiled
                    extras_regex = re.compile(r'\b(%s)\b'%('|'.\
                            join(re.escape(kw) for kw in extras.split('|'))), re.I)
                    
                    if self.sentence_extras:
                        if extras_regex.search(sentence_text.text) is not None :
                            tags.append(ID)
                    else:
                        if extras_regex.search(text) is not None :
                            tags.append(ID)
        doc._.tags = list(set(tags))
        return doc

# factory for the NLP component to give to spaCy
@Language.factory("vocabulary_tagger", default_config={
    "json_voc" : "{}",
    "json_tokenized_voc" : "{}",
    "threshold" : 3,
    "sentence_extras": True,
    "interword_threshold": False
})
def create_vocabulary_tagger(nlp: Language, name:str,
                             json_voc:str,
                             json_tokenized_voc:str,
                             threshold:int,
                             sentence_extras:bool,
                             interword_threshold:bool
                            ):
    return VocabularyTagger(nlp, json_voc, json_tokenized_voc,
                            threshold, sentence_extras,
                            interword_threshold
                           )

class VocTagger(LanguageModel) :
    """VocTagger is the main class of the vocTagger package. It is
    initialized with a vocabulary of keywords and can then perform
    tagging of texts using an NLP pipeline.
    
    Methods:
        tagText: tags a single text
        tagTextCollection: tags a collection of texts with multiprocessing
    """
    def __init__(self, Voc, lang = 'en', threshold = 3, 
                 lemmatize_vocabulary = True, lemmatize_text = True, sentence_extras = True,
                 interword_threshold = False
                ) :
        """Initialize the tagger with a vocabulary and a language specified.
        
        Parameters:
            Voc (pd.DataFrame): the vocabulary that will be used to do the
            tagging. Must contain the following fields: `ID`, `keyword`, and
            `extras`.
            (optional) lang (str): two-letter code for language. Default: 'en'.
            Currently available languages are English('en') and French('fr').
            (optional) lemmatize_text (bool): controls whether the text tokens
            are lemmatised. Defaults to True.
            (optional) threshold (int): threshold for the maximum distance
            between two given words in a multi-word keyword. Defaults to 3.
            (optional) lemmatize_vocabulary (bool): controls whether the
            vocabulary tokens should be lemmatised. Defaults to True.
            (optional) sentence_extras: controls whether the extras are searched
            within the same sentence as the focal keyword. Defaults to True.
            (optional) interword_threshold: controls whether the threshold is
            applied between individual words in multiword keywords. Defaults to False.
        """
        # super class init
        super().__init__(lang=lang)

        # keep track of the vocabulary internally
        self.Voc = Voc.astype({'ID':str})
        
        # register a custom attribute to the document, in order to store properties
        Doc.set_extension("tags", default=None, force=True)

        # convert the vocabulary to JSON to comply with spaCy's idiosyncracies
        json_voc = self.Voc.to_json()
        
        # deal with stopwords here. Start with a generic list of stopwords that
        # belong to the language, but remove those stopwords that are found in
        # the vocabulary.
        my_stopwords = self.stopwords
        unique_voc_tokens = set(Voc.keyword.apply(lambda _ : _.split()).\
                apply(pd.Series).stack().unique())
        my_stopwords = list(set(self.stopwords) - unique_voc_tokens)
    
        # tokenize and stem the keywords of the vocabulary
        self.tokenized_voc = \
                Voc.drop_duplicates('keyword').set_index('keyword').\
                index.to_series().apply(self._tokenizeVocabulary,\
                                       args=(lemmatize_vocabulary, ))

        # add the sentence tokenizer as a component of the pipeline
        config = {
            "lemmatize_text" : lemmatize_text,
             "norm_table" : self.norm_table,
             "stopwords" : my_stopwords
        }
        self.nlp.add_pipe("sentence_tokenizer",config=config, last=True)
 
        # add the components of the processing pipeline to the NLP of the model
        config = {
            "json_voc" : json_voc,
            "json_tokenized_voc" : self.tokenized_voc.to_json(),
            "threshold" : threshold,
            "sentence_extras": sentence_extras,
            "interword_threshold": interword_threshold
        }
        self.nlp.add_pipe("vocabulary_tagger", config=config, last=True)

    def _tokenizeVocabulary(self, keyword, lemmatize_vocabulary) :
        # first step is to tokenize each of the keywords of the vocabulary,
        # lemmatising or not the keywords according to the option
        # `lemmatize_vocabulary`.
        tv = self.tokenize(keyword,
                             separate_sentences=False,
                             lemmatize_text=lemmatize_vocabulary,
                             exclude_pos=())
        return tv

    def tagText(self, text) :
        """Tags a single text using the language model specified and the
        parameters defined upon initialisation of the tagger.
        
        Parameters:
            text (str): the text to tag.
        
        Returns:
            tags (list): a list containing all the IDs of the tags that were
            detected in the text.
        """
        # execute the pipe on the text, and return the tags
        doc = self.nlp(text)
        return doc._.tags

    def tagTextCollection(self, df, text_id='EID', text_column='fulltext',
                          n_cores=4, tidy=True) :
        """Tags a collection of texts grouped together in a data frame, using parallel
        computing.        

        Parameters:
            - df (pd.DataFrame): a data frame containing a the texts to tag.
            - text_id(str) [optional]: the name of the column that contains the
            identifier of each text (defaults to 'EID')
            - text_column(str) [optional]: the name of the column that contains
            the texts (defaults to 'fulltext')
            - n_cores(int) [optional]: number of cores to use for the parallel
            computing. Defaults to 4.
        """
        results = self.run_nlp(df,
                            text_id=text_id, text_column=text_column,
                            n_cores=n_cores).apply(lambda doc : doc._.tags)

        # give original name to the index of the results
        results.index.name = text_id

        # now return the results
        if tidy :
            return tidyTags(results, self.Voc.reset_index())
        else :
            return results

def tidyTags(tags, vocabulary) :
    """Creates a data frame that lists all the tags that were identified for each
    of the texts, including the information on the tags.

    Parameters:
        tags (pd.Series): A pandas Series that contains the lists of tag IDs
        that have been detected for each text.
        vocabulary (pd.DataFrame): the vocabulary of keywords that was used to
        tag the texts. 

    Returns:
        df (pd.DataFrame): a pandas DataFrame that holds the detailed
        information on every tag that was detected.
    """
    # get the name of the index in `tags`
    index_name = tags.index.name
    if index_name is None :
        index_name = 'level_0'

    tags_length = tags.apply(len)
    if tags_length.sum() == 0 :
        mycols = [index_name]
        mycols.extend(vocabulary.columns)
        return pd.DataFrame(columns=mycols)
    tidytags = tags[tags_length>0].apply(pd.Series).stack().reset_index().\
                rename(columns={0:'ID'}).drop(columns='level_1')

    # this is needed for cases when the index ID of the vocabulary is made of
    # integers, which then get converted int in the process
    tidytags.ID = tidytags.ID.astype(str)
    return tidytags.merge(vocabulary, on='ID').set_index(index_name)

