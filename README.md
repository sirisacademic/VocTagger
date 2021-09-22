# VocTagger
A multiprocess information extraction system able to identify hidden knowledge in textual documents using "controlled vocabularies".

## Background

Information retrieval from large textual corpora involves the identification of some key terms within the collection of text at hand. These terms are typically complied in "controlled vocabularies". 

To successfully carry out this task, a series of pattern matching rules must be defined to capture possible variants of the same concept, such as permutations of words within the concept and/or the presence of null words to be skipped. For this reason, we have carefully crafted matching rules that take into account permutations of words and that allow words within concept to be within a certain distance. Some relatively ambiguous keywords (which may match unwanted pieces of text), have a set of associated “extra” terms. These “extra” terms are defined as further terms that must co-appear, in the same sentence, together with their associated ambiguous keywords.

## Installation
To install the library, you can either use `pip` or download the source code directly and compile it yourself.

### Dependencies
Currently the library requires you to have installed the following libraries:

- nltk
- sklearn
- spacy (>3.0)
- matplotlib
- numpy
- pandas (>1.0)
- shutil
- shlex
- unicodedata
- requests

### Installation for end users
The most convenient way of installing the library is by using `pip`:

```bash
pip install git+https://github.com/sirisacademic/VocTagger
```
This will install the library on your system, and updates will be managed by `pip` itself.


### How to use it? A jupyter-notebook is provided in the examples folder

#### Tagging a single text
```bash
import VocTagger as vt
import pandas as pd

tagger = vt.VocTagger(voc)

txtToTag="""
Ensuring healthy lives and promoting well-being at all ages is essential to sustainable development.
Currently, the world is facing a global health crisis unlike any other — COVID-19 is spreading human suffering, 
destabilizing the global economy and upending the lives of billions of people around the globe.
Before the pandemic, major progress was made in improving the health of millions of people. 
Significant strides were made in increasing life expectancy and reducing some of the common killers associated 
with child and maternal mortality. But more efforts are needed to fully eradicate a wide range of diseases and address many different persistent 
and emerging health issues. By focusing on providing more efficient funding of health systems, improved sanitation and hygiene, 
and increased access to physicians, significant progress can be made in helping to save the lives of millions.
"""

kwIdFounded = tagger.tagText(txtToTag)
kwIdFounded #It contains the keywords ID founded in the text
```
#### Tagging a collecation of texts

```bash
import VocTagger as vt
import pandas as pd

tagger = vt.VocTagger(voc)

df=pd.DataFrame({
    "text":[""" Ensuring healthy lives and promoting well-being at all ages is essential to sustainable development.
        Currently, the world is facing a global health crisis unlike any other — COVID-19 is spreading human suffering, 
        destabilizing the global economy and upending the lives of billions of people around the globe.
        Before the pandemic, major progress was made in improving the health of millions of people. 
        Significant strides were made in increasing life expectancy and reducing some of the common killers associated 
        with child and maternal mortality. But more efforts are needed to fully eradicate a wide range of diseases and address many different persistent 
        and emerging health issues. By focusing on providing more efficient funding of health systems, improved sanitation and hygiene, 
        and increased access to physicians, significant progress can be made in helping to save the lives of millions.""",
        
        """One of the first indicators of incipient Alzheimer’s disease (AD) is the development of MCI. Subtle, 
        hard-to-detect changes in the brain accompany MCI as the condition advances.
        Now, a study from researchers at Kaunas University of Technology (KTU) in Lithuania presents a 
        newly developed deep-learning computer algorithm that can accurately detect and differentiate 
        the stages of MCI from fMRI scans.
        """   
           ],
    "idText":["ID_1",
    
              "ID_2"]
    
})

tagger.tagTextCollection(df, text_id='idText', text_column='text', n_cores=4).reset_index()
```



### Developed by Siris Academic (https://sirisacademic.com/)