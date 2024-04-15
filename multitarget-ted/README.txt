#####################################################################################
# Multitarget TED Talks Task
# v. 0.2 (2018-05-28)                                                               #
# prepared by Kevin Duh (kevinduh @ cs.jhu.edu)                                     #
#####################################################################################

This is a collection of multitarget bitexts based on TED Talks (https://www.ted.com)
The data is extracted from WIT3 (https://wit3.fbk.eu), which is also used for
the IWSLT Machine Translation Evaluation Campaigns. 

We have a different train/dev/test split from IWSLT. Here, all the dev and test 
sets have the same English side and come from the same talks. There are 20 languages, 
so this is 20-way parallel. This can support the evaluation of:

  * Multitarget translation (e.g. jointly translating to French, German, 
    Japanese, etc. from English)
  * Multisource translation (e.g. jointly translating into German given 
    French and Japanese as input)
  * Pivot translation (e.g. translating from English to Japanese through a 
    German pivot) 
  * Analysis of machine translation results across many typologically
    different languages. (While BLEU scores should not be directly compared
    across datasets, using the same dev/test should aid multilingual analysis)


The train set for different languages may have different English sides, ranging 
from 77k-188k "sentences" (1.5M to 3.9M English tokens). See stats.txt.
The dev set has 1958 sentences (38438 English tokens), extracted from 30 talks.
The test1 set has 1982 sentences (36499 English tokens), extracted from 30 talks.
There is a test2 (not released), saved for future purposes. 

Note that all talks are originally spoken and transcribed in English, then
translated by TED translators. 

------------------------------------------------------------------------------------
Directory structure

There are 19 subdirectories, corresponding to each of the target language:
${lang} = ar (Arabic), bg (Bulgarian), cs (Czech), de (German), 
          fa (Farsi), fr (French), he (Hebrew), hu (Hungarian), 
          id (Indonesian), ja (Japanese), ko (Korean), pl (Polish),
          pt (Portuguese), ro (Romanian), ru (Russian), tr (Turkish),
          uk (Ukranian), vi (Vietnamese), zh (Chinese)

Each directory represents a bitext for translating into/from English (en).

en-${lang}/
   raw/   # raw sentences: extracted from WIT3 and sentence merged.
      ted_train_en-${lang}.raw.$lang  # training bitext, foreign side
      ted_train_en-${lang}.raw.en     # training bitext, english side
      ted_dev_en-${lang}.raw.$lang    # dev bitext, foreign side
      ted_dev_en-${lang}.raw.en       # dev english (same regardless of ${lang})
      ted_test1_en-${lang}.raw.$lang  # test bitext, foreign side
      ted_test1_en-${lang}.raw.en     # test english (same regardless of ${lang}
   tok/  # tokenized version of sentences in raw/
      ted_train_en-${lang}.tok.$lang  # training bitext, foreign side
      ted_train_en-${lang}.tok.en     # training bitext, english side
      ted_train_en-${lang}.tok.clean.$lang  # filtered train above, max 80 words
      ted_train_en-${lang}.tok.clean.en     # filtered train above, max 80 words
      ted_dev_en-${lang}.tok.$lang    # dev bitext
      ted_dev_en-${lang}.tok.en       # dev bitext
      ted_test1_en-${lang}.tok.$lang  # test bitext
      ted_test1_en-${lang}.tok.en     # test bitext

Most languages are tokenized with the Moses tokenizer. The pipeline is:
raw -> [Joshua normalize.pl] -> [Moses tokenizer] -> [lowercase.pl] -> tok

Exceptions are: Arabic is tokenized with PyArabic3, Korean is tokenized with Mecab-ko, Japanese is tokenized with Kytea 0.4.7, Chinese is tokenized with Jieba. The pipeline in those cases are:
raw -> [Joshua normalize.pl] -> [language-specific tokenizer above] -> tok

For all language pairs in tok/, the training bitext contains the unfiltered version corresponding to raw/ (ted_train_en-${lang}.tok.en) as well as the filtered version (ted_train_en-${lang}.tok.clean.en). Filtering by sentence length is not performed on dev and test sets.


------------------------------------------------------------------------------------
Additional Meta-data

(1) Talk ID

In the subdirectory talkids/ 
    shared-dev and shared-test1 contain the talk IDs of the dev and test1 sets
    the training set talk IDs are contained in e.g. trainid.en-ar

These Talk IDs correspond to the talkid tag in WIT3's XML format: 
    https://wit3.fbk.eu/mono.php?release=XML_releases

A copy of the version the XML file we used (just for English) is at: 
    talkids/ted_en.xml 


(2) Audio:

The files tok/*.seekvideo contains the information needed to recover audio segments from the English side. Each line in e.g. tok/train_en-de.tok.clean.seekvideo corresponds to the same line tok/train_en-de.tok.clean.en, and contains a list of tuples of the form <talkid:seekvideo-id> indicating the talkid and seekvideo counter in seconds. 

For example, the line: "<838:0> <838:2000> <838:4000> <838:6000>" means that the corresponding English transcript comes from talkid 838, and includes audio that starts at the 0, 2000, 4000, and 6000 second counters. Note there may be more than one seekvideo id per line because we perform sentence merging on the captions.


------------------------------------------------------------------------------------
Terms of use & Acknowledgments:

TED makes its collection available under the Creative Commons BY-NC-ND license. 
Please acknowledge TED when using this data. We acknowledge the authorship of 
TED Talks (BY condition). We are not redistributing the transcripts for 
commercial purposes (NC condition) nor making derivative works of the original 
contents (ND condition). 

We kindly thank WIT3, which provides ready-to-use versions for research purposes.
For a detailed description of WIT3, see:
M. Cettolo, C. Girardi, and M. Federico. 2012. WIT3: Web Inventory of Transcribed and Translated Talks. In Proc. of EAMT, pp. 261-268, Trento, Italy

If you would like to cite this task:

@misc{duh18multitarget,
	author = {Kevin Duh},
	title = {The Multitarget TED Talks Task}, 
	howpublished = {\url{http://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/}},
	year = {2018},
}



