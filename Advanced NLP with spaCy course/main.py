### Import libraries

import enum
import nltk
from nltk.corpus.reader import lin
from nltk.util import pr
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import re
import pandas as pd
import plotly.express as px
import spacy
nlp = spacy.load("en_core_web_sm")


### global variable for file path location of the transcript.txt file
fpath = 'C:\\Users\\shaun\\Documents\\my_projects\\Natural Language Processing\\Advanced NLP with spaCy course'


def read_file():
  """
  Reads the file, and returns the text
  """
  with open(f'{fpath}\\transcript.txt', 'r') as txt_file:
    text = txt_file.read()
    txt_file.close()

    return text



def normalize_text(text):
  """
  normalize the text: tokenize -> whitespace_removal -> case norm -> hyphen handling -> punc removal -> stopwords removal (oprtional)
  returns: punc_removed: steps applied: 'tokenize', 'whitespace_removal', 'case_normal', 'hyphen_handling', 'punc_removal'
           stopwords_removed: steps applied: 'tokenize', 'whitespace_removal', 'case_normal', 'hyphen_handling', 'punc_removal', 'stopwords_removed'
  """
  all_words = nltk.tokenize.word_tokenize(text)

  ### whitespace_removal: remove leading and trailing whitesapces
  all_words = [word.strip() for word in all_words]

  ### for case-normalization I want to preserve the full uppercase words like MORPHEUS as they are of significance
  ### Also normalixe words like 'A'
  all_words_lower = [word if word.isupper() and len(word)>1 else word.lower() for word in all_words]

  ### if there are words like 'Neo' we only want the uppercase versions 
  all_words_case_norm = [word_.upper() if  word_.lower() == 'morpheus' or word_.lower() == 'neo' else word_ for word_ in all_words_lower]


  ### some words like  'burgundy-', 'leather' are broken into different words, but ideally they are the same word: the following code block corrects that
  ### store the results in words_hyphen_corrected
  words_hyphen_corrected = []
  idx_to_ignore = -1
  for idx, word in enumerate(all_words_case_norm):
    if idx == idx_to_ignore:
      continue
    if word.endswith('-') and len(word)>1:
      next_word = all_words_case_norm[idx+1]
      joined_word = word + next_word
      idx_to_ignore = idx+1
      words_hyphen_corrected.append(joined_word)
    else:
      words_hyphen_corrected.append(word)

  ### replace punctuations by whitespace
  punc_removed = [re.sub(r'[^\w\d]', ' ', word) if len(word)==1 else word for word in words_hyphen_corrected]

  ### remove stopwords: uses the nltk bulti-in stopwords list
  stop_words = set(stopwords.words('english'))
  stopwords_removed = [w for w in punc_removed if not w in stop_words]

  ### stemming: while analyzing frequency it will be useful, for example (laughing, laugh, laughed) are all forms of 'laugh' and counted together 
  # porter = PorterStemmer()
  # stemmed_words = [porter.stem(word) for word in stopwords_removed]


  return punc_removed, stopwords_removed


def analyze_frequency(punc_removed, stopwords_removed, remove_stopwords=True, cutoff_freq=1, write_to_csv='False'):
  """
  Now that we have normalized the text we can analyze the frequecy of the words
  Produces a bar chart using the following params
  punc_removed: words with punctuations removed
  stopwords_removed: words with both stopwords and punctuations removed
  remove_stopwords: whether to keep or remove stopwords in the analysis
  cutoff_freq: min freq to consider, for example if cutoff_freq=1, consider all words which occur more than 1 time
  write_to_csv: whether to write the generated frequency table to a local csv file with same location as that of transcript.txt
  """

  if remove_stopwords:
    words_list = stopwords_removed
  else:
    words_list = punc_removed
    

  freq_dict = {}
  for word in words_list:
    if word == ' ':
      continue
    if word in freq_dict:
      freq_dict[word] +=1
    else:
      freq_dict[word] = 1
    
  df_data = {'word': list(freq_dict.keys()), 'frequency': list(freq_dict.values())}

  df = pd.DataFrame(df_data).sort_values(by='frequency', ascending=False)

  df = df.query('frequency > @cutoff_freq')

  fig = px.bar(df, x='word', y='frequency', title=f'Top words with freq cutoff: {cutoff_freq} | Stopwords present: {not remove_stopwords}')
  fig.show()

  if write_to_csv:
    df.to_csv(f'{fpath}\\freq_table.csv', index=False)

  return

def extract_names():
  """
  Read the file line by line
  The names in this transcript follow a general pattern like:
  1. Start at the beginning of each line
  2. The line contains a single word
  3. The word is in uppercase
  """
  all_extracted_names = []
  with open(f'{fpath}\\transcript.txt', 'r') as txt_file:
    all_lines = txt_file.readlines()
    txt_file.close()
    for line_ in all_lines:
      line_ = line_.strip()
      words_ = line_.split()
      ### single word
      if len(words_) == 1:
        ### get the word
        probable_name = words_[0]
        if probable_name.strip() != '' and probable_name.strip().isupper():
          if probable_name not in all_extracted_names:
            all_extracted_names.append(probable_name)

  return all_extracted_names

def extract_names_all(names):
  
  """
  Runs spacy NER on the text to extract entitites with type='PERSON'
  Takes the names extracted from the previous step and appends to that
  Returns the modified list
  """


  with open(f'{fpath}\\transcript.txt', 'r') as txt_file:
    text = txt_file.read()
    txt_file.close()

    ## init a spacy doc on the text
    doc = nlp(text)
    names = []
    for ent in doc.ents:
      if ent.label_ == 'PERSON'  and ent.text.strip() != '' and ent.text.strip() not in names:
        names.append(ent.text.strip())
    return names


def main():

  ### Step1 : read the file
  text = read_file()

  ### normalize text: 
  punc_removed, stopwords_removed = normalize_text(text)


  ### some words like 'm, 're, 's still exist, should be filtered
  punc_removed_clean, stopwords_removed_clean = [], []
  for word in stopwords_removed:
    if not word.startswith('\''):
      stopwords_removed_clean.append(word)

  for word in punc_removed:
    if not word.startswith('\''):
      punc_removed_clean.append(word)

  analyze_frequency(punc_removed, stopwords_removed_clean, True, 2, False)

  extracted_names = extract_names()

  all_extracted_names = extract_names_all(extracted_names)

  print (all_extracted_names)

main()

