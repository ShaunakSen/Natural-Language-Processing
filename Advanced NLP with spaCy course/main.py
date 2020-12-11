import enum
import nltk
from nltk.util import pr


def read_file():
  """
  Reads the file, and returns the text
  """
  with open(f'{fpath}\\transcript.txt', 'r') as txt_file:
    text = txt_file.read()
    txt_file.close()

    return text
    # print (text)

    all_words = nltk.tokenize.word_tokenize(text)

    ### some words like 


def normalize_text(text, steps = ['tokenize', 'whitespace_removal', 'case_normal', 'hyphen_handling', 'punc_removal']):
  """
  normalize the text according to the pipeline specified
  """
  all_words = nltk.tokenize.word_tokenize(text)

  ### whitespace_removal: remove leading and trailing whitesapces
  all_words = [word.strip() for word in all_words]

  ### for case-normalization I want to preserve the full uppercase words like MORPHEUS as they are of significance
  ### Also normalixe words like 'A'
  all_words_lower = [word if word.isupper() and len(word)>1 else word.lower() for word in all_words]

  ### some words like  'burgundy-', 'leather' are broken into different words, but ideally they are the same word
  words_hyphen_corrected = []
  idx_to_ignore = -1
  for idx, word in enumerate(all_words_lower):
    if idx == idx_to_ignore:
      continue
    if word.endswith('-') and len(word)>1:
      next_word = all_words_lower[idx+1]
      joined_word = word + next_word
      idx_to_ignore = idx+1
      print ('Adding:', joined_word)
      words_hyphen_corrected.append(joined_word)
    else:
      words_hyphen_corrected.append(word)


  print (words_hyphen_corrected)
  
  print (nltk.corpus.stopwords.words('english'))

fpath = 'C:\\Users\\shaun\\Documents\\my_projects\\Natural Language Processing\\Advanced NLP with spaCy course'

def main():

  ### Step1 : read the file
  text = read_file()
  normalize_text(text)
main()
