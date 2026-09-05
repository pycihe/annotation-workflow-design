# Extract training data for tweet and reply processing
# Requires trains.json file to be present in the same directory

import json
import re

def replace_urls(sentence):
    # replace email addresses with [this email]
    replaced_sentence = re.sub(r'\S+@\S+', '[this email]', sentence)
    replaced_sentence = re.sub(r'https?://\S+', '[this link]', replaced_sentence)
    return replaced_sentence

def remove_at(sentence):
    words = sentence.split()
    filtered_words = [word for word in words if not word.startswith('@')]
    filtered_sentence = ' '.join(filtered_words)
    return filtered_sentence

def split_sentences(string):
    # Split the string into sentences using regular expressions
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', string)
    return sentences

def remove_unicode(string):
    cleaned_string = ''
    for char in string:
        try:
          if ord(char) < 128 and char != '\n':
            cleaned_string += char
        except ValueError:
           pass
    return cleaned_string

def remove_undef(s):
  clean = ""
  ss = remove_unicode(s)
  for ch in ss:
    try:
      # unicodedata.name(ch)
      clean += ch
    except ValueError:
      pass
  return clean

sn = 1
ave_len = 0
max_len = 0
header_list = []
twt = open("train_sample_tweets.txt", "w")
rep = open("train_sample_replies.txt", "w")
tweets = []
replies = []
with open('train.json') as cases:
    dict = json.load(cases)
    for case in dict:
      if case['agentURL']['turn'] == '2' : 
        replies = split_sentences(case['agentURL']['url_utterance'])
        id = case['agentURL']['tweet_ID']
        for r in range(len(replies)):
          replies[r] = replace_urls(remove_at(remove_undef(replies[r])))
        if len(replies) < 2: # at least two sentences in a reply, otherwise skip this case
          continue
        if '[this link]' in replies[0] or 'http' in replies[0] or 'https' in replies[0]:
          continue
        tweet = replace_urls(remove_at(remove_undef(case['dialogContent'][0]['message'])))
        if tweet in tweets or len(tweet) == 0:
          continue
        tweets.append(tweet)
        twt.write(f"{int(sn)}\t{id}\t{tweet}\n")
        for r in replies:
          if '[this link]' in r or 'http' in r or 'https' in r:
            replies = replies[:replies.index(r)]
            break
        rep.write(f"{sn}\t{' '.join(replies)}\n")
        sn += 1
print(f"Total number of tweets: {sn-1}")
