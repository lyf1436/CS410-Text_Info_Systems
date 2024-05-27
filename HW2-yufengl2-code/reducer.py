# Code based on https://blog.devgenius.io/big-data-processing-with-hadoop-and-spark-in-python-on-colab-bff24d85782f
from operator import itemgetter
import sys
import json

current_word = None
current_count_doc = {}
word = None

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    line=line.lower()

    # parse the input we got from mapper.py
    word, doc, count = line.split('\t', 2)
    
    try:
      count = int(count)
    except ValueError:
      #count was not a number, so silently
      #ignore/discard this line
      continue

    # this IF-switch only works because Hadoop sorts map output
    # by key (here: word) before it is passed to the reducer
    if current_word == word:
      if doc in current_count_doc:
        current_count_doc[doc] += count
      else:
        current_count_doc[doc] = count
        
    else:
      if current_word:
        # write result to STDOUT
        current_output = json.dumps(current_count_doc)
        print('%s\t%s' % (current_word, current_output))
      current_count_doc = {}
      current_count_doc[doc] = count
      current_word = word


# do not forget to output the last word if needed!
if current_word == word:
  current_output = json.dumps(current_count_doc)
  print('%s\t%s' % (current_word, current_output))