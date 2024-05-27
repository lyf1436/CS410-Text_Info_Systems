#Code based on https://blog.devgenius.io/big-data-processing-with-hadoop-and-spark-in-python-on-colab-bff24d85782f
import sys
import io
import re
import nltk
import pandas as pd  
from nltk.stem import LancasterStemmer

nltk.download('stopwords',quiet=True)
from nltk.corpus import stopwords
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

stop_words = set(stopwords.words('english'))
input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='latin1')
lancaster = LancasterStemmer()

docid = 1 #actually line id
for line in input_stream:
  line = line.split(',', 3)[2]
  line = line.strip()
  line = re.sub(r'[^\w\s]', '',line)
  line = line.lower()
  for x in line:
    if x in punctuations:
      line=line.replace(x, "\t")

  words=line.split()
  for word in words:
    if word not in stop_words:
      print('%s\t%s\t%s' % (lancaster.stem(word), docid, 1))
  docid +=1 
