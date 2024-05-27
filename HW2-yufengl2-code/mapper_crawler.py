#Code based on https://blog.devgenius.io/big-data-processing-with-hadoop-and-spark-in-python-on-colab-bff24d85782f
import sys
import io
import pandas as pd  

df = pd.read_csv(sys.stdin, header=None)

for i, row in df.iterrows():
  print('%s\t%s\t%s' % (row[0], row[1], 1))
