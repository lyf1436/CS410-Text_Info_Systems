For running Test.csv, run all cells in CS410_HW2_Default_Data
For using data crawled from wikipedia, run all cells in CS410_HW2_Wiki_Data
Query results are in the ipynb
inverted index is stored as word->posting as json
Note that for cross validation, the process is quite long, as it needs to loop through all documents against all documents (scoring once takes ~10min, grid search takes ~1hr)