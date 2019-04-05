import numpy as np
import pandas as pd
import re
import time
import nltk
#had to make this change https://github.com/nltk/nltk/pull/2186/files#diff-34022dd7c15afeba0f26319ac39f349dR180
from nltk.corpus import stopwords

start = time.time()

df = pd.read_csv('csv_files/clean_opinions.csv') #panda dataframe -- use cleaned data created earlier
df = df.reindex(np.random.permutation(df.index)) #randomly reshuffle samples

no_numbers = re.compile("^[^0-9]+$") #we don't want any numbers--numbers would be case and not authorship dependent

new_df = df.copy() #can't modify something we're iterating over
for index, row in df.iterrows():
    sentences = nltk.sent_tokenize(row['text'])
    initial_words = [nltk.word_tokenize(sentence) for sentence in sentences]
    new_df.loc[index]['text'] = " ".join(word for sentence in initial_words for word in sentence if no_numbers.match(word))


new_df.to_csv("csv_files/removed_numbers_apr4.csv", index=False)


end = time.time()
print("elapsed time: "+str(end-start)+" seconds")