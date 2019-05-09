import numpy as np
import pandas as pd
import re
import time
import nltk
#had to make this change https://github.com/nltk/nltk/pull/2186/files#diff-34022dd7c15afeba0f26319ac39f349dR180

start = time.time()

df = pd.read_csv('clean_opinions.csv') #panda dataframe -- use cleaned data created earlier

df = df.reindex(np.random.permutation(df.index)) #randomly reshuffle samples


#I assume that people and places are irrelevant to authorship -- rather they just have to do with the specifics of one case
#Therefore I will get rid of all named entities
#see table 5.1 at https://www.nltk.org/book/ch07.html for a detailed explanation
# named_entity_types = ['ORGANIZATION','PERSON','LOCATION','DATE','TIME','MONEY','PERCENT','FACILITY','GPE']


no_numbers = re.compile("^[^0-9]+$") #we don't want any numbers--numbers would be case and not authorship dependent

new_df = df.copy() #can't modify something we're iterating over
for index, row in df.iterrows():
    #looked at https://stackoverflow.com/questions/43742956/fast-named-entity-removal-with-nltk
    sentences = nltk.sent_tokenize(row['text'])
    initial_words = [nltk.word_tokenize(sentence) for sentence in sentences]
    pos_tagged = nltk.pos_tag_sents(initial_words, lang='eng') #part of speech tagging
    chunked_sentences = nltk.ne_chunk_sents(pos_tagged) #named entity identification

    #looked at https://gist.github.com/onyxfish/322906/2089c1f9eb10f320d552e69d99503dbeb677e19b July 2, 2015 comment
    #see https://www.nltk.org/book/ch07.html
    new_text = ""
    for tree in chunked_sentences: #nltk chunks into syntax trees
        #looked at https://stackoverflow.com/questions/43742956/fast-named-entity-removal-with-nltk
        #named entities are syntactic constituents so they have type nltk.Tree
        this_text = " ".join([leaf[0].lower() for leaf in tree if type(leaf) != nltk.Tree and no_numbers.match(leaf[0])])
        new_text += " "+this_text

    new_df.loc[index]['text'] = new_text


new_df.to_csv("tokens_numbers_ne_removed.csv", index=False)


end = time.time()
print("elapsed time: "+str(end-start)+" seconds")
