import numpy as np
import pandas as pd
import time

start = time.time()


df = pd.read_csv('scotus-opinions/all_opinions.csv') #panda dataframe

#columns: ['author_name', 'category', 'per_curiam', 'case_name', 'date_filed', 'federal_cite_one', 'absolute_url', 'cluster', 'year_filed', 'scdb_id', 'scdb_decision_direction', 'scdb_votes_majority', 'scdb_votes_minority', 'text']

#they all must have author names and text
df = df[df['author_name'].notnull() & df['text'].notnull()]

#don't want per curiam opinions because they are not usually written by all justices together (i.e. no single author)
df = df[df.category != 'per_curiam']

#use only opinions of length at least 3000 characters, as recommended (recommended by kaggle dataset author)
df = df[df.text.str.len() > 3000]

#have to fix some alternate author name spellings
df.loc[df.author_name == 'justice m\'kinley', 'author_name'] = 'justice mckinley'
df.loc[df.author_name == 'justice m\'lean', 'author_name'] = 'justice mclean'


#since I am just doing authorship attribution, all other information other than author and text is unnecessary
df = df[['author_name','text']]

#am going to do authorship attribution between 5 authors only, using the 5 most prolific authors
prolific_author_counts = df.author_name.value_counts()[:5] #a panda series
prolific_author_names = list(prolific_author_counts.index)  #the author names are the indices of the series
df = df[df.author_name.isin(prolific_author_names)]

#need to grab ONLY min_num_opinions_per_prolific_author from each author so that there is the same number from each author
min_num_opinions_per_prolific_author = prolific_author_counts[prolific_author_names[-1]]
df = df.groupby('author_name').head(min_num_opinions_per_prolific_author)

#randomly shuffle all samples
df = df.reindex(np.random.permutation(df.index))
df.to_csv("clean_opinions.csv", index=False)

end = time.time()
print("elapsed time: "+str(end-start)+" seconds")
