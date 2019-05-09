# scotus_authorship_attribution

This project performs unsupervised authorship attribution on Supreme Court of the United States opinions.

**make_clean_opinions.py** cleans the dataset of ~35,000 opinions and 96 different authors and creates a csv file with 4075 opinions whose authorships are equally distributed among the top 5 most prolific authors.

**tokenize_no_ne.py** uses the nltk tokenizer to tokenize the text of the cleaned opinions. It removes numbers and removes named entities using the nltk named entity recognition tool, and creates a csv file containing, for each opinion, its author and its tokens. This tokenization step is being done separately because the named entity recognition process is very slow and need not be repeated every time clustering is performed.

**tokenize_with_ne.py** does the same thing as tokenize_no_ne.py but does not remove named entities. Both strategies are attempted to see if named entity removal affects the classification accuracy.

**authorship_clustering.py** performs K-means or Hierarchical clustering among 2, 3, 4, or 5 authors using the dataset produced by tokenize_no_ne.py or tokenize_with_ne.py. See **scotus_authorship_attribution_project_report.pdf** for details and results.
