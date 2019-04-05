import numpy as np
import pandas as pd
import time
import math
import re
import random
import sys
import itertools
import nltk
#had to make this change https://github.com/nltk/nltk/pull/2186/files#diff-34022dd7c15afeba0f26319ac39f349dR180
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import stopwords


def get_tfidf_vecs(num_words_in_vec,num_words,word_list,word_counts,doc_word_lists):
    num_docs = len(doc_word_lists)

    #can specify whether we want to use n most frequent words
    if(num_words_in_vec != -1): #-1 means use all words
        num_words = num_words_in_vec
        word_list_sorted_by_counts = sorted(word_list,key=lambda word: word_counts[word], reverse=True)
        word_list = word_list_sorted_by_counts[:num_words] #take first n words
        random.shuffle(word_list)
        for i in range(len(doc_word_lists)):
            doc_word_lists[i] = [word for word in doc_word_lists[i] if word in word_list]

    word_index_map = {}
    for i in range(len(word_list)):
        word_index_map[word_list[i]] = i

    num_docs_with_each_word = [0]*len(word_list)
    num_words = len(word_list)
    tfs_for_each_doc = [] #term frequencies

    for doc_word_list in doc_word_lists:
        counts = [0]*num_words
        words_in_doc = np.array([0]*len(word_list)) #a 0 or 1 mask on word_list representing which words are in this doc
        for word in doc_word_list:
            word_index = word_index_map[word]
            counts[word_index]+=1
            words_in_doc[word_index] = 1

        num_docs_with_each_word = np.sum([num_docs_with_each_word, words_in_doc],axis=0)
        tfs_for_each_doc.append(np.array(counts)/num_docs)#use term frequencies

    #idf is calculated once for each term
    idfs = [math.log(num_docs/num_docs_with_each_word[word_index_map[word]]) for word in word_list]
    #tfidf is for each term for each doc
    tf_idfs = np.apply_along_axis(lambda tfs: np.multiply(tfs,idfs),1,tfs_for_each_doc) #tf_idf = tf*idf
    #normalize each doc vector
    normalized_tf_idfs = np.apply_along_axis(lambda tf_idf: tf_idf/np.linalg.norm(tf_idf),1,tf_idfs)
    return normalized_tf_idfs


def get_tf_vecs(num_words_in_vec,num_words,word_list,word_counts,doc_word_lists):
    #can specify whether we want to use n most frequent words
    if(num_words_in_vec != -1 and num_words_in_vec < num_words): #-1 means use all words
        num_words = num_words_in_vec
        word_list_sorted_by_counts = sorted(word_list,key=lambda word: word_counts[word], reverse=True)
        word_list = word_list_sorted_by_counts[:num_words] #take first n words
        random.shuffle(word_list)

    #create document vectors
    word_index_map = {}
    for i in range(len(word_list)):
        word_index_map[word_list[i]] = i

    doc_vectors = []
    for doc_word_list in doc_word_lists:
        counts = [0]*num_words
        reduced_doc_word_list = [word for word in doc_word_list if word in word_index_map]
        for word in reduced_doc_word_list:
            counts[word_index_map[word]]+=1

        doc_vectors.append(np.array(counts))

    doc_vectors = np.array(doc_vectors)
    normalized_doc_vectors = np.apply_along_axis(lambda doc_vec: doc_vec/np.linalg.norm(doc_vec),1,doc_vectors)
    return normalized_doc_vectors

def get_doc_vectors(df,num_words_in_vec):
    unique_words = set()
    doc_word_lists = []
    word_counts = {}

    stop_words = set(stopwords.words('english'))
    no_punc = re.compile("^[A-Za-z]+$")

    for index, row in df.iterrows():
        words = [word.lower() for word in row['text'].split(" ")]

        #to remove stop words and/or punctuation:
        # words = [word for word in words if word not in stop_words]

        unique_words = unique_words.union(set(words))
        doc_word_lists.append(words)
        for word in words:
            if(word in word_counts):
                word_counts[word]+=1
            else:
                word_counts[word]=1


    word_list = list(unique_words)
    num_words = len(word_list)

    return get_tf_vecs(num_words_in_vec,num_words,word_list,word_counts,doc_word_lists)
    #or alternatively
    # return get_tfidf_vecs(num_words_in_vec,num_words,word_list,word_counts,doc_word_lists)


def print_to_file(output_file_name,correct_author_for_each_cluster,all_cluster_counts,precisions,recalls,f_measures,total_results):
    with open(output_file_name,"w") as f:
        for i, author in correct_author_for_each_cluster.items():
            this_cluster_counts = all_cluster_counts[i]
            f.write(" cluster index: "+str(i)+", author: "+author+", cluster counts: "+str(this_cluster_counts)+"\n")
            f.write("precision: "+str(precisions[i])+" recall: "+str(recalls[i])+" f-measure: "+str(f_measures[i])+"\n\n")

        f.write("\ntotal percent correct: "+str(total_results[3]))
        f.write("\naverage precision: "+str(total_results[0]))
        f.write("\naverage recall: "+str(total_results[1]))
        f.write("\naverage f-measure: "+str(total_results[2]))
        f.close()

def calc_accuracy(labels,k,authors,doc_vectors,output_file_name):
    n = len(authors) #=len(doc_vectors)=total number documents i.e. number datapoints

    authors_in_each_cluster = [[] for i in range(k)]
    for i in range(n):
        cluster_index = labels[i]
        authors_in_each_cluster[cluster_index].append(authors[i])

    noise = [0]*k #number docs in a cluster not written by the author associated with the cluster
    correctly_clustered = [0]*k #number docs in a cluster written by the author associated with the cluster
    correct_author_for_each_cluster = {}
    all_cluster_counts = []
    for i in range(k): #for each cluster
        #count how many of each author this cluster has
        (author_labels,counts) = np.unique(np.array(authors_in_each_cluster[i]),return_counts=True)
        #there could be a tie for which author is most frequent in this cluster-- tie is broken by argmax returning first value
        max_index = np.argmax(counts)
        most_frequent_author = author_labels[max_index]
        correct_author_for_each_cluster[i] = most_frequent_author
        num_correct = counts[max_index]
        correctly_clustered[i] = num_correct  #the number correctly clustered to the author associated with this cluster
        noise[i] = sum(counts) - num_correct #all incorrectly attributed to the author associated with this cluster
        all_cluster_counts.append(counts)

    #since we've divided dataset intentionally, there's the same amount of docs from each author
    num_docs_by_each_author = len(doc_vectors)/k

    #number docs that should be in each cluster but are not
    silence = [num_docs_by_each_author - correctly_clustered[i] for i in range(k)]

    #get precision and recall for each cluster
    precisions = [0 if (correctly_clustered[i] == 0) else 100*correctly_clustered[i]/(correctly_clustered[i]+noise[i]) for i in range(k)]
    recalls = [0 if (correctly_clustered[i] == 0) else 100*correctly_clustered[i]/(correctly_clustered[i]+silence[i])for i in range(k)]
    f_measures = [0 if (correctly_clustered[i] == 0) else 2*((precisions[i]*recalls[i])/(precisions[i]+recalls[i])) for i in range(k)]

    percent_correct = 100*sum(correctly_clustered)/len(doc_vectors)

    #precision, recall, and f_measure is each a row, so want mean of each row i.e. axis=1
    total_results = list(np.mean(np.array([precisions,recalls,f_measures]),axis=1))+[percent_correct]

    #print a file to get confusion matrix and all precisions, recalls, f_measures for each cluster
    print_to_file(output_file_name,correct_author_for_each_cluster,all_cluster_counts,precisions,recalls,f_measures,total_results)

    return total_results



#ARGUMENTS: 1. vector length; 2. number of clusters/authors; 3. output file name; 4. csv input file name


start = time.time()

#the input file has cleaned words and author names
input_file_name = 'csv_files/removed_numbers.csv'
if(len(sys.argv) > 4):
    input_file_name = sys.argv[4]

num_words_in_vec = -1 #if -1 means we use all words, otherwise can specify to use only n most frequent words
if(len(sys.argv) > 1):
    num_words_in_vec = int(sys.argv[1])


#panda dataframe -- use cleaned data created earlier
df = pd.read_csv(input_file_name)

doc_vectors = get_doc_vectors(df,num_words_in_vec)
authors = np.array(df['author_name'].tolist())
k = len(set(authors)) #we want number of clusters = number of authors

if(len(sys.argv) > 2): #if we don't want to use all the authors -- want to compare only 2, 3, or 4
    k = int(sys.argv[2])

original_output_file_name = 'results/test.txt'
if(len(sys.argv) > 3):
    original_output_file_name = sys.argv[3]

unique_authors = list(set(authors))
output_file_name = original_output_file_name

i = 0
all_combinations_results = []
#do every combination of k authors, since different combinations will give different results
for author_sublist in itertools.combinations(unique_authors, k):
    k_authors_mask = np.isin(authors,author_sublist)
    output_file_name = original_output_file_name[:-4]+str(i)+'.txt'

    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    cluster.fit_predict(doc_vectors)
    labels = cluster.labels_

    #or alternatively
    # kmeans = KMeans(n_clusters=k, random_state=None).fit(doc_vectors)
    # labels = kmeans.labels_

    #results is an array holding in this order: average precision, average recall, average f-measure, total percent correct
    results = calc_accuracy(labels,k,authors[k_authors_mask],doc_vectors[k_authors_mask],output_file_name)
    all_combinations_results.append(np.array(results))
    i+=1

average_results = np.mean(np.array(all_combinations_results), axis=0)
print("average results for k="+str(k)+": "+str(average_results))

end = time.time()
print("elapsed time: "+str(end-start)+" seconds")
