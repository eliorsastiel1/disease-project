import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import pickle
import lda

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
import numpy as np
np.random.seed(2018)


#Data Pre-processing
#We will perform the following steps:
#Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
#Words that have fewer than 3 characters are removed.
#All stopwords are removed.
#Words are lemmatized — words in third person are changed to first person and verbs in past and future tenses are changed into present.
#Words are stemmed — words are reduced to their root form.

def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess_text(text):
    nltk.download('wordnet')
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            #result.append(lemmatize_stemming(token))
            result.append(token)
    return result


def build_dictionary(path):
    #vocab = Counter()
    vocab = dict()    
    number_of_documents = 0
    directory = os.fsencode(path)
    file2 = open("dataset\clean2.txt","w+") 
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        #file2.write("%s " % filename)  
        f = open('{0:s}/{1:s}'.format(path,filename), "r",encoding="utf8")
        words=preprocess_text(f.read())
        for word in words:
            file2.write("%s " % word) 
            if word not in vocab:
                vocab[word] = len(vocab)         
        number_of_documents += 1
        file2.write("\n")
    inverse_vocab = dict([ (value, key) for key, value in vocab.items()])
    file2.close()
    return vocab, inverse_vocab, number_of_documents



def create_document_matrix(text_file_name, vocab, number_of_documents):
    import scipy.sparse
    corpus = scipy.sparse.lil_matrix( (number_of_documents, len(vocab)),dtype=np.int)
    texts  = []
    f = open(text_file_name, "r",encoding="utf8")
    text=f.read()
    f.close()

    with open(text_file_name) as fp:
        line = fp.readline()
        texts.append(line)
        words = line.split()
        doc_idx=0
        for word in words:
            corpus[doc_idx, vocab[word]] += 1
        doc_idx=doc_idx+1
        while line:
            line = fp.readline()
            texts.append(line)
            words = line.split()
            for word in words:
                corpus[doc_idx, vocab[word]] += 1
            doc_idx=doc_idx+1
            
    return corpus,texts


def print_LDA_topics(model, inverse_vocab, top_n_words=15):
    topics = []
    for i, topic_dist in enumerate(model.topic_word_ ):
        top_word_ids = np.argsort(topic_dist)[:-top_n_words:-1]
        topic_words = [ inverse_vocab[id] for id in top_word_ids ]
        topics.append(topic_words)
        print('Topic {0:d}: {1:s}'.format(i, ' '.join(topic_words)))
    return topics


pickle_file_name =  'Results/lda.pickle'

if not os.path.isfile(pickle_file_name):
    vocab, inverse_vocab, number_of_documents = build_dictionary('dataset/Mayo/Description')
    print(f'Number of unique words (vocab size): {len(vocab)}')
    print(f'Number of documents: {number_of_documents}')
    corpus,texts = create_document_matrix("dataset\clean2.txt", vocab, number_of_documents)
    print(f'Corpus size: {corpus.shape}')
    print('Sparse matrix is {0:4.4f}% populated'.format(100*float(corpus.count_nonzero())/(corpus.shape[0]*corpus.shape[1])))

    model = lda.LDA(n_topics=8, n_iter=500)
    model.fit(corpus)

    with open(pickle_file_name, 'wb') as out_file:
        pickle.dump(model,out_file, protocol=-1)
        pickle.dump(vocab,out_file, protocol=-1)
        pickle.dump(inverse_vocab,out_file, protocol=-1)
        pickle.dump(number_of_documents,out_file, protocol=-1)
        pickle.dump(corpus,out_file, protocol=-1)
        pickle.dump(texts,out_file, protocol=-1)
else:
    with open(pickle_file_name, 'rb') as in_file:
        model = pickle.load(in_file)
        vocab = pickle.load(in_file)
        inverse_vocab = pickle.load(in_file)
        number_of_documents = pickle.load(in_file)
        corpus = pickle.load(in_file)
        texts = pickle.load(in_file)
        
        #print_LDA_topics(model,inverse_vocab,15)
        #print(model.doc_topic_[:,0].argsort()[-1])
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(model.doc_topic_)
        max_topic_id = model.doc_topic_.argmax(axis=1)
        fig, ax = plt.subplots()
        #plt.figure(figsize=(12,12))
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=max_topic_id, cmap = 'tab20')
        directory = os.fsencode('dataset/Mayo/Description')
        cnt=0
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            ax.text(tsne_results[cnt, 0], tsne_results[cnt, 1], filename, horizontalalignment='left', size='medium', color='black', weight='semibold')
            cnt=cnt+1
        plt.title('TSNE of LDA colore by topic')
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.show()



    #words=f.read().split()
    #for word in words:
        #vocab[word]+=1
#print(vocab)
#total_words=sum(vocab.values())
#common=vocab.most_common(20)
#ratio = {k: (v*100 / total_words) for k, v in common}
#print('Number of unique words in wikipedia is {0:d}'.format(len(vocab)))
#print('Total number of words is {0:d}'.format(total_words))
#print(ratio)
#plt.hist(vocab.values(), bins=20,log=True)
#plt.show()

