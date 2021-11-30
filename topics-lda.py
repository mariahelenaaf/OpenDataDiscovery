import pandas as pd
import gensim
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from gensim import corpora
import nltk
import csv 
from pathlib import Path
import matplotlib.pyplot as plt
from nltk.stem import RSLPStemmer
    
def split_corpus(corpus,value):
    textdata = []
    topiclist = []
    position = 0
    for i, word in enumerate(corpus):
        topiclist.append(word)
        position+=1
        if position == value: #split corpus at each "value"
            textdata.append(topiclist)
            topiclist = []
            position = 0
    return textdata

def getCoherenceModel(corpus,dictionary,texts):
    results = []
    for t in range(2,15):
        ldamodel = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, random_state=0, num_topics=t, workers=3)
        coherence_model_lda = CoherenceModel(model=ldamodel,texts=texts,dictionary=dictionary,coherence='c_v')
        score = coherence_model_lda.get_coherence()
        tup = t, score
        results.append(tup)
    results = pd.DataFrame(results, columns=['topic','score'])
    s = pd.Series(results.score.values, index=results.topic.values)
    s.plot()
    plt.show() 


def predict_topic(model,dictionary): 
    print("\nPredicting topics for the question... \n")
    question = ['alunos','cotistas','ingressaram','ensino','superior','turno','noturno']
    bow_vector = dictionary.doc2bow(question)
    for index,score in sorted(model[bow_vector], key=lambda tup:-1*tup[1]):
        print("\nScore: {}\t Topic: {}".format(score,model.print_topic(index,5))) #5 first words from each topic

def execute_lda(index,description):
    tpc = best_n_topics[index] #number of topics according to a source index
    print('\nTopics for ', sources[x], ': ', tpc)
    texts = split_corpus(description,20) #split corpus (source content) at each 20 words (necessary for corpora.Dictionary and id_map.doc2bow)
    id_map=corpora.Dictionary(texts) #dictionary
    corpus = [id_map.doc2bow(value) for value in texts]
    
    #using tpc value
    ldamodel = gensim.models.LdaMulticore(corpus=corpus, id2word=id_map, random_state=42, num_topics=tpc, workers=3)
    
    #using n_topics value (the same number of topics will be generated for all sources)
    #ldamodel = gensim.models.LdaMulticore(corpus=corpus, id2word=id_map, random_state=42, num_topics=n_topics, workers=3)

    #getCoherenceModel(corpus,id_map,texts) #this line plots coherence graphs for each source content
    coherence_model_lda = CoherenceModel(model=ldamodel,texts=texts,dictionary=id_map,coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print("\n***************************")
    print('\nCoherence Score for ', sources[index], ': ', coherence_lda) 
    print('\nPerplexity: ', ldamodel.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    
    #writing the main topics for source description
    for idx, topic in ldamodel.show_topics(): 
        print(idx, '->', topic)
        words=re.sub('[^A-Za-zãàáâõôóéêúíç ]+', '', topic) 
        topics.append(words)
        scores=re.findall(r'\d+\.\d+', topic) 
        #print('SCORES: ', scores)
        topic_scores.append(scores)
        write_word_score(sources[index],words,scores) #THIS LINE GENERATES FILES ON THE 'INPUT' FOLDER (IMPORTANT)
        #write_all_topics(sources[index],topic) #currently not used
    #predict_topic(ldamodel,id_map) #function to predict one topic for a given entry (not used in the current approach)
           
def write_all_topics(source_name,topic): #generates csv containing source name and topics. E.g.: |prouni|0.037*"bolsas" + 0.037*"cursos" + 0.037*"privadas" ...
    pathFile = Path("show-topics.csv")
    if pathFile.is_file():
        myFile = open('show-topics.csv','a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow([source_name,topic])
    else:
        myFile = open('show-topics.csv','w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(["Source","Topic"])
            writer.writerow([source_name,topic])
    myFile.close()

def write_word_score(sourcename,word,score): #generates csv containing source name, LDA topic words, LDA words scores. 
    print('Source name: ', sourcename)
    pathFile = Path('./input/'+sourcename+'.csv')
    if pathFile.is_file():
        myFile = open('./input/'+sourcename+'.csv','a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow([word,score])
    else:
        myFile = open('./input/'+sourcename+'.csv','w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(["TopicWords","Scores"])
            writer.writerow([word,score])
    myFile.close()


def write_preprocessed_source(source_index,text): #generates csv files that will be used for cosine, jaccard and others.
    s = sources[source_index]
    pathFile = Path('./preprocessedsources/'+s+'.csv')
    if pathFile.is_file():
        myFile = open('./preprocessedsources/'+s+'.csv','a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow([s,text])
    else:
        myFile = open('./preprocessedsources/'+s+'.csv','w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(["Source","Text"])
            writer.writerow([s,text])
    myFile.close()

def flatten(li): #function used for removing nested lists in python.
    return sum(([x] if not isinstance(x, list) else flatten(x) for x in li), [])

def preprocessing(v): 
    tokens_list = [] 
    simple_list = [] 
    words = [] 
    phrase = [] 
    if isinstance(v,list):
        for word in sentence:
            preprocessing(word)
    else:
        if isinstance(v,int):
            v = str(v)
        v = v.lower()
        v = re.sub(r"[^a-zA-Z0-9ãàáâõôóéêúíç]+", ' ', v) 
        v = re.sub(r'\b\w{1,3}\b', '', v)
        simple_list.append(v)
        if ' ' in v: 
            phrase = [word_tokenize(i) for i in simple_list] 
    if phrase:
        for v in phrase:
            words = [word for word in v if not word in stopwords] 
            tokens_list.append(words)
        tokens_list = flatten(tokens_list) 
        return tokens_list 
    else:
        return simple_list 

def createSourceAndTopicDict():
    cont = 0
    i=0
    for s in sources:
        for topic in topics[cont:]: 
            if(i==n_topics):
                i = 0
                break
            else:      
                i = i+1
                cont = cont+1
                source_and_topic.setdefault(s,[]) #receber listas como valor
                if s in source_and_topic:
                    source_and_topic[s].append(topic)
                else:
                    source_and_topic[s]=topic

def showSourceAndTopicDict():
    for source, topics in source_and_topic.items():
        print('\n\nSource: ', source)
        print('Topics: ', topics)
        print('****************************')



#**********************************************************
stemmer = RSLPStemmer()
topic_scores=[] #store topic scores under format: [['0.036','0.035',...],['0.042'...]]
sources=[] #store sources names, e.g.: [fies,prouni...]
topics=[] #store LDA topics
source_and_topic = {} #associate source to topics, e.g.: {fies:['word, word, word'], prouni:['word, word, word']...}

#n_topics = 10 #if you want to use 10 topics for all sources in the LDA model
best_n_topics=[8,8,8,8,8,10,10,10] #if you want to use different numbers of topics for each of the eight sources in the LDA model. If you are using 5 sources, modify this array

stopwords = nltk.corpus.stopwords.words('portuguese')
new_words=('dados','microdados', 'sobre', 'tabela', 'cada', 'representam', 'representa',
    'outras', 'entre', 'visualizar', 'informações', 'algum', 'código', 'àquelas', 'três','tipo',
    'possui','sabe','cede','quantos','quantas','quantidade','total') 
for i in new_words:
    stopwords.append(i) #adding new words to stoplist


#inspecting 5 sources descriptions:
#df = pd.read_csv('./sources-descriptions/5-sources.csv', error_bad_lines=False, delimiter ='\t')
#inspecting 8 sources descriptions:
df = pd.read_csv('./sources-descriptions/8-sources.csv', error_bad_lines=False, delimiter ='\t')
diction = df.to_dict(orient='list')
for k, v in diction.items(): 
    if k=='Source':
        for sourcename in v:
            sources.append(sourcename) #sources names
    elif k=='Description':
        for x, y in enumerate(v):
            y = preprocessing(y) #preprocessing for LDA (without stemming)
            execute_lda(x,y) #sending source index and processed description to LDA function
            y = [stemmer.stem(w) for w in y] #stemming is done after LDA to generate preprocessed files
            processed_list = ','.join(y)
            print('\nPreprocessed description: ', processed_list)
            #write_preprocessed_source(x,processed_list) #generates files on 'preprocessedsources' folder
#createSourceAndTopicDict()
#showSourceAndTopicDict()