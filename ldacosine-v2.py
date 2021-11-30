import pandas as pd
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from gensim import corpora
import nltk
import csv 
from pathlib import Path
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle5 as pickle
import os
from gensim.models.word2vec import Word2Vec
import pyexcel
import difflib
from scipy.stats import hypergeom
import math
from collections import Counter
from nltk.stem import RSLPStemmer

def w2v_extend_topics(topics, number_of_words, w2v_similarity): #this function adds W2V's similar words according to LDA's words
   topic_count= len(topics)
   newTopics = [None] * topic_count
   print('\nReading Word2Vec model')
   model = read_word2vec_format() 

   print('\nExtending results')
   for i in range(0, topic_count):
        newTopics[i] = []
        for j in range(0, number_of_words):
            word_topic = topics[i][j] #word_topic = [probability,word] from LDA
            words = get_similar_words(model, word_topic[1], w2v_similarity, word_topic[0])
            #print('\n\nRETURNED WORDS: ', words)
            if  (len(words) > 0): #if similar words were found
                newTopics[i].extend(words)
            else:
                newTopics[i].extend([[word_topic[1],word_topic[0],word_topic[1]]]) #if similar words were not found, the original word from LDA is included
                #print('\nOriginal word: ', word_topic[1], '->',word_topic[0])
        for j in range(number_of_words, len(topics[i])):
            newTopics[i].append([topics[i][j][1],topics[i][j][0],topics[i][j][1]])
   return newTopics

def get_similar_words(model, word, w2v_similarity, lda_word_probability):
    lda_word_probability = float(lda_word_probability)
    words = []
    try:
        all_words = model.similar_by_vector(model[word], topn=50)
        for i in all_words:
            if (i[1] >= w2v_similarity): #only similar words above a certain threshold will be included 
                words.append([word, lda_word_probability*i[1], i[0]])
            else:
                return words
        return words
    except KeyError:
        return words

def read_word2vec_format():
    if os.path.isfile('model.pkl'):
        print ("File \'model.pkl\' exist, reading it!")
        model = pickle.load(open('model.pkl', 'rb'))
    else:
        print('reading load_word2vec_format(cc.pt.300.vec)')
        model = KeyedVectors.load_word2vec_format('/home/maria/testes-bases-dados-abertos/cc.pt.300.vec', limit=20000, binary=False, unicode_errors='ignore')
        pickle.dump(model, open('model.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    return model

def save_word2vec_topics(extended_topics, w2v_output,source): 
   topic_count= len(extended_topics)
   for i in range(0, topic_count):
       class_file_name=w2v_output+'class'+source
       pyexcel.save_as(array=([['basic word', 'probability', 'word form Word2Vec']]+extended_topics[i]), dest_file_name=class_file_name)

def create_topics_list(topicWords,wordScores):
    values=[]
    all_values=[]
    all_topics=[]
    for topic, score in zip(topicWords,wordScores):
        topic = re.sub(' +', ' ', topic) #remove double spaces
        topic = list(topic.split(" "))
        for word, value in zip(topic,score):
            value = value.strip('"')
            value = value.strip("'")
            values=[value,word]
            all_values.append(values)
    all_topics.append(all_values)
    return all_topics

def flatten(li): #function used for removing nested lists in python.
    return sum(([x] if not isinstance(x, list) else flatten(x) for x in li), [])

def preprocessing2(v):
    stemmer = RSLPStemmer() 
    tokens_list = [] 
    simple_list = [] 
    words = [] 
    phrase = [] 
    stopwords = nltk.corpus.stopwords.words('portuguese')
    new_words=('dados','microdados', 'sobre', 'tabela', 'cada', 'representam', 'representa',
        'outras', 'entre', 'visualizar', 'informações', 'algum', 'código', 'àquelas', 'três','tipo',
        'possui','sabe','cede','quantos','quantas','quantidade','total','qual','Qual','quais','Quais') 
    for i in new_words:
        stopwords.append(i) 
    if isinstance(v,list):
        for word in v:
            preprocessing2(word)
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

def manualStemming(inputSentence):
    tokens_list=[]
    for w in inputSentence:
        if w.endswith('s'):
            tokens_list.append(w[:-2]) #remove ultimos dois caracteres de palavras no plural
        else:
            tokens_list.append(w)
    return tokens_list

def getRslpStemming(inputSentence):
    stemmer = RSLPStemmer() 
    new=[]
    for w in inputSentence:
        w = stemmer.stem(w)
        new.append(w)
    return new

def getInputAndSourceProbabilities(inputSentence,sourcesDict):
    ldaInput = manualStemming(inputSentence)
    sources_and_probabilities = os.listdir('./output/')
    columns = ['basic word','probability','word form Word2Vec'] #basic word is a LDA topic word
                                        #'word form Word2Vec' is a word extended from the basic word using word2vec
    all_lda_probabilities={}
    lda_and_cosine={}
    usem_dict = {}
    cosine_dict = {}
    jacc_dict = {}
    all_votes=[]
    for source in sources_and_probabilities:
        print('Information for source: ', source)
        similar_words = []
        dataframe = pd.read_csv('./output/'+source,usecols=columns)
        basicword = dataframe['basic word'].tolist()
        probabilities = dataframe['probability'].tolist()
        w2vword = dataframe['word form Word2Vec'].tolist()
        for word in ldaInput: 
            indicesWhereWordOccurs = [i for i, x in enumerate(basicword) if word in x] #getting all 'basic word' indexes where the input word were found
            w2vIndicesWhereWordOccurs = [i for i, x in enumerate(w2vword) if word in x] #getting all 'word from Word2Vec' indexes where the input word were found
            if len(indicesWhereWordOccurs) != 0: #if basic words were found
                similar_words.append(word) 
                bestProbs = []
                for index in indicesWhereWordOccurs:
                    bestProbs.append(probabilities[index]) #get all probabilities where matches were found
                prob = max(bestProbs) #pick the best prob.
            else: 
                if len(w2vIndicesWhereWordOccurs)!=0: #check if 'word form Word2Vec' matches is not empty
                    for i in w2vIndicesWhereWordOccurs:
                        similar_words.append(w2vword[i]) #store basic words corresponding to the indexes
                        similar_words.append(basicword[i])
                        prob = probabilities[i]
                        break; #pick only the first match from the list
                else: #if the input word was not found in any list
                    prob = 0.000001 #assign a minimum probability
            similar_words = list(dict.fromkeys(similar_words)) #removing duplications in list 
            all_lda_probabilities.setdefault(source,[])
            lda_and_cosine.setdefault(source,[])
            all_lda_probabilities[source].append(prob) #each source in dict will be associated with a list of probabilities, one for each input word
            lda_and_cosine[source].append(prob)
        print('Matches found: ', similar_words)
        #jaccard:
        intersection = relaxed_intersection(ldaInput,similar_words)
        jac = jaccardFunction(similar_words,ldaInput,intersection)
        jacc_dict[source]=jac
        
        #cosine:
        inputSentence = getRslpStemming(inputSentence)
        abbr = source[5:]  #from "classCadunico.csv" to "Cadunico.csv" 
        cosine = get_cosine(inputSentence,sourcesDict[abbr])
        print('Cosine: ', cosine)
        cosine_dict[source]=cosine
        
        #U(sem)/josie
        usem = CumulativeDistribution(inputSentence,sourcesDict[abbr],relaxed_intersection(inputSentence,sourcesDict[abbr]))
        usem_dict[source] = usem 

        #lda+cosine:
        lda_and_cosine[source].append(cosine)
        print('******************\n')
    print('\nThe input sentence was: ', inputSentence)
    
    #Choose one or more method of classification:
    #all_votes.append(chooseSource(inputSentence,usem_dict,'Usem')) 
    all_votes.append(chooseSource(inputSentence,cosine_dict,'Cosine')) 
    #all_votes.append(chooseSource(inputSentence,jacc_dict,'Jaccard')) 
    all_votes.append(chooseSourceLDA(inputSentence,all_lda_probabilities,'LDA4'))
    all_votes.append(chooseSourceLDA(inputSentence,lda_and_cosine,'LDACosine')) 
    return all_votes


def majorityVoting(votes,answer,correct_class): #if more than one method is being used, the majority vote choose the most voted data source  
    counting = Counter(votes)
    value, count = counting.most_common()[0]
    print('\nFinal choice: ', value)
    if value in answer:
        correct_class+=1
        print('\nCorrect. Counting: ',correct_class) #counting correct classifications
    return correct_class

def relaxed_intersection(palavra,conjunto):
    intersec_values = []
    similars = []
    for i, v in enumerate(palavra):
        for index, value in enumerate(conjunto):
            if v in value:
                similars.append(value)
                intersec_values.append(v)
                break
    print('Intersection values: ', intersec_values)
    print('Similars found: ', similars)
    return intersec_values

def jaccardFunction(similars_B,question_A,intersec):
    D = len(similars_B+question_A)
    jaccard = len(intersec)/D
    print('Jaccard similarity: ',jaccard)
    return jaccard

def chooseSource(inputSentence,my_dict,method):
    higherSim = 0.0
    chosenS = ''
    for source, sim in my_dict.items():
        if sim > higherSim:
            higherSim = sim
            chosenS = source
    #exportProbabilities(inputSentence,method,my_dict,chosenS)
    print('\nThe chosen source by ', method, ' was: ', chosenS, 'with the highest similarity: ', higherSim)
    return chosenS

def CumulativeDistribution(set1,set2,intersec): #Based on 'Table Union Search' paper (Nargesian,2018)
    a_size = len(set1)
    b_size = len(set2)
    s = len(intersec)
    d_size = a_size + b_size
    print('Tamanho do conjunto A: ', a_size)
    print('Tamanho do conjunto B: ', b_size)
    print('Tamanho da interseccao: ', s)
    print('Tamanho do dominio D: ', d_size)

    x=0
    probabilities = 0
    while x <= s:
        prob = hypergeom.pmf(x,d_size,a_size,b_size)
        probabilities = probabilities + prob      
        x+=1
    print('Usem probability: ', probabilities)
    return probabilities 

def exportProbabilities(inp,method,probList,chosen): 
    pathFile = Path('./exported-probabilities/probs.csv')
    if pathFile.is_file():
        myFile = open('./exported-probabilities/probs.csv','a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow([inp,method,probList,chosen])
    else:
        myFile = open('./exported-probabilities/probs.csv','w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(["InputSentence","Method","Probabilities","ChosenSource"])
            writer.writerow([inp,method,probList,chosen])
    myFile.close()

def chooseSourceLDA(inputSentence,sourcesProb,method):
    higherprob = 0.0
    chosenSource = ''
    #print('\n\n')
    for source, probList in sourcesProb.items():
        average = sum(probList)/len(probList) #probabilities average
        #print('Average probability for source ', source, ' is: ', average)
        if(average > higherprob):
            higherprob = average
            chosenSource = source
    #exportProbabilities(inputSentence,method,sourcesProb,chosenSource)
    print('\nThe chosen source by ', method, ' was: ', chosenSource, 'with the highest average probability: ', round(higherprob,2))
    return chosenSource

def get_cosine(inputSentence,attributes):
    vec1 = Counter(inputSentence)
    vec2 = Counter(attributes)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def getPreprocessedSourceText():
    src_texts={}
    sourcesText = os.listdir('./preprocessedsources/')
    cols = ['Source','Text']
    for i, source in enumerate(sourcesText):
        new_list=[]
        src_texts.setdefault(source,[])
        df = pd.read_csv('./preprocessedsources/'+source,usecols=cols)
        text_list = df['Text'].tolist()
        for value in text_list:
            new_list = (str(value)).split(',')
        src_texts[source] = new_list
    #print('Just checking: ',src_texts)
    return src_texts


def main():
    #extensions_number=100 #use this line if you used n_topics = 10 in the 'topics-lda.py' file.
    #use the following line if you used 'best_n_topics' in the 'topics-lda.py' file:
    extensions_number=[100,80,100,100,80,80,80,80] #extensions for ibama, cadunico, datasus, ibge, fies, censo escolar, ensino superior, prouni, respectively.
    inputSources = os.listdir('./input/')
    print('Sources list: ', inputSources)
    col_list = ['TopicWords', 'Scores']
    for i, source in enumerate(inputSources):
        print('Index: ', i, ' and source: ', source)
        df = pd.read_csv('./input/'+source,usecols=col_list)
        topicWords = df['TopicWords'].tolist()
        wordScores = df['Scores'].tolist()
        wordScores = [i.strip("[]").split(", ") for i in wordScores] #convert list of strings to list of lists
        topics_list = create_topics_list(topicWords,wordScores)
        extended_topics = w2v_extend_topics(topics_list, extensions_number[i], 0.45) #extensions_number[i]->number of topic words that will be extended, 0.45--> similarity threshold
        #print('\n\nEXTENDED TOPICS for source ', source, ': ', extended_topics)
        
        #save_word2vec_topics(extended_topics, './output/', source) #THIS LINE GENERATES THE EXTENSIONS INSIDE 'OUTPUT' FOLDER. You can comment this line if you already have files inside 'output' folder.
    
    votes=[]
    correct_classifications=0
    sources_texts = getPreprocessedSourceText()
    questionsFile = open('./test-set/questions.txt','r') #48 questions
    #questionsFile = open('./test-set/all-questions.txt','r') #74 questions
    answersFile = open('./test-set/answers-48.txt','r') #48 answers
    #answersFile = open('./test-set/answers-74.txt','r') #74 answers
    questions = questionsFile.readlines()
    answers = answersFile.readlines()
    for q, a in zip(questions,answers):
        inputSentence = preprocessing2(q)
        print('\nPreprocessed input question: ', inputSentence)
        votes = getInputAndSourceProbabilities(inputSentence,sources_texts)
        correct_classifications = majorityVoting(votes,a,correct_classifications)
        input("Press Enter to continue...") #analyze classification question by question
    print('\nCorrect classifications: ',correct_classifications)


if __name__ == "__main__":
    main()
