#!/usr/bin/env python
# coding: utf-8



import os
import locale
import spacy
import numpy as np
nlp = spacy.load('es_core_news_md')
fich="C:\\Users\\edgal\\OneDrive\\Documentos\\TFM\\python\\SPACY\\frases_user.txt"



documentos =[]
with open(fich, 'r', encoding='utf-8') as file:
    for linea in file:
        doc = nlp(linea)
        documentos.append(doc.text) 



def map_chars_to_tokens(doc):
    """
    Creates a mapping from input characters to corresponding input tokens

    For instance, given the input:

    Nuclear theory ...
    |||||||||||||||
    012345678911111...
              01234

    it returns an array of size equal to the number of input chars plus one,
    whcih looks like this:

    000000011111112...

    This means that the first 7 chars map to the first token ("Nuclear"),
    the next 7 chars (including the initial whitespace) map to the second
    token ("theory") and so on.
    """
    #n_chars = len(doc.text_with_ws)
    n_chars = len(doc)
    char2token = np.zeros(n_chars + 1, 'int')
    start_char = 0
    #for token in doc:
    for token in doc.split():
        end_char = token.idx + len(token)
        char2token[start_char:end_char] = token.i
        start_char = end_char
    char2token[-1] = char2token[-2] + 1
    return char2token 





text="¿Cuál autobús llega hasta el Tanatorio Municipal?  Gracias."



TRAIN_DATA = []
for doc in documentos:
    ini=0
    fin=0
    entities = []
    for token in doc.split():
        for i, w in enumerate(token, start=ini):
            fin=fin+1
        #print(token, ini, fin)
        entities.append((ini, fin, token))
        ini=fin+1
        fin=ini
    TRAIN_DATA.append((doc, {'entities': entities}))




TRAIN_DATA = []
for doc in documentos:
    doc=nlp(doc)
    print("------------------------------")
    print(doc.text)
    vector=map_chars_to_tokens(doc)
    entities = []
    for i in range(0,len(doc)):
        subvector=np.where(vector == i)
        if subvector:
            entities.append((min(subvector[0]), max(subvector[0])+1, doc[i]))
            print(subvector)
            print(doc[i], minimo, max(subvector[0])+1)
    TRAIN_DATA.append((doc.text, {'entities': entities}))
    print("------------------------------")




#dir = 'C:\\Users\\edgal\\OneDrive\\Documentos\\TFM\\PYTHON\\SPACY\\'
traintxt = open('train_data_txt.txt', 'w', newline='', encoding='utf-8')
for linea in TRAIN_DATA:
    #print(str(linea)+',\n')
    traintxt.write(str(linea)+',\n')
traintxt.close()


# In[ ]:
