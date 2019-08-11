#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import locale
from nltk.corpus import stopwords

def quita_stopwords(lista_w):
   lista_stopwords =stopwords.words('spanish')
   lista_w_procesada = []
   nuevas_mias=('gracias', 'hola', 'saludos', 'feliz', 'buenas', 'días',
                'muchas', 'día', 'tardes', 'ok', 'buenos', '-', 'q', 'si', 'saludo')
   for i in nuevas_mias:
    lista_stopwords.append(i)
   for w in lista_w:
      w = w.lower()
      w = w.replace('\n', '')
      w = w.replace(',', '')
      w = w.replace('.', '')
      w = w.replace('?', '')
      w = w.replace('¿', '')
      w = w.replace('!', '')
      w = w.replace('(', '')
      w = w.replace(')', '')
      w = w.replace('\\','')
      w = w.replace(';','')
      w = w.rstrip()
      w = w.lstrip()
      if w not in lista_stopwords:
         lista_w_procesada.append(w)
   return lista_w_procesada

    
#dir = "/Users/edugallardopardo/OneDrive/Documentos/TFM/CHATS/ANONIMIZADOS/"
dir = 'C:\\Users\\edgal\\OneDrive\\Documentos\\TFM\\CHATS\\ANONIMIZADOS\\'

lista_palabras=[]
lista_dicc=[]
txtfrases = open('frases.txt', 'w', newline='', encoding='utf-8')

for file in os.listdir( dir ): # listo ficheros del directorio
   if file.endswith( ".csv" ): # que sean del tipo csv
      f = open(dir+file,"r", encoding='utf-8') # abro fichero csv codificándolo como utf-8
      for cnt, line in enumerate(f): # recorro líneas de cada fichero
         if cnt > 1: # no contemplo la cabecera del csv y la primera línea propia de whatsapp
            campos_leer=[] # lista de campos de cada línea del fichero tratado
            campos_leer = line.split(',%') # separo csv, ahora me interesa el campo de diálogo
            if len(campos_leer) >= 5:
               #print('5 '+str(len(campos_leer)))
               frase = campos_leer[4].replace('%','')
               txtfrases.write(str(frase))
            else:
               #print('OTROS '+file+' '+str(len(campos_leer))+' '+campos_leer[0]+' '+campos_leer[1])
               frase =''
               clasif_tipo = campos_leer[0]
               clasif_tipo = clasif_tipo.replace('\n', '')

            if len(frase) > 0: # porque también se procesa la línea de los chats con la clasificación
               frase = quita_stopwords(frase.split())  # Quitamos las stopwords
               for palabra in frase: # Procesamos cada palabra del campo en cuestión
                  existe = False
                  for elem in lista_dicc: #Si ya existe en la lista le sumo 1 al contador
                     if elem['palabra']==palabra:
                        existe = True
                        elem['count']=elem['count']+1
                  if not existe: #Si no existe lo creo en el diccionario
                     diccionario = {}
                     diccionario['palabra'] = palabra.lower()
                     diccionario['tipo'] = clasif_tipo.lower()
                     diccionario['count'] = 1
                     lista_dicc.append(diccionario)  

               #lista_palabras.append(palabra) # En desuso
      f.close() # cierro chat

# Ordeno la lista-diccionario por el valor descendente del campo count
diccionario_ocurrencias_ordenados = sorted(lista_dicc, key=lambda k: k['count'], reverse=True)

# Generación de la salida para análisis 
csvsalida = open('palabras_count.csv', 'w', newline='', encoding='utf-8')
csvsalida.write('Palabra,Tipo,Count\n')
for elem in diccionario_ocurrencias_ordenados:
   print (str(elem['palabra'])+','+str(elem['tipo'])+','+str(elem['count']))
   csvsalida.write(str(elem['palabra'])+','+str(elem['tipo'])+','+str(elem['count'])+'\n')

csvsalida.close()
txtfrases.close()
print ('Proceso finalizado con éxito, generados frases.txt y palabras_count.csv')
   
