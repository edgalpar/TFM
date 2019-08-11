#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import locale
import csv
def getpreferredencoding(do_setlocale = True):
   return "utf-8"
locale.getpreferredencoding = getpreferredencoding


dir = "/Users/edugallardopardo/OneDrive/Documentos/TFM/CHATS/"
dir_dest = "/Users/edugallardopardo/OneDrive/Documentos/TFM/CHATS/ANONIMIZADOS/"
#dir = 'C:\\Users\\edgal\\OneDrive\\Documentos\\TFM\\CHATS\\'
#dir_dest = 'C:\\Users\\edgal\\OneDrive\\Documentos\\TFM\\CHATS\\ANONIMIZADOS\\'


for file in os.listdir( dir ): # listo ficheros del directorio
   if file.endswith( ".csv" ): # que sean del tipo csv
      f = open(dir+file,"r", encoding="utf-8") # abro fichero csv codificándolo como utf-8
      #f_out = open(dir_dest+file, 'w', newline='', encoding="utf-8") # FALTA ANONIMIZAR EL NOMBRE DEL FICHERO
      f_out = open(dir_dest+str(abs(hash(file)))+'.csv', 'w', newline='', encoding="utf-8")
      for cnt, line in enumerate(f): # recorro líneas de cada fichero
         #if cnt > 2: # no contemplo la cabecera del csv y la primera línea propia de whatsapp
            campos_leer=[] # lista de campos de cada línea del fichero tratado
            campos_leer = line.split('","') # separo csv, ahora me interesa el campo de diálogo
            if len(campos_leer) == 5:
               f_out.write(str(campos_leer[0].replace('"','%'))+',%'+str(campos_leer[1].replace('"','%'))
                           +',%'+str(abs(hash(campos_leer[2].replace('"','%'))))+',%'+str(campos_leer[3].replace('"','%'))+',%'
                           +str(campos_leer[4].replace('"','%')))
            elif len(campos_leer) == 4:
               f_out.write(str(campos_leer[0].replace('"','%'))+',%'+str(abs(hash(campos_leer[1].replace('"','%'))))+',%'+
                           str(campos_leer[2].replace('"','%'))+',%'+str(campos_leer[3].replace('"','%')))
            else:
               f_out.write(str(campos_leer[0]))
      f.close()
      f_out.close()
print('Proceso finalizado')
