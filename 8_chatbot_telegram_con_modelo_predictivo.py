#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
from telegram.ext import CommandHandler
from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters
import spacy
import geocoder


# In[2]:


#nlp=spacy.load("/Users/edugallardopardo/OneDrive/Documentos/TFM/python/SPACY/modelos/EMT_models/emtrutas_1")
nlp=spacy.load("/Users/edugallardopardo/OneDrive/Documentos/TFM/python/SPACY/modelos/EMT_models/model5")


# In[3]:


def main():
 
    # Creamos el Updater, objeto que se encargara de mandarnos las peticiones del bot
    # Por supuesto no os olvideis de cambiar donde pone "TOKEN" por el token que os ha dado BotFather
    updater = Updater("661606549:AAF4e_TE_4L-Wz1tbmazcsqL2sspgLqoLj4")

    # Cogemos el Dispatcher, en el cual registraremos los comandos del bot y su funcionalidad
    dispatcher = updater.dispatcher

    # Registramos el metodo que hemos definido antes como listener para que muestre la informacion de cada mensaje
    listener_handler = MessageHandler(Filters.text, listener)
    dispatcher.add_handler(listener_handler)

    # Ahora registramos cada metodo a los comandos necesarios
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("holamundo", hola_mundo))
    dispatcher.add_handler(CommandHandler("logo", logo))

    # Y comenzamos la ejecucion del bot a las peticiones
    updater.start_polling()
    updater.idle()


# In[4]:


# Metodo que imprimira por pantalla la informacion que reciba
def listener(bot, update):
    id = update.message.chat_id
    mensaje = update.message.text
    origen = None
    destino = None
    print("ID: " + str(id) + " MENSAJE: " + mensaje)
    doc=nlp(mensaje)
    print("Entidades", [(ent.text, ent.label_) for ent in doc.ents])
    for ent in doc.ents:
        bot.sendMessage(chat_id=update.message.chat_id, text=str(ent.label_)+': '+str(ent.text))
    """for ent in doc.ents:
        if ent.label_=='LUGAR1':
            origen=ent.text
        elif ent.label_=='LUGAR2':
            destino=ent.text
    if origen is None and destino is None:
        bot.sendMessage(chat_id=update.message.chat_id, text='Repíteme por favor el origen y el destino')
    elif origen is None and destino is not None:
        bot.sendMessage(chat_id=update.message.chat_id, text='Repíteme por favor el origen')
    elif origen is not None and destino is None:
        bot.sendMessage(chat_id=update.message.chat_id, text='Repíteme por favor el destino')
    else:
        origen = origen + ', València'
        destino = destino + ', València'
        loc1 = geocoder.osm(origen)
        loc2 = geocoder.osm(destino)
        print(str(loc1.latlng))
        print(str(loc2.latlng))
        bot.sendMessage(chat_id=update.message.chat_id, text='Hola, entonces quiere ir desde {} hasta {}'.format(origen, destino))
        bot.sendMessage(chat_id=update.message.chat_id, text="Hola, pincha aquí a ver 😍 -> http://www.emtvalencia.es/geoportal/?from={},{}&to={},{}&mode=BUSISH,WALK&usuario=Anonimo".format(loc1.latlng[1], loc1.latlng[0], loc2.latlng[1],loc2.latlng[0]))"""


# In[5]:


# MÈtodo que utilizaremos para cuando se mande el comando de "start"
def start(bot, update):
    bot.sendMessage(chat_id=update.message.chat_id, text='Bienvenido al bot de prueba cálculo de rutas!!!')
    bot.sendPhoto(chat_id=update.message.chat_id, photo=open('/Users/edugallardopardo/OneDrive/Documentos/TFM/telegram/icono.png', 'rb'))
    bot.sendMessage(chat_id=update.message.chat_id, text='Ejemplo de funcionamiento: Para ir desde Manuel Candela 8 a cines yelmo que linea debo coger?')


# In[6]:


# Metodo que mandara el mensaje "xxxxxx"
def hola_mundo(bot, update):
    bot.sendMessage(chat_id=update.message.chat_id, text='Buenas tardes, la ruta que te recomiendo es: http://www.emtvalencia.es/geoportal/?from=-0.35641638165542705,39.4887764016682&to=-0.3769174014172654,39.47654131629799&mode=BUSISH,WALK&usuario=Anonimo')


# In[7]:


# Metodo que mandara el logo de la pagina
def logo(bot, update):
    # Enviamos de vuelta una foto. Primero indicamos el ID del chat a donde
    # enviarla y despues llamamos al metodo open() indicando la dende se encuentra
    # el archivo y la forma en que queremos abrirlo (rb = read binary)
    id = update.message.chat_id
    print("ID: " + str(id))

    bot.sendPhoto(chat_id=update.message.chat_id, photo=open('/Users/edugallardopardo/OneDrive/Documentos/TFM/telegram/icono.png', 'rb'))


# In[ ]:


# Llamamos al metodo main para ejecutar lo anterior
if __name__ == '__main__':
    main()


# In[ ]:




