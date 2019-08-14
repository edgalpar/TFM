#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from spacy.util import minibatch, compounding
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


# training data
TRAIN_DATA = [('¿Cuál autobús llega hasta el Tanatorio Municipal?  Gracias.\n', {'entities': [(1, 13, "RUTAS"), (14, 19, "RUTAS"), (20, 25, "DESTINO"), (29, 48, "LUGAR2")]}),
('Buenas tardes....a que hora pasa el primer autobús por avda. Constitución-Salesianos\n', {'entities': [(23, 27, "ESTIMACION"), (31, 37, "ATRIBUTO"), (50, 79, 'LUGAR1')]}),
('Cuanto le queda al bus número 19 para llegar a la parada 722? Muchas gracias 😘\n', {'entities': [(0, 6, "ESTIMACION"), (30, 32, "LINEA"), (38, 44, "ESTIMACION"), (57, 60, "PARADA"), (77, 78, "SMILEKISS")]}),
('Parada 710 línea 79 Tiempo de.espera por favor\n', {'entities': [(7, 10, "PARADA"), (17, 19, "LINEA"), (20, 26, "ESTIMACION"), (27, 36, "ESTIMACION")]}),
('Parada 1272 línea 92, tiempo de espera ir favor\n', {'entities': [(7, 11, "PARADA"), (18, 20, "LINEA"), (22, 28, "ESTIMACION"), (32, 38, "ESTIMACION")]}),
('Parada 226 líneas 18 y.90. Tiempo de espera por  favor\n', {'entities': [(7, 10, "PARADA"), (18, 20, "LINEA"), (23, 25, "LINEA"), (27, 33, "ESTIMACION"), (37, 43, "ESTIMACION")]}),
('Buenos dias, anoche hice una recarga y esta mañana no estaba cargado en la tarjeta( la he pasado por el activador)...Número de tarjeta Móbilis: 118692521104', {'entities': [(29, 36, "VENTAONLINE"), (61, 68, "VEMTAONLINE"), (145, 157, "TARJETA")]}),
('442 67\n', {'entities': [(0, 3, "PARADA"), (4, 6, "LINEA")]}),
('La Parada 1386 funciona con normalidad para la línea 99?\n', {'entities': [(10, 14, "PARADA"), (28, 38, "INCIDENCIAS"), (53, 55, "LINEA")]}),
('Quería consultarles para llegar al cabañal desde el cc arena\n', {'entities': [(7, 19, "RUTAS"), (25, 31, "ORIGEN"), (35, 42, "LUGAR1"), (43, 48, "DESTINO"), (52, 60, "LUGAR2")]}),
('Hora de llegada número 35 parada 1951\n', {'entities': [(0, 4, "ESTIMACION"), (8, 15, "ESTIMACION"), (23, 25, "LINEA"), (26, 32, "ESTIMACION"), (33, 37, "PARADA")]}),
('Buenas noches, para ir a CE monteolivete que autobus he de coger desde nuevo centro?\n', {'entities': [(15, 24, "RUTAS"), (28, 40, "LUGAR2"), (41, 52, "RUTAS"), (59, 64, "RUTAS"), (65, 70, "ORIGEN"), (71, 83, "LUGAR1")]}),
('para ir a CE monteolivete que autobus he de coger desde nuevo centro?\n', {'entities': [(0, 7, "RUTAS"), (8, 9, "DESTINO"), (10, 25, "LUGAR2"), (44, 49, "RUTAS"), (50, 55, "ORIGEN"), (56, 68, "LUGAR1")]}),
('Queria información sobre el EMT Jove\n', {'entities': [(7, 18, "INFORMACION"), (28, 36, "BONO")]}),
('Buenas tardes. Acabo de intentar gastar el bono recargado y me dice...... 1 ticket de la reserva.\\nY no funciona. El conductor me ha indicado que no funciona bien. Que me retorneis la recarga y que compre una tarjeta nueva.\n', {'entities': [(43, 57, "VENTAONLINE"), (101, 112, "INCIDENCIAS"), (171, 180, "RECLAMACION"), (184, 191, "VENTAONLINE")]}),
('Buenas tardes,necesito ir de periodista Gil Sumbila a la cruz roja\n', {'entities': [(15, 25, "RUTAS"), (26, 28, "ORIGEN"), (29, 51, "LUGAR1"), (52, 53, "DESTINO"), (57, 66, "LUGAR2")]}),
('Hola buenas tardes me podrías decir a que hora pasa la línea 14 horno de alcedo  me encuentro en el cementerio de sedavid .gracias\n', {'entities': [(42, 46, "ESTIMACION"), (47, 51, "ESTIMACION"), (61, 63, "LINEA"), (100, 121, "LUGAR")]}),
('Buenos días desde Malvarrosa que línea tengo que coger para ir a Alameda 12\n', {'entities': [(12, 17, "ORIGEN"), (18, 28, "LUGAR1"), (33, 38, "INFORMACION"), (39, 48, "RUTAS"), (55, 62, "RUTAS"), (63, 64, "DESTINO"), (65, 72, "LUGAR2")]}),
('Para ir a la boca del metro de Ayora desde la calle doctor Vicente pallarés iranzo 4\n', {'entities': [(0, 7, "RUTAS"), (8, 9, "DESTINO"), (23, 36, "LUGAR2"), (37, 42, "ORIGEN"), (52, 84, "LUGAR1")]}),
('Estoy en plaza España donde puedo coger el 60 que va ala quad?\n', {'entities': [(0, 8, "ORIGEN"), (9, 21, "LUGAR1"), (34, 39, "RUTAS"), (43, 45, "LINEA"), (50, 52, "DESTINO")]}),
('para ir a malvarrosa tanbien me vale si lo coo en periodista llorente?\n', {'entities': [(0, 7, "RUTAS"), (8, 9, "DESTINO"), (10, 20, "LUGAR1"), (47, 49, "ORIGEN"), (50, 69, "LUGAR1")]}),
('Hola, me he dejado olvidada una bolsa en el bus a mediodia\n', {'entities': [(12, 18, "OBJETOS"), (19, 27, "OBJETOS"), (32, 37, "QUE")]}),
('Q autobús puedo cojer desde p.reig -bilbao a piscina de valencia?gracias\n', {'entities': [(10, 21, "RUTAS"), (22, 27, "ORIGEN"), (28, 34, "LUGAR1"), (43, 44, "DESTINO"), (45, 72, "LUGAR2")]}),
('Buenos días.q autobús cojo desde la estación del norte a la estación de autobuses?\n', {'entities': [(22, 26, "RUTAS"), (27, 32, "ORIGEN"), (36, 54, "LUGAR1"), (55, 56, "DESTINO"), (60, 81, "LUGAR2")]}),
('GraciasBuenas tardes.\\nQue linea de bus puedo cojer desde Plz. Ayuntamiento para ir a la exposición del ninot en la Cdad. de las Artes?\n', {'entities': [(46, 51, "RUTAS"), (52, 57, "ORIGEN"), (58, 75, "LUGAR1"), (76, 83, "RUTAS"), (84, 85, "DESTINO"), (116, 134, "LUGAR2")]}),
('Buenas tardes.\\nPara ir de Plza España a la Escuela Oficial de Idiomas (llano de Zaidia) que linea de bus puedo cojer?\n', {'entities': [(21, 23, "RUTAS"), (24, 26, "ORIGEN"), (27, 38, "LUGAR1"), (39, 40, "DESTINO"), (44, 70, "LUGAR2")]}),
('Hablé con ustedes el viernes 5 de abril porque perdí mi bufanda en uno de sus autobuses de EMT\n', {'entities': [(47, 52, "OBJETOS"), (56, 63, "QUE"), (78, 87, "DONDE")]}),
('Quería saber cómo puedo llegar a la Av. de Jesús Morante Borrás, 214 desde el centro de Valencia\n', {'entities': [(7, 12, "INFORMACION"), (24, 30, "RUTAS"), (31, 32, "DESTINO"), (36, 68, "LUGAR2"), (69, 74, "ORIGEN"), (78, 96, "LUGAR1")]}),
('Gracias.Buenos días!  Me gustaría recibir información por favor. Si llego a la estación del norte tengo una linea de autobús que me lleve al hospital general.  Muchas gracias\n', {'entities': [(34, 53, "INFORMACION"), (68, 73, "RUTAS"), (79, 97, "LUGAR1"), (132, 137, "DESTINO"), (141, 157, "LUGAR2")]}),
('Hola para ir desde la calle doctor sumsi hasta la estación de autobuses? Gracias!\n', {'entities': [(5, 12, "RUTAS"), (13, 18, "ORIGEN"), (22, 40, "LUGAR1"), (41, 46, "DESTINO"), (50, 71, "LUGAR2")]}),
('Me parece una vergüenza que no dispongas de cambio de 10€ y me toque ir andando a trabajar. Un sábado a las 16.45h. el número 28 (5146)\n', {'entities': [(14, 23, "QUEJA"), (44, 50, "QUE"), (126, 128, "LINEA")]}),
('Hola, hoy hay huelga?\n', {'entities': [(14, 20, "INCIDENCIAS")]}),
('Gracias.Bueno días , quería informaros de que últimamente el 62 se está retrasando y no cumple su horario. Cada día llega más tarde. Siempre ha llegado a las 7:50 y ahora llega a las 8:05-8:10. Y el anterior a este pasa a las 7:40, con unos 20 minutos entre bus y bus. Por favor que pasen mas buses\n', {'entities': [(21, 38, "RECLAMACION"), (46, 57, "ATRIBUTO"), (61, 63, "LINEA"), (72, 82, "QUE"), (85, 94, "ATRIBUTO")]}),
('Buenos dias, algun problema con el 26? En la parada 1137 no ha pasado todavia el de las 6.40\n', {'entities': [(19, 27, "INCIDENCIAS"), (34, 37, "LINEA"), (52, 56, "PARADA"), (70, 77, "QUE")]}),
('Buenas tardes.Para desplazarme  desde Benicalap ( Avenidas Burjasot - Peset Aleixandre - General Avilés- Campanar ) hasta la C / Santiago Ruiseñol ( Junto Campo Fútbol \\Levante\\ ), ¿ qué puedo hacer ?. Preferiblemente, sin tener que hacer transbordo si no hay que andar mucho. En caso contrario, sin problema en hacer transbordo. \\nGracias y saludos.\n', {'entities': [(19, 30, "RUTAS"), (33, 37, "ORIGEN"), (38, 47, "LUGAR1"), (116, 121, "DESTINO"), (129, 146, "LUGAR2"), (149, 154, "ATRIBUTO"), (155, 177, "LUGAR3")]}),
('La Parada Primat Reig - Ministre Luís Mayans no aparece como tal. ¿ Se tratará de un error involuntario ?. Gracias y saludos.\n', {'entities': [(3, 9, "RECLAMACION"), (10, 44, "PARADA"), (45, 55, "QUE"), (85, 90, "INCIDENCIAS")]}),
('Buenas tardes.\\nPara desplazarme  desde Benicalap ( Avenidas Burjasot - Peset Aleixandre - General Avilés- Campanar ) hasta la Plz. Conde de Carlet 3, ¿ qué puedo hacer ?. Preferiblemente, sin tener que hacer transbordo si no hay que andar mucho. En caso contrario, sin problema en hacer transbordo. \\nGracias y saludos.\n', {'entities': [(21, 32, "RUTAS"), (35, 39, "ORIGEN"), (40, 49, "LUGAR1"), (118, 123, "DESTINO"), (127, 149, "LUGAR2")]}),
('Buenos días. Para desplazarme desde Benicalap  ( Avenidas Burjasot - Peset Aleixandre - General Avilés- Campanar ) hasta la Avda. Pío Baroja 12 \n', {'entities': [(18, 29, "RUTAS"), (30, 35, "ORIGEN"), (36, 45, "LUGAR1"), (115, 120, "DESTINO"), (124, 143, "LUGAR2")]}),
('Llevo esperando 10 min al 93 en la segunda parada desde donde hace descanso e. la avenida del cid, iba a pasar el 93 en 5 min y pone ahora que se va a cochera y me marca 7 minutos\n', {'entities': [(6, 15, "RECLAMACION"), (16, 22, "TEMP"), (26, 28, "LINEA"), (151, 158, "QUE")]}),
('Si, quería saber en la parada 723, el bus 19, cuando dice próximo cuanto tiempo hay que esperar? Lo comento porque indicaba próximo esta mañana y espere 10 minutos, gracias\n', {'entities': [(4, 16, "INFORMACION"), (30, 33, "PARADA"), (42, 44, "LINEA"), (46, 52, "ESTIMACION"), (58, 65, "ESTIMACION"), (73, 79, "ESTIMACION"), (88, 95, "INCIDENCIAS")]}),
('Buenos dias me gustaria saber si la linea 32 para en la plaza de tetuan gracias\n', {'entities': [(24, 29, "RUTAS"), (42, 44, "LINEA"), (45, 49, "ITINERARIO"), (56, 71, "LUGAR1")]}),
('Buenos días. Deseo ir desde la Patacona al Hospital Arnau de Vilanova\n', {'entities': [(13, 21, "RUTAS"), (22, 27, "ORIGEN"), (28, 39, "LUGAR1"), (40, 42, "DESTINO"), (43, 69, "LUGAR2")]}),
('Hola buenos días. Un teléfono de contacto de objetos perdidos tenéis? He extraviado la cartera, por saber si la habrían llevado allí\n', {'entities': [(21, 29, 'CONTACTO'), (33, 41, 'CONTACTO'), (53, 61, 'OBJETOS'), (73, 83, 'OBJETOS'), (87, 94, 'QUE'), (128, 132, 'EMT')]}),
('Hola buenos días. Por favor para ir de peset Aleixandre a calle Albacete, que autobús puedo coger?\n', {'entities': [(28, 35, 'RUTAS'), (36, 38, 'ORIGEN'), (39, 55, 'LUGAR1'), (56, 57, 'DESTINO'), (58, 73, 'LUGAR2')]}),
('Muchas gracias. Igualmente Cuando pasa el próximo 99 en la parada 1272\n', {'entities': [(27, 38, 'ESTIMACION'), (50, 52, 'LINEA'), (68, 70, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 1272\n', {'entities': [(0, 11, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 43, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(0, 11, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(0, 11, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(0, 11, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(7, 11, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(7, 11, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(7, 11, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 89 en la parada 1679\n', {'entities': [(7, 11, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 43, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(0, 6, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 514\n', {'entities': [(0, 6, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 514\n', {'entities': [(0, 6, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 514\n', {'entities': [(0, 6, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 514\n', {'entities': [(0, 6, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 514\n', {'entities': [(0, 6, 'ESTIMACION'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Autobus de Angel guimerà a Hospital Clinico\n', {'entities': [(8, 10, 'ORIGEN'), (11, 24, 'LUGAR1'), (25, 26, 'DESTINO'), (27, 43, 'LUGAR2')]}),
('Buenas tardes, me gustaria saber la localización  de objetos perdidos. Creo que perdi mi bonobus en un autobús  y me gustaria saber si esta alli;\n', {'entities': [(36, 48, 'CONTACTO'), (52, 59, 'OBJETOS'), (60, 68, 'OBJETOS'), (79, 84, 'OBJETOS'), (88, 95, 'QUE'), (138, 142, 'EMT')]}),
('Buenos dias, me gustaria preguntar por una tarjeta perdida en la linea 71, el viernes  a las 14:30h.;\n', {'entities': [(25, 38, 'INFORMACION'), (43, 50, 'QUE'), (51, 58, 'OBJETOS'), (71, 73, 'LINEA')]}),
('Gracias!;Hola buenos días que autobús va al local Jerusalémpop&Rock desde el ayuntamiento\n', {'entities': [(38, 40, 'RUTAS'), (41, 43, 'DESTINO'), (50, 59, 'LUGAR1'), (68, 73, 'DESTINO'), (77, 89, 'LUGAR2')]}),
('Buenos días. Para no faltar a la costumbre, la app no da los horarios desde Poeta Querol bus 26 hacia Moncada\n', {'entities': [(47, 50, 'QUE'), (51, 53, 'QUEJA')]}),
('Buenas tardes...¿¿Como puedo ir desde la calle Salvador Giner a la calle San Juan de la Peña??\n', {'entities': [(29, 31, 'RUTAS'), (32, 37, 'ORIGEN'), (41, 61, 'LUGAR1'), (62, 63, 'DESTINO'), (73, 92, 'LUGAR2')]}),
('Para ir a port saplaya que bus cojo?\n', {'entities': [(0, 7, 'RUTAS'), (8, 9, 'DESTINO'), (10, 22, 'LUGAR2')]}),
('Hola Quiero ir a la calle San Vicente,  número 312, estoy en la plaza los pinazo\n', {'entities': [(5, 14, 'RUTAS'), (15, 16, 'DESTINO'), (26, 50, 'LUGAR2'), (52, 57, 'ORIGEN'), (64, 80, 'LUGAR1') ]}),
('Quiero ir a peset aleixandre, 110\n', {'entities': [(7, 9, 'RUTAS'), (10, 11, 'DESTINO'), (12, 33, 'LUGAR2')]}),
('Estoy en la calle jativa\n', {'entities': [(0, 5, 'ORIGEN'), (18, 24, 'LUGAR1')]}),
('Hola buenas estoy en la parada número 2127 y llevo 20 min esperando al 8 ha pasado uno vacío con fuera de servicio y otro a mitad también con fuera de servicio\n', {'entities': [(12, 17, 'ORIGEN'), (38, 42, 'PARADA'), (58, 67, 'QUEJA'), (71, 72, 'LINEA')]}),
('Hay huelga o algo por el estilo?\n', {'entities': [(4, 10, 'INCIDENCIAS')]}),
('La parada 958 sigue meses sin funcionar\n', {'entities': [(10, 13, 'PARADA'), (26, 39, 'QUEJA')]}),
('Hola. Mi tarjeta es 2400 2616 5947. La he recargado de forma online este mediodía. Se han recargado sin problema en un autobús de la línea 10, y al cogerlo ahora en la línea 9 y pasarlo por los lectores muchas veces, en todas pone que el título está agotado\n', {'entities': [(20, 34, 'TARJETA'), (42, 51, 'VENTAONLINE'), (61, 67, 'VENTAONLINE'), (90, 99, 'VENTAONLINE'), (174, 175, 'LINEA'), (250, 257, 'QUE')]}),
('Estoy detrás de la plaza de toros y voy a coger el 8\n', {'entities': [(0, 5, 'ORIGEN'), (19, 33, 'LUGAR1'), (36, 47, 'DESTINO'), (51, 52, 'LINEA')]}),
('Hola buenas Quiero hacerles un comentario sobre la línea 64 Creo que es la línea donde tratan peor a los seres humanos, realmente los tratan como borregos. Es una línea que siempre va llena con exceso de hecho en bastantes ocasiones hay paradas que ni abre llevando incluso el cartel de COMPLETO y en alguna ocasión ha sucedido esto con 2 bus continuos Sinceramente pienso que harán un estudio de servicio o como se llame en su caso y habrán comprobado que esto sucede. No se han planteado poner esta línea como bus doble No es una línea de paseo o compras al centro. Es una línea con destino LA FE lo que significa que la utilizan usuarios de alguna forma delicados y esto se merece tener un poco de humanidad haciéndoles más fácil el trayecto Gracias por su atención\n', {'entities': [(28, 41, 'RECLAMACION'), (57, 59, 'LINEA'), (87, 98, 'QUEJA'),(146, 154, 'QUEJA')]}),
('Por favor, que bus tengo que coger para ir de calle Yecla a calle Retor 44\n', {'entities': [(40, 42, 'RUTAS'), (43, 45, 'ORIGEN'), (46, 57, 'LUGAR1'), (58, 59, 'DESTINO'), (60, 74, 'LUGAR2')]}),
('Buenos días, necesito ir a Marqués de Lozoya, 8 desde Ángel Guimerá esquina Fdo. El Católico ¿Que autobús o autobuses he de coger? Gracias\n', {'entities': [(22, 24, 'RUTAS'), (25, 26, 'DESTINO'), (27, 47, 'LUGAR2'), (48, 53, 'ORIGEN'), (54, 92, 'LUGAR1')]}),
('Pensé que con un solo bus llegaría aunque ande un poco. Ya q han dicho está x c.c. Saler y allí no se cual va ni dónde se coge\n', {'entities': [(78, 88, 'LUGAR'), (96, 101, 'QUEJA')]}),
('Buenas.Vpel la linea 92.Numero 8012.Direccion av Puerto.Al final del bus hay dos chicas tomando alcohol.Han sacado la botella y se lo han servido y todo es indignante.\n', {'entities': [(21, 23, 'LINEA'), (96, 103, 'INCIDENCIAS'), (156, 166, 'QUEJA')]}),
("Bon dia, volia saber quina linea he d'agafar per anar des de la UPV a l'Edtació del Nord\n", {'entities': [(0, 8, 'SALUDOINIVAL'), (49, 53, 'RUTAS'), (54, 60, 'ORIGEN'), (64, 67, 'LUGAR1'), (68, 69, 'DESTINO'), (70, 88, "LUGAR2")]}),
('Buenas tardes,estamos en la Calle Castellón, para ir a Mestalla q Bus hay q coger. Gracias.\n', {'entities': [(14, 21, 'ORIGEN'), (28, 44, 'LUGAR1'), (50, 52, 'RUTAS'), (53, 54, 'DESTINO'), (55, 63, 'LUGAR2')]}),
('Buenos días, vivimos en la calle Castellón queremos coger un Bus para ir al Museo San Pío V.Gracias.\n', {'entities': [(13, 20, 'ORIGEN'), (27, 42, 'LUGAR1'), (65, 72, 'RUTAS'), (73, 75, 'DESTINO'), (76, 91, 'LUGAR2')]}),
('Buenos días,quisiera ir a la C.Jesus desde la Castellón, gracias.\n', {'entities': [(21, 23, 'RUTAS'), (24, 25, 'DESTINO'), (29, 36, 'LUGAR2'), (37, 42, 'ORIGEN'), (43, 45, 'LUGAR1')]}),
('Hola, sé que hay una línea vuestra que puedo coger en la zona de plaza España y que me acerca a Malilla. Me podéis decir qué línea es? Es que no la recuerdo. Gracias.\n', {'entities': [(65, 77, 'LUGAR1'), (87, 93, 'RUTAS'), (94, 95, 'DESTINO'), (96, 103, 'LUGAR2')]}),
('Que buses puedo coger para ir a la plaza de la virgen o a la calle de la paz?\n', {'entities': [(27, 29, 'RUTAS'), (30, 31, 'DESTINO'), (35, 53, 'LUGAR2'), (70, 77, 'LUGAR2')]}),
('Hola buenos días quisiera hacer una consulta la targeta de bonobus si la recargo para metrobus qué zona podría coger con dicha targeta\n', {'entities': [(36, 44, 'INFORMACION'), (59, 66, 'BONO'), (73, 80, 'VENTAONLINE'), (99, 103, 'QUE')]}),
('Buenos días qué línea puedo coger para ir de Zapadores esquina con Peris y Valero a lanzadera en el puerto de Valencia?\n', {'entities': [(34, 41, 'RUTAS'), (42, 44, 'ORIGEN'), (45, 81, 'LUGAR1'), (82, 83, 'DESTINO'), (84, 118, 'LUGAR2')]}),
('Cuanto le falta al 71 para llegar a la parada 534?\n', {'entities': [(10, 15, 'ESTIMACION'), (19, 21, 'LINEA'), (46, 49, 'PARADA')]}),
('Cuanto le falta al 93 para llegar a la parada 700\n', {'entities': [(10, 15, 'ESTIMACION'), (19, 21, 'LINEA'), (46, 49, 'PARADA')]}),
('Cuanto le falta al 71 para llegar a la parada 534?\n', {'entities': [(10, 15, 'ESTIMACION'), (19, 21, 'LINEA'), (46, 49, 'PARADA')]}),
('Buenas noche disculpe la pregunta estoy en alfafar quiero ir a la estación de autobús que línea debo tomar\n', {'entities': [(34, 42, 'ORIGEN'), (43, 50, 'LUGAR1'), (51, 60, 'RUTAS'), (61, 62, 'DESTINO'), (66, 85, 'LUGAR2')]}),
('Para ir a la estación de bus\n', {'entities': [(0, 7, 'RUTAS'), (8, 9, 'DESTINO'), (13, 28, 'LUGAR2')]}),
('Hola. Estoy en Manuel Candela con Port, parada 1110. El bus del 40 ha estado 7 minutos parado unos metros antes\n', {'entities': [(6, 11, 'ORIGEN'), (15, 39, 'LUGAR1'), (47, 51, 'PARADA'), (64, 66, 'LINEA'), (77, 86, 'TEMP'), (87, 93, 'QUEJA')]}),
('Que auto bus pasa por JJDomine a la altura del paseo maritimo\n', {'entities': [(13, 21, 'RUTAS'), (22, 30, 'LUGAR1')]}),
('Si gracias me lo acaban de decir Hola ! Que bus nocturno puedo coger por la zona de cortes valencianas paa ir al paseo de la pechina o alrededores ?\n', {'entities': [(84, 102, 'LUGAR1'), (107, 109, 'RUTAS'), (110, 112, 'DESTINO'), (113, 132, 'LUGAR2')]}),
('DONDE PUEDO PONER UNA RECLAMACIÓN??\n', {'entities': [(22, 33, 'RECLAMACION')]}),
('Que bus he de coger desde Tarongers para llegar a la Avenida Alfahuir?\n', {'entities': [(20, 25, 'ORIGEN'), (26, 35, 'LUGAR1'), (41, 47, 'RUTAS'), (48, 49, 'DESTINO'), (53, 69, 'LUGAR2')]}),
('Alfahuir 39\n', {'entities': [(0, 11, 'LUGAR2')]}),
('Me puedes decir el número del autobus y de la parada  para ir a zapadores a la comisaría\n', {'entities': [(19, 37, 'RUTAS'), (46, 52, 'INFORMACION'), (58, 60, 'RUTAS'), (61, 62, 'DESTINO'), (63, 72, 'LUGAR2')]}),
('Buenos días, tengo una consulta\n', {'entities': [(23, 31, 'INFORMACION')]}),
('Entiendo, gracias 👍\n', {'entities': [(18, 19, 'AFIRMACION')]}),
('A que hora cierra la oficina de la emt\n', {'entities': [(21, 28, 'CONTACTO'), (35, 38, 'EMT')]}),
('Buenas tardes. Tengo un hijo de 14 años con un grado de discapacidad del 37.Podria optar al bono oro? O a algún tipo de descuento. Gracias\n', {'entities': [(92, 101, 'BONO')]}),
('Necesito ir a la Calle del Impresor Monfort número 8 y estoy en la calle Luis Bolinches número 20\n', {'entities': [(0, 11, 'RUTAS'), (12, 13, 'DESTINO'), (27, 52, 'LUGAR2'), (55, 63, 'ORIGEN'), (73, 97, 'LUGAR1')]}),
('Como puedo ir desde pont de fusta a c democracia o cerca de ahí?\n', {'entities': [(11, 13, 'RUTAS'), (14, 19, 'ORIGEN'), (20, 33, 'LUGAR1'), (34, 35, 'DESTINO'), (36, 48, 'LUGAR2')]}),
('Buenas, me podrian decir cuanto le falta a la linea 19 en la parada 723? Gracias\n', {'entities': [(35, 40, 'ESTIMACION'), (52, 54, 'LINEA'), (68, 71, 'PARADA')]}),
('Buenas tardes más que una reclamación he hecho una felicitación por el formulario para el conductor que salvo del robo para que quede constancia un saludo\n', {'entities': [(51, 63, 'FELICITACION')]}),
('Pero que quede constancia por favor si tenemos que hacer reclamación tambien tenemos derecho a felicitar las buenas obras\n', {'entities': [(26, 35, 'EDUCADO'), (95, 104, 'FELICITACION')]}),
('A vosotros un saludo 😘\n', {'entities': [(21, 22, 'SMILEKISS')]}),
('Hay alguna que desde serrería o cabañal llegue cerca de aragon aunque sea la 92\n', {'entities': [(15, 20, 'ORIGEN'), (21, 29, 'LUGAR1'), (47, 52, 'DESTINO'), (56, 62, 'LUGAR2'), (77, 79, 'LINEA')]}),
('Buenas tardes mi Bono oro 3777 2496 8222,  q ayer 12 active formalizando todos los requisitos, he querido pagar hoy la cuota y el terminal indica q faltan requisitos.', {'entities': [(17, 25, 'BONO'), (26, 41, 'TARJETA')]}),
('Buenos días quiero saber cuando puedo pagar el bono oro? Desde el 12 y estamos a 18 y no puedo acceder al bono oro, me pueden decir algo?;\n', {'entities': [(12, 24, 'INFORMACION'), (47, 55, 'BONO')]}),
('Buenas! Como puedo llegar a mya ?\n', {'entities': [(19, 25, 'RUTAS'), (26, 27, 'DESTINO'), (28, 31, 'LUGAR2')]}),
('Hola, qué bus va desde Hipercor hasta la parada de metro 9 de octubre? Es para el sábado y horarios por favor\n', {'entities': [(14, 16, 'RUTAS'), (17, 22, 'ORIGEN'), (23, 31, 'LUGAR1'), (32, 37, 'DESTINO'), (51, 70, 'LUGAR2')]}),
('Hay algún problema con el 10 en Benimaclet\n', {'entities': [(10, 18, 'INCIDENCIAS'), (26, 28, 'LINEA'), (32, 42, 'LUGAR')]}),
('Tenía que haber pasado a las 7:12 por hermanos villalonga y no pasa\n', {'entities': [(10, 22, 'INCIDENCIAS')]}),
('Bona vesprada, per favor, a partir de les 12 de la nit, quin autobús puc agafar per anar de marítim serreria a carrer de la Reina? (Cabanyal). El nocturn pot passar per allí? Gràcies\n', {'entities': [(0, 14, 'SALUDOINIVAL'), (42, 55, 'TEMP'), (84, 88, 'RUTAS'), (89, 91, 'ORIGEN'), (92, 108, 'LUGAR1'), (109, 110, 'DESTINO'), (111, 141, 'LUGAR2')]}),
('Buenos días. Parada 496 no funciona panel\n', {'entities': [(20, 23, 'PARADA'), (24, 35, 'INCIDENCIAS')]}),
('Y la aplicación del móvil tampoco parece que funcione correctamente.\n', {'entities': [(26, 53, 'INCIDENCIAS')]}),
('En esta parada falla mucho la pantalla. casi nunca se ve la información\n', {'entities': [(15, 20, 'INCIDENCIAS')]}),
('Buenas tardes por favor que bus puedo coger desde el principio de Avda primado Reig para ir al auditori Palau de les arts reina Sofía,gracias un saludo\n', {'entities': [(44, 49, 'ORIGEN'), (66, 83, 'LUGAR1'), (84, 91, 'RUTAS'), (92, 94, 'DESTINO'), (95, 141, 'LUGAR2')]}),
('Buenas días.Ayer hize  una recarga de bonobús de 10 viajes y aún no me llegado la recarga.el número de tarjeta mobilis  es 271503002099. El número de pedido es 2080944.gracias\n', {'entities': [(26, 33, 'VENTAONLINE'), (37, 44, 'BONO'), (64, 77, 'INCIDENCIAS'), (121, 133, 'TARJETA')]}),
('Buenos días:Me llamo Boris González, DNI 44507598E.El pasado 24 de marzo hice una recarga online de mi tarjeta de transporte 514088273 de 30 viajes.Tras haberla utilizado una semana aparentemente 10 viajes no se llegaron a cargar. La semana pasada solo gasté 6 viajes.quería pedirles por favor revisen el saldo y si la recarga se hizo correctamente.Gracias\n', {'entities': [(82, 96, 'VENTAONLINE'), (125, 134, 'TARJETA')]}),
('Porfa vor para ir desde ayora hasta Túria como puedo hacer?\n', {'entities': [(0, 5, 'EDUCADO'), (10, 17, 'RUTAS'), (18, 23, 'ORIGEN'), (24, 29, 'LUGAR1'), (30, 35, 'DESTINO'), (36, 41, 'LUGAR2')]}),
('Y desde Blasco Ibáñez para ir a Túria?\n', {'entities': [(2, 7, 'ORIGEN'), (8, 21, 'LUGAR1'), (22, 29, 'RUTAS'), (30, 31, 'DESTINO'), (32, 37, 'LUGAR2')]}),
('Hola, hay un bus directo para ir desde el metro de  ángel guimera o xativa a la playa las arenas o malvarrosa\n', {'entities': [(25, 32, 'RUTAS'), (33, 38, 'ORIGEN'), (42, 64, 'LUGAR1'), (74, 75, 'DESTINO'), (79, 95, 'LUGAR2'), (96, 97, 'ALTERNATIVA')]}),
('Buenos dias para ir de patraix al palacio de justicia que buses tomar\n', {'entities': [(12, 19, 'RUTAS'), (20, 22, 'ORIGEN'), (23, 30, 'LUGAR1'), (31, 33, 'DESTINO'), (34, 53, 'LUGAR2')]}),
('Próximo 89 60 parada 1217\n', {'entities': [(0, 7, 'ESTIMACION'), (8, 10, 'LINEA'), (11, 13, 'LINEA'), (21, 25, 'PARADA')]}),
('Un bus de torres de serrano a Calipso tatto?\n', {'entities': [(7, 9, 'ORIGEN'), (10, 27, 'LUGAR1'), (28, 29, 'DESTINO'), (30, 43, 'LUGAR2')]}),
('Hola buenas! En la parada de bus 689 (ruzafa marques del turia) hace mucho que no funciona la pantalla de información\n', {'entities': [(33, 36, 'PARADA'), (37, 62, 'PARADA'), (79, 90, 'INCIDENCIAS')]}),
('parada 1332 linea 28\n', {'entities': [(7, 11, 'PARADA'), (18, 20, 'LINEA')]}),
('Hola para ir a la punta multiespacio desde la plaza Honduras que autobuses hay que coger?\n', {'entities': [(5, 12, 'RUTAS'), (13, 14, 'DESTINO'), (18, 36, 'LUGAR2'), (37, 42, 'ORIGEN'), (46, 60, 'LUGAR1')]}),
('Y para ir desde José soto mico hasta la plaza Honduras?\n', {'entities': [(2, 9, 'RUTAS'), (10, 15, 'ORIGEN'), (16, 30, 'LUGAR1'), (31, 36, 'DESTINO'), (40, 55, 'LUGAR2')]}),
('Hola para ir desde plaza España hasta la plaza América que autobuses me dejan cerca?\n', {'entities': [(5, 12, 'RUTAS'), (13, 18, 'ORIGEN'), (19, 31, 'LUGAR1'), (32, 37, 'DESTINO'), (41, 54, 'LUGAR2'), (78, 83, 'ATRIBUTO')]}),
('El 79 también serviría?\n', {'entities': []}),
('Vale, gracias! Y para volver desde ahí hasta José soto mico con las líneas nocturnas como sería?\n', {'entities': [(17, 28, 'RUTAS'), (29, 34, 'ORIGEN'), (39, 44, 'DESTINO'), (45, 59, 'LUGAR2')]}),
('Y desde la plaza América hasta el ayuntamiento no hay combinación?\n', {'entities': [(0, 1, 'OTRAC'), (2, 7, 'ORIGEN'), (11, 24, 'LUGAR1'), (25, 30, 'DESTINO'), (34, 46, 'LUGAR2')]}),
('Me he dejado olvidado un paraguas en el asiento se detrás del conductor en la linea 94. Ha sido hace diez minutos cuando pasó por navarro Reverter\n', 
{'entities': [(6, 12, 'OBJETOS'), (13, 21, 'OBJETOS'), (25, 33, 'QUE'), (84, 86, 'LINEA')]}),
('Buenos dias señores, x favor mi hija tiene el carnet de bonobús mensual pero ahora en adelante cogerá más el tranvía q tengo q hacer y donde\n', {'entities': [(56, 71, 'BONO'), (117, 132, 'INFORMACION')]}), 
('Buenos días, Me podrían informar  a que hora pasa en la mañana el primer bus de la ruta 92 los sábados en la parada de joan verdeguer pare porta\n', {'entities': [(33, 48, 'ESTIMACION'), (87, 89, 'LINEA'), (118, 132, 'PARADA')]}),
('Próximo 9 o 10 parada 742\n', {'entities': [(0, 7, 'ESTIMACION'), (8, 9, 'LINEA'), (10, 11, 'ALTERNATIVA'), (12, 14, 'LINEA'), (22, 25, 'PARADA')]}),
("Bona vesprada. Este mati he carregat la targeta i no s'ha fet efectiva. He tingut que pagar un bitllet\n", {'entities': [(28, 36, 'VENTAONLINE'), (50, 71, 'INCIDENCIAS')]}),
('Hola buenos días, hay alteración del servicio de la linea 15 de Pinedo a la pl. Del ajuntament ??\n', {'entities': [(22, 32, 'INCIDENCIAS'), (58, 60, 'LINEA'), (61, 63, 'ORIGEN'), (64, 70, 'LUGAR1'), (71, 72, 'DESTINO'), (76, 94, 'LUGAR2')]}),
('Línea 71 parada 511\n', {'entities': [(6, 8, 'LINEA'), (16, 19, 'PARADA')]}),
('Línea 71 parada 2007\n', {'entities': [(6, 8, 'LINEA'), (16, 20, 'PARADA')]}),
('Buenos días quisiera saber si ese viaje , digamos de cortesía, que te da la tarjeta cuando tienen saldo cero, se cobra cuando se recarga la tarjeta y se sube de nuevo al bus. Espero haberme explicado. Gracias\n', {'entities': [(12, 26, 'INFORMACION'), (129, 136, 'VENTAONLINE')]}),
('Gracias a ti, feliz lunes!De plaza España o pintor segrelles a maestro Rodrigo 4. Gracias, Para mañana por la mañana\n', {'entities': [(26, 28, 'ORIGEN'), (29, 41, 'LUGAR1'), (61, 62, 'DESTINO'), (63, 80, 'LUGAR2')]}),
('Hola, tras recargar mi tarjeta EMT ayer a las 18:00 via Online\n', {'entities': [(11, 19, 'VENTAONLINE'), (56, 62, 'VENTAONLINE')]}),
('Cuando tendre disponibles mis viajes? Al realizar la recarga se hablaba de 30 minutos\n', {'entities': [(53, 60, 'VENTAONLINE')]}),
('Buenas tardes. Me llamo Agata Janiak, tengo una pregunta, porque hoy cerca de las 20 dejé mi cartera en el bus número 19. Podría fijarlo alguien? (Si está aquí todo el tiempo) En la cartera tenía tarjetas i mi carne de identidad.\n', {'entities': [(65, 68, 'TEMP'), (85, 89, 'OBJETOS'), (93, 100, 'QUE'), (118, 120, 'LINEA')]}),
('Buenos días. Sí, sigo teniendo dudas. Se trata de mi cartera, que dejé al viernes cerca de las 20 en el bus número 19. A mi me gustaría saber, que alguien la he traido a su oficina o tal vez el conductor la he notado. Para mi es muy importante porque allí tenía mi carne de identidad y sin esto no pueda volver a mi país (no soy española)\n', {'entities': [(53, 60, 'QUE'), (66, 70, 'OBJETOS')]}),
('Hola! Me gustaría preguntar si alguien sabe algo sobre mi caso. Hay mi cartera en su oficina?\n', {'entities': [(71, 78, 'QUE'), (85, 92, 'CONTACTO')]}),
('Muchísimas gracias! Hoy voy a pasar. 😊\n', {'entities': [(0, 19, 'AGRADECIMIENTO'), (37, 38, 'SMILE')]}),
('Hola estoy en la plaza ayuntamiento, qué bus tengo que coger para ir a avda primado reig 39?', {'entities': [(5, 10, 'ORIGEN'), (17, 36, 'LUGAR1'), (61, 68, 'RUTAS'), (69, 70, 'DESTINO'), (71, 100, 'LUGAR2')]}),
('Porque desde las 7,20 en la pantalla ponen que faltan 5 minutos y llevamos 20 esperando\n', {'entities': [(28, 36, 'QUE'), (78, 87, 'QUEJA')]}),
('Poca vergüenza\n', {'entities': [(5, 14, 'QUEJA')]}),
('El n99 tiene 2 paradas en pio xIl ?? Verdad\n', {'entities': [(4, 6, 'LINEA'), (15, 22, 'ITINERARIO'), (26, 33, 'LUGAR')]}),
('Hola, como puedo llegar desde la av del puerto hasta el multiespai de la punta?\n', {'entities': [(17, 23, 'RUTAS'), (24, 29, 'ORIGEN'), (33, 46, 'LUGAR1'), (47, 52, 'DESTINO'), (56, 79, 'LUGAR2')]}),
('Buenos días. Ayer fui a atención al cliente de la calle colon porque no se había hecho efectiva la recarga del bonobus y me la hicieron manualmente, pero hoy miro en la aplicación y siguen sin aparecer los viajes. Tengo que ir de nuevo a atención al cliente? Espero vuestra respuesta. Gracias.\n', {'entities': [(81, 95, 'INCIDENCIAS'), (99, 106, 'VENTAONLINE'), (111, 118, 'BONO'), (238, 257, 'CONTACTO')]}),
('Es el 1203 1418 3471\n', {'entities': [(6, 20, 'TARJETA')]}),
('Sí, ayer por la tarde. Entonces cuando suba al bus la valido donde el conductor? Y cuanto tiempo tengo? Porque hasta el sábado próximo no lo voy a usar.\n', {'entities': [(54, 60, 'VENTAONLINE')]}),
('Ok muchas gracias¿Que es lo que pasa con el 13 que lleva 2 mañanas no pasando por la 637 a las 7:16 cuando siempre pasa sobre esa hora?\n', {'entities': [(28, 36, 'QUEJA'), (44, 46, 'LINEA'), (85, 88, 'PARADA')]}),
('Ok gracias, es todo cuestion de dinero, compro el bonobus mensual y ya he pagado 2 taxis en 2 dias seguidos🤷🏻\u200d♀🤦🏻\u200d♀\n', {'entities': [(50, 65, 'BONO'), (83, 88, 'QUEJA')]}),
('Hola quiero saber si el carnet Jove si llevas la tarjeta de familia númerosa me descuentan?\n', {'entities': [(5, 17, 'INFORMACION'), (24, 35, 'BONO')]}),
('Bns dias. Perdi mi amb tu y me hice otra pero resulta que me la han quitado y la que perdi la he encontrado puedo activar la primera o me tengo que hacer otra?\n', {'entities': [(10, 15, 'OBJETOS'), (19, 25, 'BONO'), (85, 90, 'OBJETOS')]}),
('Hola para ir de la fuente San Luis a Pérez Galdós, 38?\n', {'entities': [(5, 12, 'RUTAS'), (13, 15, 'ORIGEN'), (19, 34, 'LUGAR1'), (35, 36, 'DESTINO'), (37, 54, 'LUGAR2')]}),
('Hola para ir a la calle Torrente de Tres Forques puedo coger la línea 99 no?\n', {'entities': [(5, 12, 'RUTAS'), (13, 14, 'DESTINO'), (18, 32, 'LUGAR2'), (33, 35, 'ORIGEN'), (36, 48, 'LUGAR1'), (70, 72, 'LINEA')]}),
('Desde Antonio Ferrandis...\n', {'entities': [(0, 5, 'ORIGEN'), (6, 23, 'LUGAR1')]}),
('En cuanto pasa el bus 16 Número de parada 1151', {'entities': [(10, 14, 'ESTIMACION'), (22, 24, 'LINEA'), (42, 46, 'PARADA')]}),
('Hola en cuanto pasa el 6 en la parada barques Número 837\n', {'entities': [(15, 19, 'ESTIMACION'), (23, 24, 'LINEA'), (53, 56, 'PARADA')]}),
('Que autobus hay que cojer para ir desde yelmo cines hasta el colegio guadalaviar?\n', {'entities': [(26, 33, 'RUTAS'), (34, 39, 'ORIGEN'), (40, 51, 'LUGAR1'), (52, 57, 'DESTINO'), (61, 80, 'LUGAR2')]}),
('Hola, buenos días. He perdido mi tarjeta bono oro y quería saber que tengo que hacer para que anulen esa y pedir un duplicado. Gracias\n', {'entities': [(19, 29, 'OBJETOS'), (41, 49, 'BONO'), (94, 100, 'QUE'), (116, 125, 'CONTACTO')]}),
('Creo que fue en la fe, ayer llamé pero no la había entregado nadie\n', {'entities': []}),
('Buenos días autobús para ir a la calle Maximiliano Thous 23 desde Cirilo Amoros 65 ida y vuelta, gracias\n', {'entities': [(20, 27, 'RUTAS'), (28, 29, 'DESTINO'), (33, 59, 'LUGAR2'), (60, 65, 'ORIGEN'), (66, 82, 'LUGAR1')]}),
('Hola cuantos minutos pasa el 99 Palau de congressos Número de parada 2243\n', {'entities': [(5, 20, 'ESTIMACION'), (21, 25, 'ESTIMACION'), (29, 31, 'LINEA'), (69, 73, 'PARADA')]}),
('Hola a que horas pasa el 99 parada 1752 Fausto Elio (impar) Gracias\n', {'entities': [(5, 21, 'ESTIMACION'), (25, 27, 'LINEA'), (35, 39, 'PARADA')]}),
('Me hice una recarga online a las 6:30 de la mañana...y la operacion me dio aceptada....y son las 12 y no me ha llegado aun\n', {'entities': [(12, 26, 'VENTAONLINE'), (111, 122, 'INCIDENCIAS')]}),
('058454523484\n', {'entities': [(0, 12, 'TARJETA')]}),
('Autobús desde la plaza de San Agustín calle San Vicente para ir a Blasco Ibáñez\n', {'entities': [(8, 13, 'ORIGEN'), (17, 37, 'LUGAR1'), (56, 65, 'RUTAS'), (64, 65, 'DESTINO'), (66, 79, 'LUGAR2')]}),
('Hola desde la calle San Pancracio 29 a la calle Maestro gozalbo N 23 q autobús tengo q coger? Tengo q estar a las 14 h\n', {'entities': [(5, 10, 'ORIGEN'), (20, 36, 'LUGAR1'), (37, 38, 'DESTINO'), (48, 68, 'LUGAR2')]}),
('Buenos días,  como podría ir de Blasco Ibañez,  50 a calle Brasil\n', {'entities': [(18, 27, 'RUTAS'), (28, 30, 'ORIGEN'), (31, 48, 'LUGAR1'), (49, 50, 'DESTINO'), (57, 63, 'LUGAR2')]}),
('Hola, quería deciros que llevamos 2 semanas con retrasos y autobuses con fuera de servicio en la línea 62. Cada día se retrasa más y se tira 10 minutos poniendo que llega: prox.\n', {'entities': [(6, 20, 'QUEJA'), (48, 56, 'QUEJA'), (103, 105, 'LINEA'), (119, 126, 'QUEJA')]}),
('Que autobus tengo que coger para ir a colón y estoy en blasco ibañezBuenas tardes. Queria saber si me podeis comprobar que una recarga de bonobus está correcta. Acabo de subir al bus y no sale todavia\n', {'entities': [(28, 35, 'RUTAS'), (36, 37, 'DESTINO'), (38, 43, 'LUGAR2'), (46, 51, 'ORIGEN'), (55, 68, 'LUGAR1'), (109, 118, 'INFORMACION'), (127, 134, 'VENTAONLINE'), (138, 145, 'BONO')]}),
('Gracias 😊\n', {'entities': [(0, 7, 'AGRADECIMIENTO'), (8, 9, 'SMILE')]}),
('Un saludo. GraciasBuenas es que no pude renovar el bono oro en enero lo puedo renovar en los estancos aun?\\nY cuanto me costaría?\\n\\nGracias\n', {'entities': [(40, 47, 'INFORMACION'), (51, 59, 'BONO')]}),
('Bon dia, como podria hacer para ir a la Quiron en blasco ibañez, desde calle olta?\n', {'entities': [(0, 7, 'SALUDOINIVAL'), (27, 34, 'RUTAS'), (35, 36, 'DESTINO'), (40, 64, 'LUGAR2'), (65, 70, 'ORIGEN'), (71, 81, 'LUGAR1')]}),
('Buenos días !!A día de hoy, hace ya año y medio que, la parada de autobús número 126 sigue sin funcionar el panel informativo\n', {'entities': [(36, 47, 'TEMP'), (81, 84, 'PARADA'), (91, 104, 'QUEJA'), (108, 125, 'QUE')]}),
('El domingo hola oye llámame que ahora estoy tranquila aquí en casa viendo la televisión y me voy a dormir que me llame que quiero hacerte una videollamada a ti y no me va no sé porqué volver e ir otra vez a ver a dónde me lo arreglaron ayer a ver porque me he arreglado una cosa y les arregla la otra llámame tú a ver si vas por favor que ahora ya puedo hablar contigo ya te contaré porque no he podido hoy ya te lo contaré cariño y si podemos hablar por videoconferencia te lo te lo diré y si no pues cuando vengas a verme te lo contaré vale cariño vale besos llámame por favor\n', {'entities': []}),
('Que disculpar disculpad que me he equivocado pensaba hablar a una persona y no me he dado cuenta cuando ya se lo había enseñado yo sé que este canal no es para nada de eso es para una cosa seria y cómo lo hacemos pero me he equivocado disculparme y borrarlo por favor se lo pido por favor soy Carmen sierra disculparme me he equivocado\n', {'entities': []}),
('Muchísimas Gracias 🤗🤗🤗🤗🤗\n', {'entities': [(0, 18, 'AGRADECIMIENTOS'), (19, 24, 'SMILE')]}),
('😜gracias\n', {'entities': [(0, 8, 'AGRADECIMIENTO')]}),
('Hola vivo en río Nervión qué autobús puedo coger para ir a la estación del Norte gracias\n', {'entities': [(5, 12, 'ORIGEN'), (13, 24, 'LUGAR1'), (49, 56, 'RUTAS'), (57, 58, 'DESTINO'), (62, 80, 'LUGAR2')]}),
('Hola desde la calle traginers para ir a la estación del Norte qué autobús puedo coger gracias\n', {'entities': [(5, 10, 'ORIGEN'), (20, 29, 'LUGAR1'), (30, 37, 'RUTAS'), (38, 39, 'DESTINO'), (43, 61, 'LUGAR2')]}),
('Buenas tardes, podrían decirme cuando pasa el 19 en la 1462?\n', {'entities': [(31, 42, 'ESTIMACION'), (46, 48, 'LINEA'), (55, 59, 'PARADA')]}),
('Carrer de Sant Josep de Calassanç, 18, 46008 València\n', {'entities': [(10, 37, 'LUGAR2')]}),
('Desde Jaime Roig, 8\n', {'entities': [(0, 5, 'ORIGEN'), (6, 19, 'LUGAR1')]}),
('👍🏽👍🏽👍🏽\n', {'entities': [(0, 6, 'AFIRMACION')]}),
('Buses al Saler desde gran vía?\n', {'entities': [(6, 8, 'DESTINO'), (9, 14, 'LUGAR2'), (15, 20, 'ORIGEN'), (21, 29, 'LUGAR1')]}),
('Me gustaría información sobre la línea n 89 parto desde el paseo alameda 49\n', {'entities': [(12, 23, 'INFORMACION'), (39, 43, 'LINEA'), (50, 55, 'ORIGEN'), (59, 75, 'LUGAR1')]}),
('La línea nocturna n 89 quería saber que parada tengo cercana al paseo alameda 49\n', {'entities': [(9, 17, 'ATRIBUTO'), (18, 22, 'LINEA'), (40, 46, 'ITINERARIO'), (64, 80, 'LUGAR')]}),
('Buenas tardes, q autobús debo coger para ir desde avda Pérez Galdos, 115. Al palacio de Congresos. Gracias.\n', {'entities': [(36, 43, 'RUTAS'), (44, 49, 'ORIGEN'), (50, 72, 'LUGAR1'), (74, 76, 'DESTINO'), (77, 98, 'LUGAR2')]}),
('Buenos días, me pueden indicar que bus debo coger para ir d Avenida Pérez Galdos, 115.  Al Tanatorio municipal de Valencia. Gracias. Saludos\n', {'entities': [(50, 57, 'RUTAS'), (58, 59, 'ORIGEN'), (60, 85, 'LUGAR1'), (87, 89, 'DESTINO'), (90, 122, 'LUGAR2')]}),
('Q bus puedo coger de la ciudad de la justicia a calle colon\n', {'entities': [(18, 20, 'ORIGEN'), (24, 45, 'LUGAR1'), (46, 47, 'DESTINO'), (48, 59, 'LUGAR2')]}),
('Por favor quisiera preguntar el horario que sale del horno Alcedo hacia Valencia muchas gracias\n', {'entities': [(19, 28, 'INFORMACION'), (32, 39, 'ESTIMACION'), (53, 65, 'LUGAR1'), (72, 80, 'LUGAR2')]}),
('Bon día. Que autobus va desde la estación de autobuses a la del norte??.\n', {'entities': [(24, 29, 'ORIGEN'), (33, 54, 'LUGAR1'), (55, 56, 'DESTINO'), (60, 69, 'LUGAR2')]}),
('👍🏼\n', {'entities': [(0, 2, 'AFIRMACION')]}),
('porque el 60 no pasa por la calle editor manuel aguilar?\n', {'entities': [(10, 12, 'LINEA'), (13, 15, 'INCIDENCIAS'), (16, 24, 'ESTIMACION'), (34, 55, 'LUGAR')]}),
('llevamos 40 minutos esperando\n', {'entities': [ (20, 29, 'QUEJA')]}),
('Desde la calle Comte de Altea 44\n', {'entities': [(0, 5, 'ORIGEN'), (15, 32, 'LUGAR')]}),
('Sabe cómo puedo llegar a la Calle Roteros 16\n', {'entities': [(16, 22, 'RUTAS'), (23, 24, 'DESTINO'), (28, 44, 'LUGAR2')]}),
('Y desde la calle Comte de Altea 44\n', {'entities': [(2, 7, 'ORIGEN'), (17, 34, 'LUGAR1')]}),
('Cómo puedo llegar a Ruzafa en bus? Por la calle Cádiz, Cuba, Literato Azorín..\n', {'entities': [(11, 17, 'RUTAS'), (18, 19, 'DESTINO'), (20, 26, 'LUGAR2')]}),
('Y desde Comte de Altea 44\n', {'entities': [(0, 1, 'OTRAC'), (2, 7, 'ORIGEN'), (8, 25, 'LUGAR1')]}),
('Hasta el Carrer de Conca, 31?\n', {'entities': [(0, 5, 'DESTINO'), (9, 28, 'LUGAR2')]}),
('Soy Milagros Llorca se me ha perdido una carterita marrón con el bono amb tu y otras cosas\n', {'entities': [(26, 36, 'OBJETOS'), (41, 50, 'QUE'), (70, 76, 'BONO'), ]}),
('El bus 90\n', {'entities': [(7, 9, 'LINEA')]}),
('Hola Tengo que ir a la dirección de pasaje Ventura Feliu desde la avenida del puerto, me podéis indicar qué autobús tengo que coger\n', {'entities': [(5, 17, 'RUTAS'), (18, 19, 'DESTINO'), (36, 56, 'LUGAR2'), (58, 63, 'ORIGEN'), (66, 85, 'LUGAR1')]}),
('Hola para ir a la avenida de Suecia desde Islas Canarias, qué autobús me viene bien\n', {'entities': [(5, 12, 'RUTAS'), (13, 14, 'DESTINO'), (18, 25, 'LUGAR2'), (36, 41, 'ORIGEN'), (42, 57, 'LUGAR1')]}),
('De av Arango a fe hospital\n', {'entities': [(0, 2, 'ORIGEN'), (3, 12, 'LUGAR1'), (13, 14, 'DESTINO'), (15, 26, 'LUGAR2')]}),
('Hola, buenos días. A Ver qué pasa con el 62. El que tenía que pasar por la parada 2161, ha pasado con un poco más de 10 minutos de retraso. Iban tan seguidos que estaba detrás en la parada del mercado central.  Esto lleva así desde Navidades. Y fastidia mucho por qué nos hace perder combinaciones de otras líneas, no solo a mi, muchas de las personas que lo cogemos a esa hora tenemos el problema. La semana pasada venía en hora, pero hoy ha vuelto a las andadas.  A ver si solucionais esto, que esto no sucede por culpa de semáforos....\n', {'entities': [(25, 33, 'QUEJA'), (41, 43, 'LINEA'), (85, 89, 'PARADA'), (121, 131, 'TEMP'), (133, 140, 'QUEJA'), (475, 486, 'RECLAMACION')]}),
('Ok. GraciasMe gustaría información sobre la línea n 89 parto desde el paseo alameda 49\n', {'entities': [(23, 34, 'INFORMACION'), (50, 54, 'LINEA'), (55, 66, 'ORIGEN'), (70, 86, 'LUGAR')]}),
('La línea nocturna n 89 quería saber que parada tengo cercana al paseo alameda 49\n', {'entities': [(9, 17, 'ATRIBUTO'), (18, 22, 'LINEA'), (23, 35, 'INFORMACION'), (40, 46, 'QUE'), (64, 80, 'LUGAR')]}),
('Recargue la tarjeta ayer con veinte viajes y todavía no puedo utilizarlos, podéis ayudarme?\n', {'entities': [(0, 8, 'VENTAONLINE'), (20, 24, 'TEMP'), (45, 55, 'QUEJA')]}),
('Ayer sobre las once no funcionó y tuve que pagar\n', {'entities': [(20, 31, 'INCIDENCIAS')]}),
('1208 6538 6741\n', {'entities': [(0, 14, 'TARJETA')]}),
('Hola! Por favor me pueden decir para ir de Malvarrosa a ver Falla ANTIGA de Campanar,la del primer premio?\n', {'entities': [(32, 39, 'RUTAS'), (40, 42, 'ORIGEN'), (43, 53, 'LUGAR1'), (54, 55, 'DESTINO'), (60, 84, 'LUGAR2')]}),
('😊\n', {'entities': [(0, 1, 'SMILE')]}),
('Por favor me pueden decir para ir desde la plaza de la virgen hasta decatlón de Campanar como puede ir en  bus?\n', {'entities': [(26, 33, 'RUTAS'), (34, 39, 'ORIGEN'), (43, 61, 'LUGAR1'), (62, 67, 'DESTINO'), (68, 88, 'LUGAR2')]}),
('Si realmente tuvieran vocación de servicio , en lugar de contestarme con el horario , me hubieran recomendado hacer la reclamación por escrito .\n', {'entities': [(119, 130, 'RECLAMACION')]}),
('Y que hacemos con las incidencias que pasan fuera del horario de atención al cliente ?\n', {'entities': [(22, 33, 'INCIDENCIAS'), (65, 73, 'CONTACTO')]}),
('A qué hora pasa el 99 en tres creus/Bolivia?\n', {'entities': [(2, 10, 'ESTIMACION'), (19, 21, 'LINEA'), (25, 43, 'PARADA')]}),
('Hola buenas tardes tengo una consulta lo que pasa es que tengo q salir y en mi tarjeta me sale q no es válida en esa tarjeta tengo 4 viajes y pues como hago para q me pasen esos viajes a otra q tengo por que según el chico me dijo que tengo q comprar otro billete y la verdad no tengo\n', {'entities': [(29, 37, 'INFORMACION'), (79, 86, 'QUE')]}),
('1834\n', {'entities': [(0, 4, 'PARADA')]}),
('Cuando pasa\n', {'entities': [(0, 11, 'ESTIMACION')]}),
('1834\n', {'entities': [(0, 4, 'PARADA')]}),
('Cuando pasa?\n', {'entities': [(0, 11, 'ESTIMACION')]}),
('Hola buenas tardes para ir a la calle sagunto número 164\n', {'entities': [(19, 26, 'RUTAS'), (27, 28, 'DESTINO'), (32, 56, 'LUGAR2')]}),
('Desde av tres cruces con hospital general\n', {'entities': [(0, 5, 'ORIGEN'), (6, 41, 'LUGAR1')]}),
('1570 5783 8712;\n', {'entities': [(0, 15, 'TARJETA')]}),
('Buenos días, y muchas gracias por contestar,  nesecito saber qué bus puedo coger de la calle  la paz a la plaza España, disculpe las molestias\n', {'entities': [(45, 59, 'INFORMACION'), (80, 82, 'ORIGEN'), (83, 98, 'LUGAR1'), (99, 100, 'DESTINO'), (104, 117, 'LUGAR2')]}),
('Hola para ir al museo Fallero de Oriols, desde Blasco Ibáñez, que línea puedo coger.\n', {'entities': [(5, 12, 'RUTAS'), (13, 15, 'DESTINO'), (16, 40, 'LUGAR2'), (41, 46, 'ORIGEN'), (47, 61, 'LUGAR1')]}),
('Buenos días para ir a c/ Periodista Lorente 3, desde plaza Ayuntamiento, que línea puedo subir.\n', {'entities': [(12, 19, 'RUTAS'), (20, 21, 'DESTINO'), (22, 45, 'LUGAR2'), (47, 52, 'ORIGEN'), (53, 72, 'LUGAR1')]}),
('Hola el bus 92, me enlaza con el 8, para llegar a La Fe nueva.\n', {'entities': [(12, 14, 'LINEA'), (36, 47, 'RUTAS'), (48, 49, 'DESTINO'), (50, 61, 'LUGAR2')]}),
('Hola para llegar a la c/ Llano de la Zaida 1, desde la c/ Lenin, que línea puedo coger.\n', {'entities': [(5, 16, 'RUTAS'), (17, 18, 'DESTINO'), (19, 44, 'LUGAR1'), (46, 51, 'ORIGEN'), (52, 63, 'LUGAR1')]}),
('Hola, que bus puedo coger desde Calle Portal de Valldigna para ir al.parque de Cabecera? Gracias\n', {'entities': [(26, 31, 'ORIGEN'), (32, 57, 'LUGAR1'), (58, 65, 'RUTAS'), (66, 87, 'LUGAR2')]}),
('Por favor a qué hora empieza a pasar el 60 por santo domingo el sabio torrefiel gracias\n', {'entities': [(16, 20, 'ESTIMACION'), (31, 36, 'ESTIMACION'), (40, 42, 'LINEA'), (47, 79, 'LUGAR')]}),
('Gracias por su respuesta. Pero se va a hacer algo para solucionarlo? Llevamos así ya tres días y la verdad es que si el trayecto fuese corto, daría igual ir andando que aquí dentro.\n', {'entities': [(55, 67, 'RECLAMACION')]}),
('Siendo todas las paradas del 62 en terramelar desviadas\n', {'entities': [(29, 31, 'LINEA'), (46, 55, 'INCIDENCIAS')]}),
('La q esta suprimida es la 390\n', {'entities': [(10, 19, 'INCIDENCIAS'), (26, 29, 'PARADA')]}),
('Hola perdona estoy en la parada piu 12-campanar y no pasa ningún autobús\n', {'entities': [(32, 47, 'PARADA'), (53, 57, 'ESTIMACION')]}),
('hola! un bus desde Tarongers hasta el hospital 9 de octubre?\n', {'entities': [(13, 18, 'ORIGEN'), (19, 28, 'LUGAR1'), (29, 34, 'DESTINO'), (35, 59, 'LUGAR2')]}),
('Que bus puedo coger desde gran vía Fernando el católico hacia micer masco o cerca?\n', {'entities': [(20, 25, 'ORIGEN'), (26, 55, 'LUGAR1'), (56, 61, 'DESTINO'), (62, 73, 'LUGAR2')]}),
('buenos días, creo que he perdido mi tarjeta de emt amb tú.  Es posible que alguien os haya avisado?\n', {'entities': [(22, 32, 'OBJETOS'), (36, 43, 'QUE')]}),
('Cuánto falta para q pase el bus por la parada 509?\n', {'entities': [(20, 24, 'ESTIMACION'), (46, 49, 'PARADA')]}),
('En cuanto tiempo pasa el 72 por la parada 509\n', {'entities': [(17, 21, 'ESTIMACION'), (25, 27, 'LINEA'), (42, 45, 'PARADA')]}),
('En cuanto tiempo pasa el 72 por la parada 509?\n', {'entities': [(17, 21, 'ESTIMACION'), (25, 27, 'LINEA'), (42, 45, 'PARADA')]}),
('En cuanto tiempo pasa el 99 por la parada 2100???\n', {'entities': [(17, 21, 'ESTIMACION'), (25, 27, 'LINEA'), (42, 46, 'PARADA')]}),
('Para ir al hospital la fé desde el hospital arnau qué autobus puedo coger?\n', {'entities': [(0, 7, 'RUTAS'), (8, 10, 'DESTINO'), (11, 25, 'LUGAR2'), (26, 31, 'ORIGEN'), (35, 49, 'LUGAR1')]}),
('Y luego para volver desde la falla na jordana hasta la calle ayora?\n', {'entities': [(13, 19, 'RUTAS'), (20, 25, 'ORIGEN'), (26, 45, 'LUGAR1'), (46, 51, 'DESTINO'), (55, 66, 'LUGAR2')]}),
('Buenas tardes! Podría decirme la mejor opción para llegar en bus desde la calle ayora  al centro Arrupe (av. Gran via Fernando el católico 78)? Gracias\n', {'entities': [(46, 57, 'RUTAS'), (65, 70, 'ORIGEN'), (71, 85, 'LUGAR1'), (86, 88, 'DESTINO'), (89, 102, 'LUGAR2')]}),
('Quiero ir a peset aleixandre, 110 Estoy en la calle jativa', {'entities': [(0, 9, 'RUTAS'), (10, 11, 'ORIGEN'), (12, 33, 'LUGAR2'), (34, 42, 'ORIGEN'), (43, 58, 'LUGAR1') ]}),
('Buenas noches  no se si se me ha caído el bono amb tu en el autobús de la unidad 27 sobre las 17h30Desde campillo altobuey hasta calle sangre que bus puedo coger o buses\n', {'entities': [(23, 37, 'OBJETOS'), (41, 52, 'QUE')]}),
('120785005966\n', {'entities': [(0, 12, 'TARJETA')]}),
('que autobus podia cojer desde dl hospital.el clinico que me dejara cerca de renfe\n', {'entities': [(24, 29, 'ORIGEN'), (30, 52, 'LUGAR1'), (60, 66, 'DESTINO'), (76, 81, 'LUGAR2')]}),
('Para venir calle tomas sanz en mislata a escultor José capuz que bus\n', {'entities': [(0, 10, 'RUTAS'), (11, 38, 'LUGAR2'), (39, 40, 'DESTINO'), (41, 60, 'LUGAR1')]}),
('Llevo 20 minutos esperando al 99 y ponía que eram 12\n', {'entities': [(17, 26, 'INCIDENCIAS'), (30, 32, 'LINEA')]}),
('Para ir a Músico Gines desde la Cuidad Fallera q buses he d coger\n', {'entities': [(0, 7, 'RUTAS'), (8, 9, 'DESTINO'), (10, 22, 'LUGAR2'), (23, 28, 'ORIGEN'), (29, 46, 'LUGAR1')]}),
('Para ir a Isabel la católica desde la cuidad fallera q buses he d coger\n', {'entities': [(0, 7, 'RUTAS'), (8, 9, 'DESTINO'), (10, 28, 'LUGAR2'), (29, 34, 'ORIGEN'), (35, 52, 'LUGAR1')]}),
('GraciasBuenas tardes. El otro dis recargué via internet un bonobús. Pero no me salen los viajes cargados. Me lo podrian mirar ?\n', {'entities': [(34, 42, 'VENTAONLINE'), (59, 66, 'BONO'), (68, 72, 'INCIDENCIAS')]}),
('¿Que es lo que pasa con el 13 que lleva 2 mañanas no pasando por la 637 a las 7:16 cuando siempre pasa sobre esa hora?\n', {'entities': [(1, 19, 'INCIDENCIAS'), (27, 29, 'LINEA')]}),
('¿Que es lo que pasa con el 13 que lleva 2 mañanas no pasando por la 637 a las 7:16 cuando siempre pasa sobre esa hora?\n', {'entities': [(1, 19, 'INCIDENCIAS'), (27, 29, 'LINEA')]}),
('Estoy en la Calle Guillem de Castro numero 100 y quiero ir a la calle Guardia Civil 21. Me gustaría saber como puedo ir hasta allí. Gracias.\n', {'entities': [(0, 8, 'ORIGEN'), (10, 47, 'LUGAR1'), (50, 59, 'RUTAS'), (60, 61, 'DESTINO'), (62, 88, 'LUGAR2')]}),
('1268 73\n', {'entities': [(0, 4, 'PARADA'), (5, 7, 'LINEA')]}),
('Hola cuanto tarda el bus d la línea 14 número de parada 1290?\n', {'entities': [(5, 17, 'ESTIMACION'), (36, 38, 'LINEA'), (56, 60, 'PARADA')]}),
('Hola cuanto tarda el bus d la línea 14 número de parada 664?\n', {'entities': [(5, 17, 'ESTIMACION'), (36, 38, 'LINEA'), (56, 59, 'PARADA')]}),
('Hola cuanto tarda el bus d la línea 14 número de parada 664?\n', {'entities': [(5, 17, 'ESTIMACION'), (36, 38, 'LINEA'), (56, 59, 'PARADA')]}),
('Cuanto tarda el bus d la 70 número de parada 506?\n', {'entities': [(0, 12, 'ESTIMACION'), (25, 27, 'LINEA'), (45, 48, 'PARADA')]}),
('Hola cuanto tarda el bus d la línea 14 número de parada 685?\n', {'entities': [(5, 17, 'ESTIMACION'), (36, 38, 'LINEA'), (56, 59, 'PARADA')]}),
('Cuanto tarda el bus d la línea 14 número de parada 910?\n', {'entities': [(0, 12, 'ESTIMACION'), (31, 33, 'LINEA'), (51, 54, 'PARADA')]}),
('Hola cuanto tarda el bus d la línea 14 número de parada 664?\n', {'entities': [(5, 17, 'ESTIMACION'), (36, 38, 'LINEA'), (56, 59, 'PARADA')]}),
('Se me ha caido el monedero en uno de vuestros autobuses\n', {'entities': [(0, 14, 'OBJETOS'), (18, 26, 'QUE')]}),
('Buenos días el miércoles pasado recargué mi tarjeta 206725790765 online. Hoy me he subido por primera vez desde la recarga al bus y la tarjeta no se había recargado ( la he validado en lo del conductor ) y le salía al conductor error\n', {'entities': [ (32, 40, 'VENTAONLINE'), (52, 64, 'TARJETA'), (228, 233, 'INCIDENCIAS')]}),
('Parada 64 pasará el 98????\n', {'entities': [(7, 9, 'PARADA'), (10, 16, 'ESTIMACION'), (20, 22, 'LINEA')]}),
('El sabado recargué el bonobus y ahora al subir no tenia saldo\n', {'entities': [(10, 18, 'VENTAONLINE'), (22, 29, 'BONO')]}),
('4025 7699 7478\n', {'entities': [(0, 14, 'TARJETA')]}),
('Estoy en la estación del cabañal,,, que bus puedo coger para ir hacia la calle clariano\n', {'entities': [(0, 8, 'ORIGEN'), (9, 32, 'LUGAR1'), (56, 63, 'RUTAS'), (64, 69, 'DESTINO'), (70, 87, 'LUGAR2')]}),
('Buenas tardes!  Necesito me informes por favor, cuando te sea posible, origen: calle Ruzafa  destino: hacia zona Hipercor en Campanar (Valencia) Qué bus puedo coger y a dónde desde la calle Ruzafa.\n', {'entities': [(70, 76, 'ORIGEN'), (78, 90, 'LUGAR1'), (91, 98, 'DESTINO'), (111, 131, 'LUGAR2')]}),
('Buenos días!!  Necesito me informes que autobús puedo coger para ir desde Hipercor hasta el.museo Ivam en Cl. Guillem de Castro.  Valencia. Gracias\n', {'entities': [(59, 66, 'RUTAS'), (67, 72, 'ORIGEN'), (73, 81, 'LUGAR1'), (82, 87, 'DESTINO'), (88, 101, 'DESTINO')]}),
('Cojo el 11 a la altura de número 16 de avda primado reig , voy a Poeta Querol , donde me conviene bajar ? Gracias\n', {'entities': [(8, 10, 'LINEA'), (26, 56, 'LUGAR1'), (63, 64, 'DESTINO'), (65, 77, 'LUGAR2')]}),
('Desde primado reig 16 tengo que ir a la C/ Francisco Sempere número 5 que autobús me viene bien coger\n', {'entities': [(0, 5, 'ORIGEN'), (6, 21, 'LUGAR1'), (22, 34, 'RUTAS'), (35, 36, 'DESTINO'), (37, 69, 'LUGAR2')]}),
]



@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model="es_core_news_md", output_dir="/Users/edugallardopardo/OneDrive/Documentos/TFM/python/SPACY/modelos/EMT_models/model5", n_iter=300):
#def main(model="C:\\Users\\edgal\\OneDrive\\Documentos\\TFM\\python\\SPACY\\modelos\\EMT_models\\model1", 
#output_dir="C:\\Users\\edgal\\OneDrive\\Documentos\\TFM\\python\\SPACY\\modelos\\EMT_models\\model1", n_iter=300):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("es")  # create blank Language class
        print("Created blank 'es' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            print(ent[2])
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
       


if __name__ == "__main__":
    plac.call(main)
