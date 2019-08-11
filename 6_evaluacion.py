#!/usr/bin/env python
# coding: utf-8

# In[3]:


# training data
TRAIN_DATA = [('¿Cuál autobús llega hasta el Tanatorio Municipal?  Gracias.\n', {'entities': [(1, 13, "RUTAS"), (14, 19, "RUTAS"), (20, 25, "DESTINO"), (29, 48, "LUGAR2"), (51, 58, "AGRADECIMIENTO")]}),
('¿En qué calles tiene paradas el 10 por el centro?\n', {'entities': [(8, 14, "ITINERARIO"), (21, 28, "ITINERARIO"), (32, 34, "LINEA"), (42, 48, "LUGAR2")]}),
('Gracias\n', {'entities': [(0, 7, "AGRADECIMIENTO")]}),
('Buenas tardes....a que hora pasa el primer autobús por avda. Constitución-Salesianos\n', {'entities': [(0, 13, "SALUDOINI"), (23, 27, "ESTIMACION"), (31, 37, "ATRIBUTO"), (50, 79, 'LUGAR1')]}),
('Muchas gracias\n', {'entities': [(0, 14, "AGRADECIMIENTO")]}),
('Si no fuera mucho preguntar....me pueden informar de la duración del trayecto hasta la renfe....\n', {'entities': [(41, 49, "INFORMACION"),(56, 64, "DURACION"), (78, 83, "DESTINO"), (87, 92, "LUGAR2")]}),
('Cuanto le queda al bus número 19 para llegar a la parada 722? Muchas gracias 😘\n', {'entities': [(0, 6, "ESTIMACION"), (30, 32, "LINEA"), (38, 44, "ESTIMACION"), (57, 60, "PARADA"), (62, 76, "AGRADECIMIENTO"), (77, 78, "SMILE")]}),
('Parada 710 línea 79 Tiempo de.espera por favor\n', {'entities': [(7, 10, "PARADA"), (17, 19, "LINEA"), (20, 26, "ESTIMACION"), (27, 36, "ESTIMACION"), (37, 46, "EDUCADO")]}),
('Parada 1272 línea 92, tiempo de espera ir favor\n', {'entities': [(7, 11, "PARADA"), (18, 20, "LINEA"), (22, 28, "ESTIMACION"), (32, 38, "ESTIMACION")]}),
('Parada 226 líneas 18 y.90. Tiempo de espera por  favor\n', {'entities': [(7, 10, "PARADA"), (18, 20, "LINEA"), (23, 25, "LINEA"), (27, 33, "ESTIMACION"), (37, 43, "ESTIMACION"), (44, 54, "EDUCADO")]}),
('Buenos dias, anoche hice una recarga y esta mañana no estaba cargado en la tarjeta( la he pasado por el activador)...Número de tarjeta Móbilis: 118692521104', {'entities': [(0, 11, "SALUDO"), (13, 19, "TEMP"), (29, 36, "VENTAONLINE"), (39, 50, "TEMP"), (51, 53, "NEGACION"), (61, 68, "VEMTAONLINE"), (145, 157, "TARJETA")]}),
('442 67\n', {'entities': [(0, 3, "PARADA"), (4, 6, "LINEA")]}),
('258\n', {'entities': [(0, 3, "PARADA")]}),
('Y el siguiente?\n', {'entities': [(4, 14, "ESTIMACION")]}),
('La Parada 1386 funciona con normalidad para la línea 99?\n', {'entities': [(10, 14, "PARADA"), (15, 23, "INCIDENCIAS"), (28, 38, "INCIDENCIAS"), (53, 55, "LINEA")]}),
('Quería consultarles para llegar al cabañal desde el cc arena\n', {'entities': [(7, 19, "RUTAS"), (25, 31, "ORIGEN"), (35, 42, "LUGAR1"), (43, 48, "DESTINO"), (52, 60, "LUGAR2")]}),
('Hora de llegada número 35 parada 1951\n', {'entities': [(0, 4, "ESTIMACION"), (8, 15, "ESTIMACION"), (23, 25, "LINEA"), (26, 32, "ESTIMACION"), (33, 37, "PARADA")]}),
('Buenas noches, para ir a CE monteolivete que autobus he de coger desde nuevo centro?\n', {'entities': [(0, 13, "SALUDOINI"), (15, 24, "RUTAS"), (28, 40, "LUGAR2"), (41, 52, "RUTAS"), (59, 64, "RUTAS"), (65, 70, "ORIGEN"), (71, 83, "LUGAR1")]}),
('para ir a CE monteolivete que autobus he de coger desde nuevo centro?\n', {'entities': [(0, 7, "RUTAS"), (8, 9, "DESTINO"), (10, 25, "LUGAR2"), (44, 49, "RUTAS"), (50, 55, "ORIGEN"), (56, 68, "LUGAR1")]}),
('Ok muchas gracias\n', {'entities': [(0, 2, "AFIRMACION"), (3, 17, "AGRADECIMIENTO")]}),
('Feliz día\n', {'entities': [(0, 9, "EDUCADO")]}),
('Hola\n', {'entities': [(0, 4, "SALUDOINI")]}),
('Buenas tardes\n', {'entities': [(0, 13, "SALUDOINI")]}),
('Queria información sobre el EMT Jove\n', {'entities': [(7, 18, "INFORMACION"), (28, 36, "BONO")]}),
('Y tengo que pedir cita?\n', {'entities': [(12, 22, "OFICINAS")]}),
('Vale, muchas gracias, saludos!\n', {'entities': [(0, 4, "AFIRMACION"), (6, 20, "AGRADECIMIENTO"), (22, 29, "SALUDOFIN")]}),
('Buenas tardes. Acabo de intentar gastar el bono recargado y me dice...... 1 ticket de la reserva.\\nY no funciona. El conductor me ha indicado que no funciona bien. Que me retorneis la recarga y que compre una tarjeta nueva.\n', {'entities': [(0, 13, "SALUDOINI"), (43, 57, "VENTAONLINE"), (101, 112, "INCIDENCIAS"), (171, 180, "RECLAMACION"), (184, 191, "VENTAONLINE")]}),
('El bono sigue sin ir\n', {'entities': [(14, 20, "INCIDENCIAS")]}),
('El conductor me dijo que era la tj estaba defectuosa\n', {'entities': [(42, 52, "RECLAMACION")]}),
('Gracias\n', {'entities': [(0, 7, "AGRADECIMIENTO")]}),
('Buenas que numero es la parada que está al lado del campo del Levante en la avenida alfahuir cual es?\n', {'entities': [(0, 6, "SALUDOINI"), (24, 30, "RUTAS"), (52, 69, "LUGAR2"), (76, 92, "LUGAR2")]}),
('Que te deja en el centro\n', {'entities': [(7, 11, "RUTAS"), (18, 24, "LUGAR2")]}),
('Y el número de parada?\n', {'entities': [(5, 21, "RUTAS")]}),
('Y para ir al campo del Levante también en 70 no?\n', {'entities': [(2, 9, "RUTAS"), (13, 30, "LUGAR2"), (42, 44, "LINEA")]}),
('Hola, buenas tardes. Quería hacerles una consulta. Tengo el bono EMT joven. En marzo cumplo los 25. Puedo seguir recargàndola?\n', {'entities': [(0, 4, "SALUDOINI"), (6, 19, "SALUDOINI"), (41, 49, "INFORMACION"), (65, 74, "BONO")]}),
('Muchísimas gracias\n', {'entities': [(0, 18, "AGRADECIMIENTO")]}),
('Con el carnet jove se queda en 38.25? Hay otra tarjeta que sea más económica?\n', {'entities': [(7, 18, "BONO"), (67, 76, "INFORMACION")]}),
('Una última consulta: me caduca el día 13 de este mes, ahí todavía puedo recargar la jove, no? Porque yo cumplo los 25 el día 7 de marzo\n', {'entities': [(11, 19, "INFORMACION"), (81, 88, "BONO")]}),
('Buenas tardes,necesito ir de periodista Gil Sumbila a la cruz roja\n', {'entities': [(15, 25, "RUTAS"), (26, 28, "ORIGEN"), (29, 51, "LUGAR1"), (52, 53, "DESTINO"), (57, 66, "LUGAR2")]}),
('Calle Alboraya\n', {'entities': [(0, 14, "LUGAR2")]}),
('Buenas tardes me podrían informar por donde va a pasar la línea 14 me encuentro en la gran via.\n', {'entities': [(0, 13, "SALUDOINI"), (25, 33, "INFORMACION"), (38, 43, "RUTAS"), (47, 54, "RUTAS"), (64, 66, "LINEA"), (70, 79, "ORIGEN"), (86, 94, "LUGAR1")]}),
('Horno de alcedo\n', {'entities': [(0, 15, "LUGAR2")]}),
('Hola buenas tardes me podrías decir a que hora pasa la línea 14 horno de alcedo  me encuentro en el cementerio de sedavid .gracias\n', {'entities': [(0, 4, "SALUDOINI"), (5, 18, "SALUDOINI"), (30, 35, "INFORMACION"), (42, 46, "ESTIMACION"), (47, 51, "ESTIMACION"), (61, 63, "LINEA"), (82, 96, "ORIGEN"), (100, 121, "LUGAR")]}),
('Buenos días desde Malvarrosa que línea tengo que coger para ir a Alameda 12\n', {'entities': [(0, 11, "SALUDOINI"), (12, 17, "ORIGEN"), (18, 28, "LUGAR1"), (33, 38, "INFORMACION"), (39, 48, "RUTAS"), (55, 62, "RUTAS"), (63, 64, "DESTINO"), (65, 72, "LUGAR2")]}),
('Para ir a la boca del metro de Ayora desde la calle doctor Vicente pallarés iranzo 4\n', {'entities': [(0, 7, "RUTAS"), (8, 9, "DESTINO"), (23, 36, "LUGAR2"), (37, 42, "ORIGEN"), (52, 84, "LUGAR1")]}),
('Estoy en plaza España donde puedo coger el 60 que va ala quad?\n', {'entities': [(0, 8, "ORIGEN"), (9, 21, "LUGAR1"), (34, 39, "RUTAS"), (43, 45, "LINEA"), (50, 52, "DESTINO")]}),
('hola buenas tardes, una pregunta, como se llama la parada siguiente de la de burjasot zaidia en la linea 95?\n', {'entities': [(20, 32, "INFORMACION"), (51, 67, "ITINERARIO"), (77, 92, "LUGAR"), (105, 107, "LINEA")]}),
('okey, gracias\n', {'entities': [(0, 4, "AFIRMACION"), (6, 13, "AGRADECIMIENTO")]}),
('casi me olvido de preguntarlo, a estas alturas, hay algun bus que pase cerca de las playas?\n', {'entities': [(18, 29, "INFORMACION"), (58, 61, "RUTAS"), (66, 70, "RUTAS"), (71, 76, "ATRIBUTO"), (84, 90, "LUGAR2")]}),
('malvarrosa\n', {'entities': [(0, 10, "LUGAR2")]}),
('desde la parada, pla de la zaidia- burjassot\n', {'entities': [(0, 5, "ORIGEN"), (9, 15, "INFORMACION"), (17, 44, "LUGAR1")]}),
('cual es la parada de poeta fernandez heredia?\n', {'entities': [(0, 4, "INFORMACION"), (11, 17, "INFORMACION"), (21, 44, "LUGAR2")]}),
('para ir a malvarrosa tanbien me vale si lo coo en periodista llorente?\n', {'entities': [(0, 7, "RUTAS"), (8, 9, "DESTINO"), (10, 20, "LUGAR1"), (47, 49, "DESTINO"), (50, 69, "LUGAR2")]}),
('y que linea llega a la playa entonces?\n', {'entities': [(2, 11, "RUTAS"), (12, 19, "DESTINO"), (23, 28, "LUGAR2")]}),
('si cojo mañana en bus en  alqueria de estrella e dejará en malvarrosa?\n', {'entities': [(3, 7, "RUTAS"), (8, 14, "TEMP"), (22, 24, "ORIGEN"), (27, 46, "LUGAR1"), (49, 58, "DESTINO"),(59, 69, "LUGAR2")]}),
('Hola, me he dejado olvidada una bolsa en el bus a mediodia\n', {'entities': [(0, 4, "SALUDOINI"), (12, 18, "OBJETOS"), (19, 27, "OBJETOS"), (32, 37, "QUE"), (44, 47, "DONDE"), (48, 58, "TEMP")]}),
('A qué número tengo que llamar o lo puedo gestionar por aquí\n', {'entities': [(6, 12, "TELEFONO"), (23, 29, "INFORMACION")]}),
('Me gustaría saber si puedo viajar con mi perro cachorrito en el autobús\n', {'entities': [(3, 17, "INFORMACION"), (41, 57, "BONO")]}),
('Q autobús puedo cojer desde p.reig -bilbao a piscina de valencia?gracias\n', {'entities': [(10, 21, "RUTAS"), (22, 27, "ORIGEN"), (28, 34, "LUGAR1"), (43, 44, "DESTINO"), (45, 72, "LUGAR2")]}),
('Buenos días.q autobús cojo desde la estación del norte a la estación de autobuses?\n', {'entities': [(22, 26, "RUTAS"), (27, 32, "ORIGEN"), (36, 54, "LUGAR1"), (55, 56, "DESTINO"), (60, 81, "LUGAR2")]}),
('GraciasBuenas tardes.\\nQue linea de bus puedo cojer desde Plz. Ayuntamiento para ir a la exposición del ninot en la Cdad. de las Artes?\n', {'entities': [(46, 51, "RUTAS"), (52, 57, "ORIGEN"), (58, 75, "LUGAR1"), (76, 83, "RUTAS"), (84, 85, "DESTINO"), (116, 134, "LUGAR2")]}),
('Buenas tardes.\\nPara ir de Plza España a la Escuela Oficial de Idiomas (llano de Zaidia) que linea de bus puedo cojer?\n', {'entities': [(21, 23, "RUTAS"), (24, 26, "ORIGEN"), (27, 38, "LUGAR1"), (39, 40, "DESTINO"), (44, 70, "LUGAR2")]}),
('Hablé con ustedes el viernes 5 de abril porque perdí mi bufanda en uno de sus autobuses de EMT\n', {'entities': [(47, 52, "OBJETOS"), (56, 63, "QUE"), (78, 87, "DONDE")]}),
('Quería saber cómo puedo llegar a la Av. de Jesús Morante Borrás, 214 desde el centro de Valencia\n', {'entities': [(7, 12, "INFORMACION"), (24, 30, "RUTAS"), (31, 32, "DESTINO"), (36, 68, "LUGAR2"), (69, 74, "ORIGEN"), (78, 96, "LUGAR1")]}),
('¿Como se llama la parada del bus 95 más próxima a la avenida Jesús Morante?\n', {'entities': [(9, 14, "INFORMACION"), (18, 24, "INFORMACION"), (33, 35, "LINEA"), (36, 47, "ATRIBUTO"), (53, 74, "LUGAR")]}),
('Un saludo\n', {'entities': [(0, 8, "SALUDOFIN")]}),
('Gracias.Buenos días!  Me gustaría recibir información por favor. Si llego a la estación del norte tengo una linea de autobús que me lleve al hospital general.  Muchas gracias\n', {'entities': [(34, 53, "INFORMACION"), (68, 73, "RUTAS"), (79, 97, "LUGAR1"), (132, 137, "DESTINO"), (141, 157, "LUGAR2")]}),
('Cada cuánto salen?\n', {'entities': [(12, 17, "ESTIMACION")]}),
('Muy amable\n', {'entities': [(0, 10, "AGRADECIMIENTO")]}),
('Después del hospital general  cojo la misma linea para volver a la estación?\n', {'entities': [(12, 28, "LUGAR1"), (31, 34, "RUTAS"), (55, 61, "RUTAS"), (67, 75, "LUGAR2")]}),
('Hola para ir desde la calle doctor sumsi hasta la estación de autobuses? Gracias!\n', {'entities': [(5, 12, "RUTAS"), (13, 18, "ORIGEN"), (22, 40, "LUGAR1"), (41, 46, "DESTINO"), (50, 71, "LUGAR2")]}),
('Hola\\n\\nEn la aplicación emt todos los autobuses pone próximo. Lo digo para que lo mireis\n', {'entities': [(14, 28, "DONDE"), (83, 89, "INCIDENCIAS")]}),
('Me parece una vergüenza que no dispongas de cambio de 10€ y me toque ir andando a trabajar. Un sábado a las 16.45h. el número 28 (5146)\n', {'entities': [(14, 23, "QUEJA"), (44, 50, "QUE"), (126, 128, "LINEA")]}),
('Hola, hoy hay huelga?\n', {'entities': [(14, 20, "INCIDENCIAS")]}),
('Hola  .el sábado  hay desvío de  autobuses  por lo de los chinos?\n', {'entities': [(10, 16, "TEMP"), (22, 28, "INCIDENCIAS")]}),
('Gracias.Bueno días , quería informaros de que últimamente el 62 se está retrasando y no cumple su horario. Cada día llega más tarde. Siempre ha llegado a las 7:50 y ahora llega a las 8:05-8:10. Y el anterior a este pasa a las 7:40, con unos 20 minutos entre bus y bus. Por favor que pasen mas buses\n', {'entities': [(21, 38, "RECLAMACION"), (46, 57, "ATRIBUTO"), (61, 63, "LINEA"), (72, 82, "QUE"), (85, 94, "ATRIBUTO")]}),
('Buenos dias, algun problema con el 26? En la parada 1137 no ha pasado todavia el de las 6.40\n', {'entities': [(19, 27, "INCIDENCIAS"), (34, 37, "LINEA"), (52, 56, "PARADA"), (70, 77, "QUE")]}),
('Buenas tardes.Para desplazarme  desde Benicalap ( Avenidas Burjasot - Peset Aleixandre - General Avilés- Campanar ) hasta la C / Santiago Ruiseñol ( Junto Campo Fútbol \\Levante\\ ), ¿ qué puedo hacer ?. Preferiblemente, sin tener que hacer transbordo si no hay que andar mucho. En caso contrario, sin problema en hacer transbordo. \\nGracias y saludos.\n', {'entities': [(19, 30, "RUTAS"), (33, 37, "ORIGEN"), (38, 47, "LUGAR1"), (116, 121, "DESTINO"), (129, 146, "LUGAR2"), (149, 154, "ATRIBUTO"), (155, 177, "LUGAR3")]}),
('La Parada Primat Reig - Ministre Luís Mayans no aparece como tal. ¿ Se tratará de un error involuntario ?. Gracias y saludos.\n', {'entities': [(3, 9, "RECLAMACION"), (10, 44, "PARADA"), (45, 55, "QUE"), (85, 90, "INCIDENCIAS")]}),
('Reitero gratitud y saludos. Buenas tardes.\n', {'entities': [(8, 16, "AGRADECIMIENTO")]}),
('Buenas tardes.\\nPara desplazarme  desde Benicalap ( Avenidas Burjasot - Peset Aleixandre - General Avilés- Campanar ) hasta la Plz. Conde de Carlet 3, ¿ qué puedo hacer ?. Preferiblemente, sin tener que hacer transbordo si no hay que andar mucho. En caso contrario, sin problema en hacer transbordo. \\nGracias y saludos.\n', {'entities': [(21, 32, "RUTAS"), (35, 39, "ORIGEN"), (40, 49, "LUGAR1"), (118, 123, "DESTINO"), (127, 149, "LUGAR2")]}),
('Buenos días. Para desplazarme desde Benicalap  ( Avenidas Burjasot - Peset Aleixandre - General Avilés- Campanar ) hasta la Avda. Pío Baroja 12 \n', {'entities': [(18, 29, "RUTAS"), (30, 35, "ORIGEN"), (36, 45, "LUGAR1"), (115, 120, "DESTINO"), (124, 143, "LUGAR2")]}),
('Llevo esperando 10 min al 93 en la segunda parada desde donde hace descanso e. la avenida del cid, iba a pasar el 93 en 5 min y pone ahora que se va a cochera y me marca 7 minutos\n', {'entities': [(6, 15, "RECLAMACION"), (16, 22, "TEMP"), (26, 28, "LINEA"), (151, 158, "QUE")]}),
('Los horarios de los buses no cooinciden\n', {'entities': [(4, 12, "INCIDENCIAS"), (26, 39, "QUE")]}),
('Perdon llevo esperando 20 minutos\n', {'entities': [(0, 6, "EDUCADO"), (7, 22, "RECLAMACION"),(23, 33, "QUE")]}),
("Disculpa, el n 14 forn d'alcedo lleva a la punta?\n", {'entities': [(0, 8, "EDUCADO"), (15, 17, "LINEA"), (32, 37, "RUTAS"), (38, 39, "DESTINO"), (40, 48, "LUGAR2")]}),
('Multiespai la punta *\n', {'entities': [(0, 19, "LUGAR")]}),
('Donde para el 95?\n', {'entities': [(0, 5, "INFORMACION"), (6, 10, "ITINERARIO"), (14, 16, "LINEA")]}),
('Si, quería saber en la parada 723, el bus 19, cuando dice próximo cuanto tiempo hay que esperar? Lo comento porque indicaba próximo esta mañana y espere 10 minutos, gracias\n', {'entities': [(4, 16, "INFORMACION"), (30, 33, "PARADA"), (42, 44, "LINEA"), (46, 52, "ESTIMACION"), (58, 65, "ESTIMACION"), (73, 79, "ESTIMACION"), (88, 95, "INCIDENCIAS")]}),
('OK gracias\n', {'entities': [(0, 2, "AFIRMACION"), (3, 10, "AGRADECIMIENTO")]}),
('Buenos dias me gustaria saber si la linea 32 para en la plaza de tetuan gracias\n', {'entities': [(24, 29, "RUTAS"), (42, 44, "LINEA"), (45, 49, "ITINERARIO"), (56, 71, "LUGAR1")]}),
('Gracias que pases un buen dia\n', {'entities': [(0, 7, "AGRADECIMIENTO"), (12, 29, "EDUCADO")]}),
('Buenos días. Deseo ir desde la Patacona al Hospital Arnau de Vilanova\n', {'entities': [(13, 21, "RUTAS"), (22, 27, "ORIGEN"), (28, 39, "LUGAR1"), (40, 42, "DESTINO"), (43, 69, "LUGAR2")]}),
('Cojo linea 31, ¿Desde bajo y cojo otro?\n', {'entities': [(11, 13, 'LINEA')]}),
('Hola buenos días. Un teléfono de contacto de objetos perdidos tenéis? He extraviado la cartera, por saber si la habrían llevado allí\n', {'entities': [(21, 29, 'CONTACTO'), (33, 41, 'CONTACTO'), (53, 61, 'OBJETOS'), (73, 83, 'OBJETOS'), (87, 94, 'QUE'), (128, 132, 'EMT')]}),
('El martes me di cuenta ayer por la noche que no la tenía\n', {'entities': [(3, 9, 'TEMP')]}),
('En el número 26\n', {'entities': [(13, 15, 'LINEA')]}),
('Sobre las 21h de la noche lo cogí en el centro\n', {'entities': [(10, 13, 'TEMP'), (37, 46, 'LUGAR')]}),
('Hola buenos días. Por favor para ir de peset Aleixandre a calle Albacete, que autobús puedo coger?\n', {'entities': [(28, 35, 'RUTAS'), (36, 38, 'ORIGEN'), (39, 55, 'LUGAR1'), (56, 57, 'DESTINO'), (58, 73, 'LUGAR2')]}),
('Muchas gracias. Igualmente Cuando pasa el próximo 99 en la parada 1272\n', {'entities': [(27, 38, 'ESTIMACION'), (42, 49, 'ATRIBUTO'), (50, 52, 'LINEA'), (68, 70, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 1272\n', {'entities': [(0, 11, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 43, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(0, 11, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(0, 11, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(0, 11, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(7, 11, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(7, 11, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(7, 11, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 89 en la parada 1679\n', {'entities': [(7, 11, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 43, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 515\n', {'entities': [(0, 6, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Y el próximo\n', {'entities': [(0, 1, 'OTRAC'), (5, 12, 'ATRIBUTO')]}),
('Cuando pasa el próximo 99 en la parada 514\n', {'entities': [(0, 6, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 514\n', {'entities': [(0, 6, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Y el próximo\n', {'entities': [(0, 1, 'OTRAC'), (5, 12, 'ATRIBUTO')]}),
('Cuando pasa el próximo 99 en la parada 514\n', {'entities': [(0, 6, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 514\n', {'entities': [(0, 6, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Cuando pasa el próximo 99 en la parada 514\n', {'entities': [(0, 6, 'ESTIMACION'), (15, 22, 'ATRIBUTO'), (23, 25, 'LINEA'), (39, 42, 'PARADA')]}),
('Autobus de Angel guimerà a Hospital Clinico\n', {'entities': [(8, 10, 'ORIGEN'), (11, 24, 'LUGAR1'), (25, 26, 'DESTINO'), (27, 43, 'LUGAR2')]}),
('Buenas tardes, me gustaria saber la localización  de objetos perdidos. Creo que perdi mi bonobus en un autobús  y me gustaria saber si esta alli;\n', {'entities': [(36, 48, 'CONTACTO'), (52, 59, 'OBJETOS'), (60, 68, 'OBJETOS'), (79, 84, 'OBJETOS'), (88, 95, 'QUE'), (138, 142, 'EMT')]}),
('Fue con la linea 71;\n', {'entities': [(17, 19, 'LINEA')]}),
('Gracias, sabes si atencion al cliente  para sacar duplicado, esta por las tardes?;\n', {'entities': [(18, 37, 'CONTACTO'), (43, 48, 'QUE'), (49, 58, 'ATRIBUTO')]}),
('Buenos dias, me gustaria preguntar por una tarjeta perdida en la linea 71, el viernes  a las 14:30h.;\n', {'entities': [(25, 38, 'INFORMACION'), (43, 50, 'QUE'), (51, 58, 'OBJETOS'), (71, 73, 'LINEA'), (78, 85, 'TEMP')]}),
('Gracias!;Hola buenos días que autobús va al local Jerusalémpop&Rock desde el ayuntamiento\n', {'entities': [(38, 40, 'RUTAS'), (41, 43, 'DESTINO'), (50, 59, 'LUGAR1'), (68, 73, 'DESTINO'), (77, 89, 'LUGAR2')]}),
('Si C/convento Jerusalén,55\n', {'entities': [(3, 26, 'LUGAR2')]}),
('Ah perfecto\n', {'entities': [(3, 11, 'AGRADECIMIENTO')]}),
('Buenos días. Para no faltar a la costumbre, la app no da los horarios desde Poeta Querol bus 26 hacia Moncada\n', {'entities': [(47, 50, 'QUE'), (51, 53, 'QUEJA')]}),
('Porfavor el 1764\n', {'entities': [(0, 8, 'EDUCADO'), (12, 16, 'PARADA')]}),
('Cuento tarde el 95\n', {'entities': [(7, 12, 'QUEJA'), (16, 18, 'LINEA')]}),
('Buenos días porfavor 559\n', {'entities': [(21, 24, 'PARADA')]}),
('Cuánto tarda\n', {'entities': [(0, 12, 'ESTIMACION')]}),
('1764.    95\n', {'entities': [(0, 4, 'PARADA'), (8, 10, 'LINEA')]}),
('Hola  buenas 1764 95\n', {'entities': [(12, 16, 'PARADA'), (17, 19, 'LINEA')]}),
('Cuánto tarda gracias\n', {'entities': [(0, 12, 'ESTIMACION')]}),
('1764.   95\n', {'entities': [(0, 4, 'PARADA'), (8, 10, 'LINEA')]}),
('Buenas tardes...¿¿Como puedo ir desde la calle Salvador Giner a la calle San Juan de la Peña??\n', {'entities': [(29, 31, 'RUTAS'), (32, 37, 'ORIGEN'), (41, 61, 'LUGAR1'), (62, 63, 'DESTINO'), (73, 92, 'LUGAR2')]}),
('Para ir a port saplaya que bus cojo?\n', {'entities': [(0, 7, 'RUTAS'), (8, 9, 'DESTINO'), (10, 22, 'LUGAR2')]}),
('Estoy en la calle jativa\n', {'entities': [(0, 5, 'ORIGEN'), (18, 24, 'LUGAR1')]}),
('Hola Quiero ir a la calle San Vicente,  número 312, estoy en la plaza los pinazo\n', {'entities': [(5, 14, 'RUTAS'), (15, 16, 'DESTINO'), (26, 50, 'LUGAR2'), (52, 57, 'ORIGEN'), (64, 80, 'LUGAR1')]}),
('Quiero ir a peset aleixandre, 110\n', {'entities': [(7, 9, 'RUTAS'), (10, 11, 'DESTINO'), (12, 33, 'LUGAR2')]}),
('Estoy en la calle jativa\n', {'entities': [(0, 5, 'ORIGEN'), (18, 24, 'LUGAR1')]}),
('Hola buenas estoy en la parada número 2127 y llevo 20 min esperando al 8 ha pasado uno vacío con fuera de servicio y otro a mitad también con fuera de servicio\n', {'entities': [(12, 17, 'ORIGEN'), (38, 42, 'PARADA'), (58, 67, 'QUEJA'), (71, 72, 'LINEA')]}),
('Hay huelga o algo por el estilo?\n', {'entities': [(4, 10, 'INCIDENCIAS')]}),
('La parada 958 sigue meses sin funcionar\n', {'entities': [(10, 13, 'PARADA'), (26, 39, 'QUEJA')]}),
('Hola. Mi tarjeta es 2400 2616 5947. La he recargado de forma online este mediodía. Se han recargado sin problema en un autobús de la línea 10, y al cogerlo ahora en la línea 9 y pasarlo por los lectores muchas veces, en todas pone que el título está agotado\n', {'entities': [(20, 34, 'TARJETA'), (42, 51, 'VENTAONLINE'), (61, 67, 'VENTAONLINE'), (90, 99, 'VENTAONLINE'), (174, 175, 'LINEA'), (250, 257, 'QUE')]}),
('Estoy detrás de la plaza de toros y voy a coger el 8\n', {'entities': [(0, 5, 'ORIGEN'), (19, 33, 'LUGAR1'), (36, 47, 'DESTINO'), (51, 52, 'LINEA')]}),
('A qué hora pasa?\n', {'entities': [(6, 10, 'ESTIMACION'), (11, 16, 'ESTIMACION')]}),
('Hola buenas Quiero hacerles un comentario sobre la línea 64 Creo que es la línea donde tratan peor a los seres humanos, realmente los tratan como borregos. Es una línea que siempre va llena con exceso de hecho en bastantes ocasiones hay paradas que ni abre llevando incluso el cartel de COMPLETO y en alguna ocasión ha sucedido esto con 2 bus continuos Sinceramente pienso que harán un estudio de servicio o como se llame en su caso y habrán comprobado que esto sucede. No se han planteado poner esta línea como bus doble No es una línea de paseo o compras al centro. Es una línea con destino LA FE lo que significa que la utilizan usuarios de alguna forma delicados y esto se merece tener un poco de humanidad haciéndoles más fácil el trayecto Gracias por su atención\n', {'entities': [(28, 41, 'RECLAMACION'), (57, 59, 'LINEA'), (87, 98, 'QUEJA'),(146, 154, 'QUEJA')]}),
('Por favor, que bus tengo que coger para ir de calle Yecla a calle Retor 44\n', {'entities': [(40, 42, 'RUTAS'), (43, 45, 'ORIGEN'), (46, 57, 'LUGAR1'), (58, 59, 'DESTINO'), (60, 74, 'LUGAR2')]}),
('Buenos días, necesito ir a Marqués de Lozoya, 8 desde Ángel Guimerá esquina Fdo. El Católico ¿Que autobús o autobuses he de coger? Gracias\n', {'entities': [(22, 24, 'RUTAS'), (25, 26, 'DESTINO'), (27, 47, 'LUGAR2'), (48, 53, 'ORIGEN'), (54, 92, 'LUGAR1')]}),
('Pensé que con un solo bus llegaría aunque ande un poco. Ya q han dicho está x c.c. Saler y allí no se cual va ni dónde se coge\n', {'entities': [(78, 88, 'LUGAR'), (96, 101, 'QUEJA')]}),
('Buenas.Vpel la linea 92.Numero 8012.Direccion av Puerto.Al final del bus hay dos chicas tomando alcohol.Han sacado la botella y se lo han servido y todo es indignante.\n', {'entities': [(21, 23, 'LINEA'), (96, 103, 'INCIDENCIAS'), (156, 166, 'QUEJA')]}),
("Bon dia, volia saber quina linea he d'agafar per anar des de la UPV a l'Edtació del Nord\n", {'entities': [(0, 8, 'SALUDOINIVAL'), (49, 53, 'RUTAS'), (54, 60, 'ORIGEN'), (64, 67, 'LUGAR1'), (68, 69, 'DESTINO'), (70, 88, "LUGAR2")]}),
('Moltes gracies\n', {'entities': [(0, 14, 'AGRADECIMIENTOVAL')]}),
('Buenas tardes,estamos en la Calle Castellón, para ir a Mestalla q Bus hay q coger. Gracias.\n', {'entities': [(14, 21, 'ORIGEN'), (28, 44, 'LUGAR1'), (50, 52, 'RUTAS'), (53, 54, 'DESTINO'), (55, 63, 'LUGAR2')]}),
('Buenos días, vivimos en la calle Castellón queremos coger un Bus para ir al Museo San Pío V.Gracias.\n', {'entities': [(13, 20, 'ORIGEN'), (27, 42, 'LUGAR1'), (65, 72, 'RUTAS'), (73, 75, 'DESTINO'), (76, 91, 'LUGAR2')]}),
('Buenos días,quisiera ir a la C.Jesus desde la Castellón, gracias.\n', {'entities': [(21, 23, 'RUTAS'), (24, 25, 'DESTINO'), (29, 36, 'LUGAR2'), (37, 42, 'ORIGEN'), (43, 45, 'LUGAR1')]}),
('Hola, sé que hay una línea vuestra que puedo coger en la zona de plaza España y que me acerca a Malilla. Me podéis decir qué línea es? Es que no la recuerdo. Gracias.\n', {'entities': [(65, 77, 'LUGAR1'), (87, 93, 'RUTAS'), (94, 95, 'DESTINO'), (96, 103, 'LUGAR2')]}),
('Que buses puedo coger para ir a la plaza de la virgen o a la calle de la paz?\n', {'entities': [(27, 29, 'RUTAS'), (30, 31, 'DESTINO'), (35, 53, 'LUGAR2'), (70, 77, 'LUGAR2')]}),
('Hola buenos días quisiera hacer una consulta la targeta de bonobus si la recargo para metrobus qué zona podría coger con dicha targeta\n', {'entities': [(0, 4, 'SALUDOINI'), (5, 16, 'SALUDOINI'), (36, 44, 'INFORMACION'), (59, 66, 'BONO'), (73, 80, 'VENTAONLINE'), (99, 103, 'QUE')]}),
('Buenos días qué línea puedo coger para ir de Zapadores esquina con Peris y Valero a lanzadera en el puerto de Valencia?\n', {'entities': [(34, 41, 'RUTAS'), (42, 44, 'ORIGEN'), (45, 81, 'LUGAR1'), (82, 83, 'DESTINO'), (84, 118, 'LUGAR2')]}),
('Una pregunta con un billete sencillo símbolo bus bueno volver a coger otra línea durante la primera hora o eso solo funciona con el bono bus?\n', {'entities': [(20, 36, 'BONO'), (111, 124, 'INFORMACION')]}),
('A que hora salen los últimos autobuses de la linea 27 desde la parada Jesus Pare Jofre?\n', {'entities': [(6, 10, 'ESTIMACION'), (21, 28, 'ATRIBUTO'), (51, 53, 'LINEA'), (70, 87, 'PARADA')]}),
('Cuanto le falta al 71 para llegar a la parada 534?\n', {'entities': [(10, 15, 'ESTIMACION'), (19, 21, 'LINEA'), (46, 49, 'PARADA')]}),
('Cuanto le falta al 93 para llegar a la parada 700\n', {'entities': [(10, 15, 'ESTIMACION'), (19, 21, 'LINEA'), (46, 49, 'PARADA')]}),
('Hola, hoy la línea 72 entra a plaza Ayuntamiento?\n', {'entities': [(6, 9, 'TEMP'), (19, 21, 'LINEA'), (22, 27, 'INCIDENCIAS'), (30, 48, 'LUGAR')]}),
('Cuanto le falta al 71 para llegar a la parada 534?\n', {'entities': [(10, 15, 'ESTIMACION'), (19, 21, 'LINEA'), (46, 49, 'PARADA')]}),
('Buenas noche disculpe la pregunta estoy en alfafar quiero ir a la estación de autobús que línea debo tomar\n', {'entities': [(0, 12, 'SALUDOINI'), (34, 42, 'ORIGEN'), (43, 50, 'LUGAR1'), (51, 60, 'RUTAS'), (61, 62, 'DESTINO'), (66, 85, 'LUGAR2')]}),
('Para ir a la estación de bus\n', {'entities': [(0, 7, 'RUTAS'), (8, 9, 'DESTINO'), (13, 28, 'LUGAR2')]}),
('Hola. Estoy en Manuel Candela con Port, parada 1110. El bus del 40 ha estado 7 minutos parado unos metros antes\n', {'entities': [(6, 11, 'ORIGEN'), (15, 39, 'LUGAR1'), (47, 51, 'PARADA'), (64, 66, 'LINEA'), (77, 86, 'TEMP'), (87, 93, 'QUEJA')]}),
('Que auto bus pasa por JJDomine a la altura del paseo maritimo\n', {'entities': [(13, 21, 'RUTAS'), (22, 30, 'LUGAR1')]}),
('Aver si poneis 99 paseig maritmo en verano donde para 93\n', {'entities': [(5, 14, 'QUEJA'), (15, 17, 'LINEA'), (18, 32, 'LUGAR'), (54, 56, 'LINEA')]}),
('Una pregunta en la parada del perello 2186 en el lector QR sale esto\n', {'entities': [(38, 42, 'PARADA'), (49, 58, 'QUE'), (59, 63, 'INCIDENCIAS')]}),
('Una cosita en la aplicación tampoco sale los horarios de la parada del perello\n', {'entities': [(17, 27, 'QUE'), (28, 40, 'INCIDENCIAS'), (71, 78, 'PARADA')]}),
('Y una última cosa\n', {'entities': [(0, 1, 'OTRAC')]}),
('Tengo una tarjeta que me pone incidencia\n', {'entities': [(30, 40, 'INCIDENCIAS')]}),
('330941019057\n', {'entities': [(0, 12, 'TARJETA')]}),
('Si gracias me lo acaban de decir Hola ! Que bus nocturno puedo coger por la zona de cortes valencianas paa ir al paseo de la pechina o alrededores ?\n', {'entities': [(48, 56, 'ATRIBUTO'), (84, 102, 'LUGAR1'), (107, 109, 'RUTAS'), (110, 112, 'DESTINO'), (113, 132, 'LUGAR2')]}),
('Estoy en la 1695\n', {'entities': [(0, 5, 'ORIGEN'), (12, 16, 'PARADA')]}),
('Cuánto le queda al 90?\n', {'entities': [(0, 15, 'ESTIMACION'), (19, 22, 'LINEA')]}),
('Gracias a ti, feliz finde!CONFIRMAR EL HORARIO DEL 73 EN LA PARADA DE SAN ISIDRO EL DE LAS 8:48H .\n', {'entities': [(0, 13, 'AGRADECIMIENTO'), (14, 26, 'EDUCADO'),(39, 46, 'ESTIMACION'), (51, 53, 'LINEA'), (70, 80, 'PARADA')]}),
('HABER SI ES NORMAL\n', {'entities': [(12, 18, 'QUEJA')]}),
('LUNES Y MIÉRCOLES NINGUNO PASÓ\n', {'entities': [(18, 30, 'QUEJA')]}),
('DONDE PUEDO PONER UNA RECLAMACIÓN??\n', {'entities': [(22, 33, 'RECLAMACION')]}),
('Y POR SUPUESTO QUE DEJO LA QUEJA ...SERVIRA PARA ALGO ...???\n', {'entities': [(0, 1, 'OTRAC'), (27, 32, 'QUEJA')]}),
('Que bus he de coger desde Tarongers para llegar a la Avenida Alfahuir?\n', {'entities': [(20, 25, 'ORIGEN'), (26, 35, 'LUGAR1'), (41, 47, 'RUTAS'), (48, 49, 'DESTINO'), (53, 69, 'LUGAR2')]}),
('Alfahuir 39\n', {'entities': [(0, 11, 'LUGAR2')]}),
('Me puedes decir el número del autobus y de la parada  para ir a zapadores a la comisaría\n', {'entities': [(19, 37, 'RUTAS'), (46, 52, 'INFORMACION'), (58, 60, 'RUTAS'), (61, 62, 'DESTINO'), (63, 72, 'LUGAR2')]}),
('Desde tavernes blanques\n', {'entities': [(0, 5, 'ORIGEN'), (6, 23, 'LUGAR1')]}), 
('Gracias\n', {'entities': [(0, 7, 'AGRADECIMIENTOS')]}),
('Buenos días, tengo una consulta\n', {'entities': [(23, 31, 'INFORMACION')]}),
('Eso ya lo hice el viernes en la noche y aún nada. Por eso escribo, gracias\n', {'entities': [(18, 25, 'TEMP'), (40, 49, 'QUEJA')]}),
('Entiendo, gracias 👍\n', {'entities': [(18, 19, 'AFIRMACION')]}),
('A que hora cierra la oficina de la emt\n', {'entities': [(21, 28, 'CONTACTO'), (35, 38, 'EMT')]}),
('Buenas tardes. Tengo un hijo de 14 años con un grado de discapacidad del 37.Podria optar al bono oro? O a algún tipo de descuento. Gracias\n', {'entities': [(92, 101, 'BONO')]}),
('Necesito ir a la Calle del Impresor Monfort número 8 y estoy en la calle Luis Bolinches número 20\n', {'entities': [(0, 11, 'RUTAS'), (12, 13, 'DESTINO'), (27, 52, 'LUGAR2'), (55, 63, 'ORIGEN'), (73, 97, 'LUGAR1')]}),
('Gracias y muy amable\n', {'entities': [(10, 20, 'AGRADECIMIENTO')]}),
('Como puedo ir desde pont de fusta a c democracia o cerca de ahí?\n', {'entities': [(11, 13, 'RUTAS'), (14, 19, 'ORIGEN'), (20, 33, 'LUGAR1'), (34, 35, 'DESTINO'), (36, 48, 'LUGAR2'), (51, 56, 'ATRIBUTO')]}),
('Buenas, me podrian decir cuanto le falta a la linea 19 en la parada 723? Gracias\n', {'entities': [(35, 40, 'ESTIMACION'), (52, 54, 'LINEA'), (68, 71, 'PARADA')]}),
('El autobus municipal de alboraya tambien lo lleva emt?\n', {'entities': [(50, 53, 'EMT')]}),
('Buenas tardes más que una reclamación he hecho una felicitación por el formulario para el conductor que salvo del robo para que quede constancia un saludo\n', {'entities': [(51, 63, 'FELICITACION')]}),
('Pero que quede constancia por favor si tenemos que hacer reclamación tambien tenemos derecho a felicitar las buenas obras\n', {'entities': [(26, 35, 'EDUCADO'), (95, 104, 'FELICITACION')]}),
('A vosotros un saludo 😘\n', {'entities': [(21, 22, 'SMILEKISS')]}),
('Buenas tardes horario comenzó línea 93 desde serrería a avenida del cid\n', {'entities': [(0, 13, 'SALUDOINI'), (14, 21, 'ITINERARIO'), (36, 38, 'LINEA'), (39, 44, 'ORIGEN'), (45, 53, 'LUGAR1'), (54, 55, 'DESTINO'), (56, 71, 'LUGAR2')]}),
('Hay alguna que desde serrería o cabañal llegue cerca de aragon aunque sea la 92\n', {'entities': [(15, 20, 'ORIGEN'), (21, 29, 'LUGAR1'), (47, 52, 'DESTINO'), (56, 62, 'LUGAR2'), (77, 79, 'LINEA')]}),
('Buenas tardes mi Bono oro 3777 2496 8222,  q ayer 12 active formalizando todos los requisitos, he querido pagar hoy la cuota y el terminal indica q faltan requisitos.', {'entities': [(17, 25, 'BONO'), (26, 41, 'TARJETA'), (44, 48, 'TEMP')]}),
('Buenos días quiero saber cuando puedo pagar el bono oro? Desde el 12 y estamos a 18 y no puedo acceder al bono oro, me pueden decir algo?;\n', {'entities': [(12, 24, 'INFORMACION'), (47, 55, 'BONO')]}),
('Buenas! Como puedo llegar a mya ?\n', {'entities': [(0, 7, 'SALUDOINI'), (19, 25, 'RUTAS'), (26, 27, 'DESTINO'), (28, 31, 'LUGAR2')]}),
('Porta de la mar\n', {'entities': [(0, 15, 'LUGAR1')]}),
('Hola, qué bus va desde Hipercor hasta la parada de metro 9 de octubre? Es para el sábado y horarios por favor\n', {'entities': [(14, 16, 'RUTAS'), (17, 22, 'ORIGEN'), (23, 31, 'LUGAR1'), (32, 37, 'DESTINO'), (51, 70, 'LUGAR2')]}),
('Pasa cada media hora?\n', {'entities': [(0, 4, 'ESTIMACION')]}),
('Puedes pasarme el horario? Tan pronto no quiero coger el bus\n', {'entities': [(18, 25, 'ITINERARIO')]}),
('Bon dia\n', {'entities': [(0, 7, 'SALUDOINIVAL')]}),
('Hay algún problema con el 10 en Benimaclet\n', {'entities': [(10, 18, 'INCIDENCIAS'), (26, 28, 'LINEA'), (32, 42, 'LUGAR')]}),
('Tenía que haber pasado a las 7:12 por hermanos villalonga y no pasa\n', {'entities': [(10, 22, 'INCIDECIAS')]}),
("Bon dia. Els viatges carregats al bonobus l'any passat, serveixen per al següent any?\n", {'entities': [(0, 7, 'SALUDOINIVAL'), (34, 41, 'BONO')]}),
('Gràcies\n', {'entities': [(0, 7, 'AGRADECIMIENTOVAL')]}),
('Bona vesprada, per favor, a partir de les 12 de la nit, quin autobús puc agafar per anar de marítim serreria a carrer de la Reina? (Cabanyal). El nocturn pot passar per allí? Gràcies\n', {'entities': [(0, 14, 'SALUDOINIVAL'), (42, 55, 'TEMP'), (84, 88, 'RUTAS'), (89, 91, 'ORIGEN'), (92, 108, 'LUGAR1'), (109, 110, 'DESTINO'), (111, 141, 'LUGAR2')]}),
('Buenos días. Parada 496 no funciona panel\n', {'entities': [(20, 23, 'PARADA'), (24, 35, 'INCIDENCIAS'), (36, 41, 'QUE')]}),
('Y la aplicación del móvil tampoco parece que funcione correctamente.\n', {'entities': [(0, 1, 'OTRAC'), (5, 15, 'QUE'), (26, 53, 'INCIDENCIAS')]}),
('Hace 4 minutos ponía la información al revés. El 98 10 minutos y el 95 7 minutos\n', {'entities': [(5, 14, 'TEMP')]}),
('El 98 ya  ha llegado\n', {'entities': [(3, 5, 'LINEA')]}),
('En esta parada falla mucho la pantalla. casi nunca se ve la información\n', {'entities': [(8, 14, 'DONDE'), (15, 20, 'INCIDENCIAS'), (30, 39, 'QUE')]}),
('Buenas tardes por favor que bus puedo coger desde el principio de Avda primado Reig para ir al auditori Palau de les arts reina Sofía,gracias un saludo\n', {'entities': [(44, 49, 'ORIGEN'), (50, 62, 'ATRIBUTO'), (66, 83, 'LUGAR1'), (84, 91, 'RUTAS'), (92, 94, 'DESTINO'), (95, 141, 'LUGAR2')]}),
('Buenas días.Ayer hize  una recarga de bonobús de 10 viajes y aún no me llegado la recarga.el número de tarjeta mobilis  es 271503002099. El número de pedido es 2080944.gracias\n', {'entities': [(26, 33, 'VENTAONLINE'), (37, 44, 'BONO'), (64, 77, 'INCIDENCIAS'), (121, 133, 'TARJETA')]}),
('Buenos días:Me llamo Boris González, DNI 44507598E.El pasado 24 de marzo hice una recarga online de mi tarjeta de transporte 514088273 de 30 viajes.Tras haberla utilizado una semana aparentemente 10 viajes no se llegaron a cargar. La semana pasada solo gasté 6 viajes.quería pedirles por favor revisen el saldo y si la recarga se hizo correctamente.Gracias\n', {'entities': [(82, 96, 'VENTAONLINE'), (125, 134, 'TARJETA')]}),
('Porfa vor para ir desde ayora hasta Túria como puedo hacer?\n', {'entities': [(0, 5, 'EDUCADO'), (10, 17, 'RUTAS'), (18, 23, 'ORIGEN'), (24, 29, 'LUGAR1'), (30, 35, 'DESTINO'), (36, 41, 'LUGAR2')]}),
('Y desde Blasco Ibáñez para ir a Túria?\n', {'entities': [(0, 1, 'OTRAC'), (2, 7, 'ORIGEN'), (8, 21, 'LUGAR1'), (22, 29, 'RUTAS'), (30, 31, 'DESTINO'), (32, 37, 'LUGAR2')]}),
('Hola, hay un bus directo para ir desde el metro de  ángel guimera o xativa a la playa las arenas o malvarrosa\n', {'entities': [(0, 4, 'SALUDOINI'), (25, 32, 'RUTAS'), (33, 38, 'ORIGEN'), (42, 64, 'LUGAR1'), (74, 75, 'DESTINO'), (79, 95, 'LUGAR2'), (96, 97, 'ALTERNATIVA'), (98, 108, 'LUGAR3')]}),
('Buenos dias para ir de patraix al palacio de justicia que buses tomar\n', {'entities': [(12, 19, 'RUTAS'), (20, 22, 'ORIGEN'), (23, 30, 'LUGAR1'), (31, 33, 'DESTINO'), (34, 53, 'LUGAR2')]}),
('Próximo 89 60 parada 1217\n', {'entities': [(0, 7, 'ESTIMACION'), (8, 10, 'LINEA'), (11, 13, 'LINEA'), (21, 25, 'PARADA')]}),
('Un bus de torres de serrano a Calipso tatto?\n', {'entities': [(7, 9, 'ORIGEN'), (10, 27, 'LUGAR1'), (28, 29, 'DESTINO'), (30, 43, 'LUGAR2')]}),
('Hola buenas! En la parada de bus 689 (ruzafa marques del turia) hace mucho que no funciona la pantalla de información\n', {'entities': [(33, 36, 'PARADA'), (37, 62, 'PARADA'), (79, 90, 'INCIDENCIAS'), (94, 117, 'QUE')]}),
('parada 1332 linea 28\n', {'entities': [(7, 11, 'PARADA'), (18, 20, 'LINEA')]}),
('Hola para ir a la punta multiespacio desde la plaza Honduras que autobuses hay que coger?\n', {'entities': [(5, 12, 'RUTAS'), (13, 14, 'DESTINO'), (18, 36, 'LUGAR2'), (37, 42, 'ORIGEN'), (46, 60, 'LUGAR1')]}),
('Y para ir desde José soto mico hasta la plaza Honduras?\n', {'entities': [(0, 1, 'OTRAC'), (2, 9, 'RUTAS'), (10, 15, 'ORIGEN'), (16, 30, 'LUGAR1'), (31, 36, 'DESTINO'), (40, 55, 'LUGAR2')]}),
('Hola para ir desde plaza España hasta la plaza América que autobuses me dejan cerca?\n', {'entities': [(5, 12, 'RUTAS'), (13, 18, 'ORIGEN'), (19, 31, 'LUGAR1'), (32, 37, 'DESTINO'), (41, 54, 'LUGAR2'), (78, 83, 'ATRIBUTO')]}),
('El 79 también serviría?\n', {'entities': []}),
('Vale, gracias! Y para volver desde ahí hasta José soto mico con las líneas nocturnas como sería?\n', {'entities': [(6, 14, 'AGRADECIMIENTO'), (15, 16, 'OTRAC'), (17, 28, 'RUTAS'), (29, 34, 'ORIGEN'), (39, 44, 'DESTINO'), (45, 59, 'LUGAR2')]}),
('Y desde la plaza América hasta el ayuntamiento no hay combinación?\n', {'entities': [(0, 1, 'OTRAC'), (2, 7, 'ORIGEN'), (11, 24, 'LUGAR1'), (25, 30, 'DESTINO'), (34, 46, 'LUGAR2')]}),
('Me he dejado olvidado un paraguas en el asiento se detrás del conductor en la linea 94. Ha sido hace diez minutos cuando pasó por navarro Reverter\n', 
{'entities': [(6, 12, 'OBJETOS'), (13, 21, 'OBJETOS'), (25, 33, 'QUE'), (84, 86, 'LINEA'), (101, 113, 'TEMP')]}),
('Buenos dias señores, x favor mi hija tiene el carnet de bonobús mensual pero ahora en adelante cogerá más el tranvía q tengo q hacer y donde\n', {'entities': [(21, 28, 'EDUCADO'), (56, 71, 'BONO'), (117, 132, 'INFORMACION')]}), 
('Buenos días, Me podrían informar  a que hora pasa en la mañana el primer bus de la ruta 92 los sábados en la parada de joan verdeguer pare porta\n', {'entities': [(33, 48, 'ESTIMACION'), (87, 89, 'LINEA'), (94, 101, 'ATRIBUTO'), (118, 132, 'PARADA')]}),
('Si\n', {'entities': [(0, 2, 'AFIRMACION')]}),
('Próximo 9 o 10 parada 742\n', {'entities': [(0, 7, 'ESTIMACION'), (8, 9, 'LINEA'), (10, 11, 'ALTERNATIVA'), (12, 14, 'LINEA'), (22, 25, 'PARADA')]}),
('👍\n', {'entities': [(0, 1, 'AFIRMACION')]}),
('258\n', {'entities': [(0, 3, 'PARADA')]}),
("Bona vesprada. Este mati he carregat la targeta i no s'ha fet efectiva. He tingut que pagar un bitllet\n", {'entities': [(0, 14, 'SALUDOINIVAL'), (28, 36, 'VENTAONLINE'), (50, 71, 'INCIDENCIAS')]}),
('416931808307\n', {'entities': [(0, 12, 'TARJETA')]}),
('Hola buenos días, hay alteración del servicio de la linea 15 de Pinedo a la pl. Del ajuntament ??\n', {'entities': [(22, 32, 'INCIDENCIAS'), (58, 60, 'LINEA'), (61, 63, 'ORIGEN'), (64, 70, 'LUGAR1'), (71, 72, 'DESTINO'), (76, 94, 'LUGAR2')]}),
('Línea 71 parada 511\n', {'entities': [(6, 8, 'LINEA'), (16, 19, 'PARADA')]}),
('Línea 71 parada 2007\n', {'entities': [(6, 8, 'LINEA'), (16, 20, 'PARADA')]}),
('Buenos días quisiera saber si ese viaje , digamos de cortesía, que te da la tarjeta cuando tienen saldo cero, se cobra cuando se recarga la tarjeta y se sube de nuevo al bus. Espero haberme explicado. Gracias\n', {'entities': [(12, 26, 'INFORMACION'), (53, 61, 'QUE'), (129, 136, 'VENTAONLINE')]}),
('Gracias a ti, feliz lunes!De plaza España o pintor segrelles a maestro Rodrigo 4. Gracias, Para mañana por la mañana\n', {'entities': [(26, 28, 'ORIGEN'), (29, 41, 'LUGAR1'), (42, 43, 'ALTERNATIVA'), (44, 60, 'LUGAR3'), (61, 62, 'DESTINO'), (63, 80, 'LUGAR2')]}),
('Hola, tras recargar mi tarjeta EMT ayer a las 18:00 via Online\n', {'entities': [(11, 19, 'VENTAONLINE'), (35, 39, 'TEMP'), (56, 62, 'VENTAONLINE')]}),
('A fecha de hoy 07:46 aun no hay disponibilidad de viajes registrados\n', {'entities': [(11, 14, 'TEMP'), (25, 46, 'RECLAMACION'), (50, 56, 'QUE')]}),
('Con el consiguiente pago necesario en mis viajes de autobus\n', {'entities': [(20, 24, 'QUEJA')]}),
('Cuando tendre disponibles mis viajes? Al realizar la recarga se hablaba de 30 minutos\n', {'entities': [(53, 60, 'VENTAONLINE')]}),
('293028897326\n', {'entities': [(0, 12, 'TARJETA')]}),
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
('Bns dias. Perdi mi amb tu y me hice otra pero resulta que me la han quitado y la que perdi la he encontrado puedo activar la primera o me tengo que hacer otra?\n', {'entities': [(0, 9, 'SALUDOINI'), (10, 15, 'OBJETOS'), (19, 25, 'BONO'), (85, 90, 'OBJETOS')]}),
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
('El 62\n', {'entities': [(3, 5, 'linea')]}),
('Que autobuses tienen parada en los juzgados del centro comercial el  saler\n', {'entities': [(21, 27, 'ITINERARIO'), (48, 73, 'LUGAR')]}),
('Y el 95 o 99 paran después de pasar el río de nuevo centro\n', {'entities': [(5, 7, 'LINEA'), (8, 9, 'ALTERNATIVA'), (10, 12, 'LINEA'), (13, 18, 'ITINERARIO'), (46, 58, 'LUGAR')]}),
('El 95 pasa por la plaza Ajuntamiento ?\n', {'entities': [(3, 5, 'LINEA'), (6, 14, 'ITINERARIO'), (18, 36, 'LUGAR')]}),
('O por San Agustín y nuevo centro ?\n', {'entities': [(0, 1, 'ALTERNATIVA'), (2, 5, 'ITINERARIO'), (6, 17, 'LUGAR')]}),
('Si pero para coger el 62 donde me viene más cerca ?\n', {'entities': [(22, 24, 'LINEA')]}),
('Y el 95 pasa por la plaza del ayuntamiento\n', {'entities': [(0, 1, 'OTRAC'), (5, 7, 'LINEA'), (8, 16, 'ITINERARIO'), (20, 42, 'LUGAR')]}),
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
('llevamos 40 minutos esperando\n', {'entities': [(9, 19, 'TEMP'), (20, 29, 'QUEJA')]}),
('Desde la calle Comte de Altea 44\n', {'entities': [(0, 5, 'ORIGEN'), (15, 32, 'LUGAR')]}),
('Sabe cómo puedo llegar a la Calle Roteros 16\n', {'entities': [(16, 22, 'RUTAS'), (23, 24, 'DESTINO'), (28, 44, 'LUGAR2')]}),
('Y desde la calle Comte de Altea 44\n', {'entities': [(0, 1, 'OTRAC'), (2, 7, 'ORIGEN'), (17, 34, 'LUGAR1')]}),
('Cómo puedo llegar a Ruzafa en bus? Por la calle Cádiz, Cuba, Literato Azorín..\n', {'entities': [(11, 17, 'RUTAS'), (18, 19, 'DESTINO'), (20, 26, 'LUGAR2')]}),
('Y desde Comte de Altea 44\n', {'entities': [(0, 1, 'OTRAC'), (2, 7, 'ORIGEN'), (8, 25, 'LUGAR1')]}),
('Hasta el Carrer de Conca, 31?\n', {'entities': [(0, 5, 'DESTINO'), (9, 28, 'LUGAR2')]}),
('Soy Milagros Llorca se me ha perdido una carterita marrón con el bono amb tu y otras cosas\n', {'entities': [(26, 36, 'OBJETOS'), (41, 50, 'QUE'), (51, 57, 'ATRIBUTO'), (70, 76, 'BONO'), ]}),
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
('Y que hacemos con las incidencias que pasan fuera del horario de atención al cliente ?\n', {'entities': [(0, 1, 'OTRAC'), (18, 21, 'las'), (22, 33, 'INCIDENCIAS'), (65, 73, 'CONTACTO')]}),
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
('Buenos días para ir a c/ Periodista Lorente 3, desde plaza Ayuntamiento, que línea puedo subir.\n', {'entities': [(12, 19, 'RUTAS'), (20, 21, 'DESTINO'), (22, 45, 'LUGAR2'), (47, 52, 'ORIGEN'), (53, 72, 'LUGAR1')]})
]


# In[4]:


print(len(TRAIN_DATA))


# In[5]:


TRAIN=TRAIN_DATA[:300]


# In[6]:


print(len(TRAIN))


# In[7]:


TEST=TRAIN_DATA[300:]


# In[2]:


import spacy
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score
import pandas as pd


# In[34]:


c=0
nlp = spacy.load("/Users/edugallardopardo/OneDrive/Documentos/TFM/python/SPACY/modelos/EMT_models/model4")
lista=[]
for text, annot in TEST:
    print('TEXTO -> '+str(text), 'Anotaciones -> '+str(annot))
    doc_to_test=nlp(text)
    print([(ent.text, ent.label_) for ent in doc_to_test.ents])

    d={}

    for ent in doc_to_test.ents:
        d[ent.label_]=[0,0,0,0,0,0]
        
    for ent in doc_to_test.ents:
        doc_gold_text= nlp.make_doc(text)
        gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
        print(annot.get("entities"))
        print(gold.ner)
        y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
        y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]  
        if(d[ent.label_][0]==0):
            #f.write("For Entity "+ent.label_+"\n")   
            #f.write(classification_report(y_true, y_pred)+"\n")
            (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
            a=accuracy_score(y_true,y_pred)
            d[ent.label_][0]=1
            d[ent.label_][1]+=p
            d[ent.label_][2]+=r
            d[ent.label_][3]+=f
            d[ent.label_][4]+=a
            d[ent.label_][5]+=1
    c+=1
    elemento=[]
    for i in d:
        """elemento.append(text)
        print("\n For Entity "+i+"\n")
        elemento.append(i)
        print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
        elemento.append((d[i][4]/d[i][5])*100)
        print("Precision : "+str(d[i][1]/d[i][5]))
        elemento.append(d[i][1]/d[i][5])
        print("Recall : "+str(d[i][2]/d[i][5]))
        elemento.append(d[i][2]/d[i][5])
        print("F-score : "+str(d[i][3]/d[i][5]))
        elemento.append(d[i][3]/d[i][5])"""
        lista.append([text, i, (d[i][4]/d[i][5])*100, (d[i][1]/d[i][5])*100, (d[i][2]/d[i][5])*100, (d[i][3]/d[i][5])*100])


# In[10]:


df = pd.DataFrame(lista, columns=['Consulta','Entidad','Accuracy', 'Precision', 'Recall', 'F-score'])


# In[17]:


df_plot=pd.DataFrame(df.groupby('Entidad').mean())
print(df.groupby('Entidad').mean())
print(df.groupby('Entidad').count())
print(df[df['Entidad'] == 'LUGAR1'])


# In[12]:


import matplotlib.pyplot as plt


# In[13]:


df_plot.plot(kind='bar')


# In[14]:


df_plot.plot()


# In[18]:


df_plot.plot(kind='barh')


# In[74]:


print(df[df['Entidad'] == 'LUGAR2'])


# In[49]:


from spacy.gold import offsets_from_biluo_tags
from spacy.gold import spans_from_biluo_tags
from spacy.gold import biluo_tags_from_offsets
nlp = spacy.load("/Users/edugallardopardo/OneDrive/Documentos/TFM/python/SPACY/modelos/EMT_models/model4")


# In[73]:


lista=[]

for text, annot in TEST:
    print('TEXTO -> '+str(text), 'ANOTACIONES MANUALES-> '+str(annot.get("entities")))
    entities_manual=annot.get("entities")
    doc=nlp(text)
    tags = biluo_tags_from_offsets(doc, entities_manual)
    print(tags)
    entities = offsets_from_biluo_tags(doc, tags)
    print('ENTIDADES MATCHEADAS ->', entities)
    #doc_gold_text= nlp.make_doc(text)
    #gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
    #tags = gold.ner
    
    #entities = spans_from_biluo_tags(doc_to_test, tags)
    #print('PREDICCION -> ',[(ent.text, ent.label_) for ent in doc.ents])
    prediccion=[(ent.text, ent.label_) for ent in doc.ents]
    print (prediccion)
    #print('PREDECIDO MODELO -> '+str(entities))

    for enti in entities_manual: # recorremos las anotaciones 
        VP=0
        FP=0
        FN=0
        FA=0
        if enti in entities: # si hay match entonces sumamos en verdadero positivo
            VP=VP+1
        else:
            for x in prediccion:
                if x[1]==enti[2]: # Fallo anotación
                    FA=FA+1
                    FP = FP+1
                    print(x[1])
            if FA==0: # Si está en las anotaciones pero no esta en las predicciones es un Falso Negativo
                FN=FN+1
        lista.append([text, enti[2], VP, FP, FN])
      
        
    print('***************************************************')
    print(lista)
    print('***************************************************')
    
    #lista.append([text, i, (d[i][4]/d[i][5])*100, (d[i][1]/d[i][5])*100, (d[i][2]/d[i][5])*100, (d[i][3]/d[i][5])*100])


# In[75]:


df_confusion = pd.DataFrame(lista, columns=['Consulta','Entidad','VP', 'FP', 'FN'])


# In[76]:


print(df_confusion)


# In[78]:


print(df_confusion[df_confusion['Entidad'] == 'LUGAR2'])


# In[ ]:




