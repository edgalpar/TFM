Hola!

El presente modelo, clasifica y extrae entidades (NER) de consultas de tipo rutas, cómo llegar a..., deseo ir desde....., 
que bus puedo coger para ir a... que hacen los ciudadanos de Valencia para transporte en superficie.

Para cargar modelo simplemente usar en python, con R se puede cargar trozo de código en snippet o algo así ;-), 
si no lo tenéis hay que instalar previamente la libreria SPACY 

IMPORTANTE:el modelo también contiene los vectores de cada palabra (WORD2VEC), con lo cual podemos calcular la 
similaridad de frases para determinar que quiere decir la persona


pip install -U spacy

import spacy
nlp=spacy.load("<RUTA DONDE DEJEIS CARPETA emtrutas_1>")

# Ejemplo de uso NER

text="Buenos días. Deseo ir desde la Patacona al Hospital Arnau de Vilanova\n"
doc=nlp(text)
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

# Resultado esperado: 
# "DESEO IR" -> RUTAS
# "DESDE" -> ORIGEN
# "PATACONA" -> LUGAR1
# "AL" -> DESTINO
# "HOSPITAL ARNAU DE VILANOVA" -> LUGAR2

# Ejemplo de uso WORD2VEC

target = nlp("Me gustaría ir a la Calle Valencia")
 
doc1=nlp(u"Buenos días, como podria hacer para ir a la Quiron en blasco ibañez, desde calle olta?")

doc2=nlp(u"Buenos días, a que horas pasa el 99 parada 1752 Fausto Elio (impar) Gracias, tengo que llegar a la primera parada del trayecto")

doc3=nlp(u"Buenos días, Tengo que ir a la avenida Constitucion desde Primado Reig n12")

doc4=nlp(u"Don Quijote y Sancho Panza son leyendas vivas de nuestra literatura")
 
print(target.similarity(doc1))  # 0.9513887453926154
print(target.similarity(doc2))  # 0.9506958853141597
print(target.similarity(doc3))  # 0.9599712176197848
print(target.similarity(doc4))  # 0.8876825440911528
