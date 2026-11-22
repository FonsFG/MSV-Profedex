# MSV para Profedex

import comentarios_por_categoria as datos

import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import json


# 1. Cargar palabras vacías y normalizarlas
def normalizar_palabras_vacias(palabras_vacias):
  palabras_vacias = palabras_vacias.lower()
  palabras_vacias = unicodedata.normalize("NFD", palabras_vacias)
  palabras_vacias = "".join(c for c in palabras_vacias if unicodedata.category(c) != "Mn")
  palabras_vacias = re.sub(r"[^a-zA-Z0-9\s]", "", palabras_vacias)
  return palabras_vacias.split()

with open("stopwords-es.txt", "r", encoding="utf-8") as f:
  stopwords_es = f.read()

PALABRAS_VACIAS = set(normalizar_palabras_vacias(stopwords_es))

# 2. Normalización de comentarios
def normalizar_comentario(comentario):
  comentario = comentario.lower()
  comentario_descompuesto = unicodedata.normalize("NFD",comentario)
  # unicodedata.normalize descompone el texto con unicode NFD
  comentario_noacentos = "".join(
    c
    for c in comentario_descompuesto
    if unicodedata.category(c) != "Mn"
  )


  comentario_normalizado = re.sub(r"[^a-zA-Z0-9\s]", "", comentario_noacentos)

  pre_token = comentario_normalizado.split()

  token = [palabra for palabra in pre_token if palabra not in PALABRAS_VACIAS]

  return token


#2.1 Pequeña muestra de los comentarios normalizados
grupos = {
  "Dominio de la materia":datos.comentarios_dominio_de_la_materia,
  "Evaluación y actividades":datos.comentarios_evaluacion_actividades,
  "Explicación y claridad":datos.comentarios_explicacion_claridad,
  "Organización y cumplimiento":datos.comentarios_organizacion_cumplimiento,
  "Trato y atención al estudiante":datos.comentarios_trato_atencion_al_estudiante,
}
comentarios=[]
categorias=[]

for categoria, lista_comentarios in grupos.items():
  comentarios.extend(lista_comentarios)
  categorias.extend([categoria] * len(lista_comentarios))

resultados_normalizados = [normalizar_comentario(comentario) for comentario in comentarios]

X_train, X_test, y_train, y_test = train_test_split(comentarios, categorias, test_size=0.2, random_state=42,stratify=categorias)


# 3. Vectorización de comentarios
vectorizar_comentario = CountVectorizer(tokenizer=normalizar_comentario,lowercase=False,token_pattern=None,ngram_range=(1,2),min_df=2,binary=True)

X_train = vectorizar_comentario.fit_transform(X_train)
X_test = vectorizar_comentario.transform(X_test)


modelo = LinearSVC(C=0.1)
modelo.fit(X_train, y_train)  # Etiquetas de ejemplo para entrenamiento

predicciones_test = modelo.predict(X_test)
print(f"Exactitud en prueba: {accuracy_score(y_test, predicciones_test):.2%}\n")
print(classification_report(y_test, predicciones_test))

nuevo_comentario = "El profesor explica los conceptos de manera clara y responde a todas las preguntas de los estudiantes."
vector_nuevo = vectorizar_comentario.transform([nuevo_comentario])

categoria_predicha = modelo.predict(vector_nuevo)[0]
puntajes = modelo.decision_function(vector_nuevo)[0]

print(f"Comentario: '{nuevo_comentario}'")
print(f"Categorías asignadas: {categoria_predicha}")
for cate, pts in zip(modelo.classes_, puntajes):
    print(f"  - {cate}: {pts:+.4f}")

#3.1 Mostrar el vocabulario, la matriz de caracteristicas y los pesos de cada palabra en cada comentario
vocabulario = vectorizar_comentario.get_feature_names_out()
pesos_palabras = modelo.coef_.T

libro_pesos ={}
for palabra, peso in zip(vocabulario, pesos_palabras):
    lista_p = [round(float(p), 6) for p in peso]
    if any(p != 0 for p in lista_p):
        libro_pesos[palabra] = lista_p

datos_pesos = {
   "Categorias": modelo.classes_.tolist(),
   "Sesgos": modelo.intercept_.tolist(),
   "Pesos": libro_pesos
}
with open("pesos_palabras.json", "w", encoding="utf-8") as f:
    json.dump(datos_pesos, f, ensure_ascii=False, indent=2)

print("Orden de clases:", modelo.classes_)
print("Forma de coef_ (clases x palabras):", modelo.coef_.shape)
print("Sesgos por clase:", modelo.intercept_)
