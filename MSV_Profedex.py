# MSV para Profedex

import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# 1. Cargar palabras vacías y normalizarlas
def normalizar_palabras_vacias(palabras_vacias):
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


#2. Pequeña muestra de los comentarios normalizados
comentarios = [
  
  "El profe es excelente, muy claro en sus explicaciones.",
  "No me gustó la forma en que enseña, es confuso.",
  "¡Excelente profesor! Muy didáctico y claro.",
  "Siemplemente the goat, el mejor profe que he tenido.",
  "No recomiendo a este profesor, no explica bien.",
]

resultados_normalizados = [normalizar_comentario(comentario) for comentario in comentarios]

print("Comentarios normalizados:")
for i, resultado in enumerate(resultados_normalizados):
  print(f"Comentario {i+1}: {resultado}")

# 3. Vectorización de comentarios
vectorizar_comentario = CountVectorizer(tokenizer=normalizar_comentario,lowercase=False,token_pattern=None,ngram_range=(1,2),binary=True)

X = vectorizar_comentario.fit_transform([ " ".join(token) for token in resultados_normalizados ])

vocabulario = vectorizar_comentario.get_feature_names_out()
print("\nVocabulario:")
print(vocabulario)

print("\nMatriz de características (X):")
print(X.toarray())