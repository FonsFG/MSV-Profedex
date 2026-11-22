# MSV para Profedex

import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# 1. Limpieza de datos
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

  token = comentario_normalizado.split()

  return token

#1.2 Pequeña muestra de los comentarios normalizados
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

# 2. Vectorización de comentarios
vectorizar_comentario = CountVectorizer(tokenizer=normalizar_comentario,lowercase=False,token_pattern=None,binary=True)

X = vectorizar_comentario.fit_transform([ " ".join(token) for token in resultados_normalizados ])

vocabulario = vectorizar_comentario.get_feature_names_out()
print("\nVocabulario:")
print(vocabulario)

print("\nMatriz de características (X):")
print(X.toarray())