# MSV para Profedex

import comentarios_por_categoria as datos

import re
import unicodedata
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

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


#2.1 Definición de grupos de comentarios por categoría
grupos = {
  "Dominio de la materia":datos.comentarios_dominio_de_la_materia,
  "Evaluación y actividades":datos.comentarios_evaluacion_actividades,
  "Explicación y claridad":datos.comentarios_explicacion_claridad,
  "Organización y cumplimiento":datos.comentarios_organizacion_cumplimiento,
  "Trato y atención al estudiante":datos.comentarios_trato_atencion_al_estudiante,
}

#2.2 Construcción del dataset de comentarios y categorías
def construir_dataset(grupos):
    """Devuelve (textos, categorias) sin duplicados ni comentarios ambiguos.

    La comparación se hace sobre el texto ya normalizado (sin acentos, sin
    puntuación y sin stopwords), que es lo que realmente ve el modelo.
    """
    categoria_por_clave = {}   # clave normalizada -> categoría de su primera aparición
    conflictos = set()         # claves que aparecen en más de una categoría
    registros = []             # (texto original, categoría, clave)
    duplicados = 0
    vacios = 0

    for categoria, lista_comentarios in grupos.items():
        for comentario in lista_comentarios:
            clave = " ".join(normalizar_comentario(comentario))
            if not clave:
                vacios += 1
                continue
            if clave in categoria_por_clave:
                if categoria_por_clave[clave] != categoria:
                    conflictos.add(clave)
                else:
                    duplicados += 1
                continue
            categoria_por_clave[clave] = categoria
            registros.append((comentario, categoria, clave))

    textos = [t for t, _, k in registros if k not in conflictos]
    categorias = [c for _, c, k in registros if k not in conflictos]

    print(f"Comentarios sin duplicados: {len(textos)}")
    print(f"  Duplicados eliminados: {duplicados}")
    print(f"  Vacíos tras normalizar (eliminados): {vacios}")
    print(f"  En más de una categoría (eliminados): {len(conflictos)}")
    for clave in sorted(conflictos):
        print(f"    - {clave}")
    print("Comentarios por categoría:")
    for cat, n in sorted(Counter(categorias).items()):
        print(f"  - {cat}: {n}")
    print()
    return textos, categorias


comentarios, categorias = construir_dataset(grupos)

# 2.3 División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    comentarios, categorias, test_size=0.2, random_state=42, stratify=categorias
)

# 3. Vectorización de comentarios y entrenamiento del modelo SVM con validación cruzada
ensamble = Pipeline([
   ("vec",CountVectorizer(tokenizer=normalizar_comentario,lowercase=False,token_pattern=None,ngram_range=(1,2),min_df=2,binary=True)),
   ("svm",LinearSVC(class_weight="balanced",random_state=42)),
])

parametros = {
  "vec__ngram_range": [(1, 1), (1, 2)],
  "vec__min_df": [1, 2, 3],
  "svm__C": [0.01, 0.1, 1, 10],
}

n_folds = min(5,min(Counter(y_train).values()))
validacion = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)


busqueda = GridSearchCV(ensamble, parametros, cv=validacion, scoring="f1_macro")
busqueda.fit(X_train, y_train)

print(f"Mejores parámetros: {busqueda.best_params_}")
print(f"Validación cruzada: {n_folds} folds, métrica f1_macro")
print(f"F1 macro (validación cruzada): {busqueda.best_score_:.4f}\n")

#3.1 Evaluación del modelo en el conjunto de prueba
mejor_modelo = busqueda.best_estimator_
predicciones_test=mejor_modelo.predict(X_test)

print(f"Exactitud en prueba: {accuracy_score(y_test, predicciones_test):.2%}\n")
print(classification_report(y_test, predicciones_test, zero_division=0))

clases=list(mejor_modelo.named_steps["svm"].classes_)
matriz=confusion_matrix(y_test, predicciones_test, labels=clases)
print("Matriz de confusión (filas=real, columnas=predicho):")
for i, clase in enumerate(clases):
    print(f"  [{i}] {clase}")
print("     " + "".join(f"{i:>5}" for i in range(len(clases))))
for i, fila in enumerate(matriz):
    print(f"[{i}]  " + "".join(f"{valor:>5}" for valor in fila))
print()

#3.2 Predicción de categoría para un nuevo comentario
nuevo_comentario = "El profesor explica los conceptos de manera clara y responde a todas las preguntas de los estudiantes."

categoria_predicha = mejor_modelo.predict([nuevo_comentario])[0]
puntajes = mejor_modelo.decision_function([nuevo_comentario])[0]

print(f"Comentario: '{nuevo_comentario}'")
print(f"Categorías asignadas: {categoria_predicha}")
for cate, pts in zip(clases, puntajes):
    print(f"  - {cate}: {pts:+.4f}")
if puntajes.max() < 0:
   print("  Ningún puntaje es positivo: el comentario probablemente no encaja en ninguna categoria.")
print()

#4. Guardar pesos de palabras y sesgos por clase en un archivo JSON
vectorizador = mejor_modelo.named_steps["vec"]
svm=mejor_modelo.named_steps["svm"]

vocabulario = vectorizador.get_feature_names_out()
pesos_palabras = svm.coef_.T

libro_pesos ={}
for palabra, peso in zip(vocabulario, pesos_palabras):
    lista_p = [round(float(p), 6) for p in peso]
    if any(p != 0 for p in lista_p):
        libro_pesos[palabra] = lista_p

datos_pesos = {
   "Categorias": svm.classes_.tolist(),
   "Sesgos": svm.intercept_.tolist(),
   "Rango_ngramas":list(vectorizador.ngram_range),
   "Palabras_vacias": sorted(PALABRAS_VACIAS),
   "Pesos": libro_pesos
}
with open("pesos_palabras.json", "w", encoding="utf-8") as f:
    json.dump(datos_pesos, f, ensure_ascii=False, indent=2)

print("Orden de clases:", svm.classes_)
print("Forma de coef_ (clases x palabras):", svm.coef_.shape)
print("Sesgos por clase:", svm.intercept_)
