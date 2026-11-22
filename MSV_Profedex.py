# MSV para Profedex

import re
import unicodedata


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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

  comentario_normalizado = re.sub(r"")

  token = 

  return token

# 2. Modelo
MSVProfedexIA = {"Kernel RBF": SVC(kernel="rbf", C=1, gamma="scale")}

fig = plt.figure(figsize=(16, 5))

# Malla 2D para fronteras de decisión
xx, yy = np.meshgrid(
    np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200)
)
puntos_malla = np.c_[xx.ravel(), yy.ravel()]

# 3. Paneles 1 y 2: Comparativa 2D (Lineal vs RBF)
for i, (nombre, modelo) in enumerate(modelos.items(), 1):
  modelo.fit(X_train, y_train)
  acc = modelo.score(X_test, y_test)

  Z = modelo.predict(puntos_malla).reshape(xx.shape)

  ax = fig.add_subplot(1, 3, i)
  ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
  ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)

  # Graficar datos
  ax.scatter(
      X_train[:, 0],
      X_train[:, 1],
      c=y_train,
      cmap="coolwarm",
      edgecolors="k",
      s=30,
  )

  # Resaltar Vectores de Soporte en dorado
  sv = modelo.support_vectors_
  ax.scatter(
      sv[:, 0],
      sv[:, 1],
      s=80,
      facecolors="none",
      edgecolors="gold",
      linewidths=1.5,
      label="Vectores de Soporte",
  )

  ax.set_title(f"{nombre}\nExactitud: {acc:.1%}")
  ax.set_xlabel("X1")
  ax.set_ylabel("X2")
  ax.legend(loc="lower right")
  ax.axis("equal")

# 4. Panel 3: Elevación 3D (Explicación conceptual del Kernel Trick)
z = X[:, 0] ** 2 + X[:, 1] ** 2
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.scatter(X[:, 0], X[:, 1], z, c=y, cmap="coolwarm", s=30, edgecolors="k")
ax3.set_title("Explicación Geométrica 3D\n(Transformación Z = X1² + X2²)")
ax3.set_xlabel("X1")
ax3.set_ylabel("X2")
ax3.set_zlabel("Dimensión Z")

plt.tight_layout()
plt.show()