
# 🧪 Evaluación de Calidad de los Datos en Python

Este documento describe herramientas y técnicas para evaluar la **calidad de los datos** en proyectos de análisis, limpieza y ciencia de datos. Conocer la calidad de tu dataset es esencial antes de aplicar modelos o tomar decisiones.

---

## 📦 Librerías necesarias

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
```

---

## 1️⃣ Valores nulos

```python
# Conteo total y por columna
print(df.isnull().sum())

# Porcentaje de valores nulos por columna
print((df.isnull().mean() * 100).round(2))

# Visualización de valores nulos
msno.matrix(df)
plt.title("Matriz de valores nulos")
plt.show()

msno.heatmap(df)
plt.title("Correlación de valores nulos")
plt.show()
```

---

## 2️⃣ Valores duplicados

```python
# Filas completamente duplicadas
duplicados = df.duplicated().sum()
print(f"Número de filas duplicadas: {duplicados}")

# Eliminar duplicados
df = df.drop_duplicates()
```

---

## 3️⃣ Cardinalidad (valores únicos)

```python
# Valores únicos por columna
print(df.nunique())

# Alta cardinalidad en columnas categóricas
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} valores únicos")
```

---

## 4️⃣ Consistencia de tipos de datos

```python
# Verifica tipos
print(df.dtypes)

# Intentar conversión automática
df['columna'] = pd.to_numeric(df['columna'], errors='coerce')
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
```

---

## 5️⃣ Valores extremos y outliers

```python
# Detección con rango intercuartílico (IQR)
Q1 = df['var'].quantile(0.25)
Q3 = df['var'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['var'] < Q1 - 1.5 * IQR) | (df['var'] > Q3 + 1.5 * IQR)]
print(f"Número de outliers: {outliers.shape[0]}")

# Visualizar outliers
sns.boxplot(x=df['var'])
plt.title("Boxplot de outliers")
plt.show()
```

---

## 6️⃣ Balance de clases (clasificación)

```python
# Para problemas de clasificación
print(df['target'].value_counts(normalize=True))

sns.countplot(x='target', data=df)
plt.title("Distribución de clases")
plt.show()
```

---

## 7️⃣ Completitud cruzada

```python
# Porcentaje de columnas completas por fila
df['porcentaje_completo'] = df.notnull().mean(axis=1) * 100

# Histograma de completitud
df['porcentaje_completo'].hist(bins=20)
plt.title("Distribución de completitud por fila")
plt.xlabel("% de columnas completas")
plt.show()
```

---

## 8️⃣ Correlación entre variables faltantes

```python
msno.heatmap(df)
plt.title("Mapa de correlación de valores faltantes")
plt.show()
```

---

## ✅ Recomendaciones

- Siempre analiza los datos antes de imputar o modelar.
- Documenta el % de calidad (faltantes, duplicados, valores únicos).
- Automatiza estas evaluaciones con funciones reutilizables.
