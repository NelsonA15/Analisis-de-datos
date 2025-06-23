
# 📊 Análisis Exploratorio de Datos (EDA) en Python

Este documento muestra un flujo de trabajo completo y profesional para realizar un Análisis Exploratorio de Datos (EDA) sobre cualquier DataFrame en Python.

---

## 📦 Librerías necesarias

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import scipy.stats as stats
import itertools

sns.set(style="whitegrid")
```

---

## 1️⃣ Información general del dataset

```python
def info_general(df):
    print("📌 Dimensiones:", df.shape)
    print("\n📌 Tipos de datos:")
    print(df.dtypes)
    print("\n📌 Primeras filas:")
    print(df.head())
    print("\n📌 Últimas filas:")
    print(df.tail())
    print("\n📌 Valores nulos por columna:")
    print(df.isnull().sum())
    print("\n📌 Porcentaje de valores nulos:")
    print((df.isnull().mean() * 100).round(2))
    print("\n📌 Filas duplicadas:", df.duplicated().sum())
```

---

## 2️⃣ Clasificación de variables

```python
def clasificar_variables(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    date_cols = df.select_dtypes(include='datetime').columns.tolist()

    print("🔢 Numéricas:", num_cols)
    print("🔠 Categóricas:", cat_cols)
    print("🔘 Booleanas:", bool_cols)
    print("📅 Fecha:", date_cols)

    return num_cols, cat_cols, bool_cols, date_cols
```

---

## 3️⃣ Estadísticas descriptivas

```python
def resumen_estadistico(df, num_cols, cat_cols):
    print("\n📊 Estadísticas numéricas:")
    print(df[num_cols].describe().T)

    if cat_cols:
        print("\n📋 Estadísticas categóricas:")
        print(df[cat_cols].describe().T)

    print("\n🔍 Valores únicos:")
    print(df.nunique())
```

---

## 4️⃣ Distribuciones y normalidad

```python
def distribucion_y_normalidad(df, num_cols):
    for col in num_cols:
        serie = df[col].dropna()
        plt.figure(figsize=(6, 4))
        sns.histplot(serie, kde=True, bins=30)
        plt.title(f'Distribución de {col}')
        plt.show()

        print(f"\n📌 {col}:")
        print(" Media:", serie.mean())
        print(" Desviación estándar:", serie.std())
        print(" Asimetría:", serie.skew())
        print(" Curtosis:", serie.kurtosis())
        if len(serie) >= 3:
            stat, p = stats.shapiro(serie.sample(min(5000, len(serie))))
            print(" Test de normalidad (Shapiro-Wilk) p-valor:", p)
```

---

## 5️⃣ Visualización de valores nulos

```python
def graficar_nulos(df):
    msno.matrix(df)
    plt.title("🔎 Matriz de valores nulos")
    plt.show()

    msno.heatmap(df)
    plt.title("🔥 Mapa de correlación de nulos")
    plt.show()
```

---

## 6️⃣ Boxplots para outliers

```python
def boxplots_outliers(df, num_cols):
    for col in num_cols:
        plt.figure(figsize=(6, 2.5))
        sns.boxplot(x=df[col])
        plt.title(f'Outliers en {col}')
        plt.show()
```

---

## 7️⃣ Matriz de correlación

```python
def matriz_correlacion(df, num_cols):
    corr = df[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("📈 Matriz de correlación")
    plt.show()
```

---

## 8️⃣ Pairplot

```python
def graficar_pairplot(df, num_cols):
    if len(num_cols) >= 2:
        sns.pairplot(df[num_cols[:4]].dropna())
        plt.suptitle("📷 Pairplot (Top 4 variables numéricas)", y=1.02)
        plt.show()
```

---

## 9️⃣ Frecuencias de variables categóricas

```python
def frecuencias_categoricas(df, cat_cols):
    for col in cat_cols:
        print(f"\n📌 {col} (Top 10 frecuencias):")
        print(df[col].value_counts().head(10))
        plt.figure(figsize=(6, 3))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index[:10])
        plt.xticks(rotation=45)
        plt.title(f'Frecuencias de {col}')
        plt.show()
```

---

## 🔟 Tablas cruzadas entre variables categóricas

```python
def crosstab_categoricas(df, cat_cols):
    for var1, var2 in itertools.combinations(cat_cols, 2):
        if df[var1].nunique() <= 10 and df[var2].nunique() <= 10:
            tabla = pd.crosstab(df[var1], df[var2])
            plt.figure(figsize=(8, 4))
            sns.heatmap(tabla, annot=True, fmt="d", cmap="YlGnBu")
            plt.title(f'Crosstab entre {var1} y {var2}')
            plt.tight_layout()
            plt.show()
```

---

## ✅ Ejemplo de uso

```python
df = pd.read_csv("data/archivo.csv")

info_general(df)
num_cols, cat_cols, _, _ = clasificar_variables(df)
resumen_estadistico(df, num_cols, cat_cols)
graficar_nulos(df)
distribucion_y_normalidad(df, num_cols)
boxplots_outliers(df, num_cols)
matriz_correlacion(df, num_cols)
graficar_pairplot(df, num_cols)
frecuencias_categoricas(df, cat_cols)
crosstab_categoricas(df, cat_cols)
```
