
# 📊 Herramientas de Visualización de Datos en Python

Este documento resume un catálogo de herramientas y funciones útiles para visualizar datos en proyectos de ciencia de datos. Todas están basadas en `matplotlib`, `seaborn`, `plotly`, `missingno`, y otras librerías populares.

---

## 📦 Librerías necesarias

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
```

---

## 1️⃣ Histogramas

```python
sns.histplot(data=df, x='variable', bins=30, kde=True)
plt.title('Histograma de variable')
plt.show()
```

---

## 2️⃣ Boxplots (para detectar outliers)

```python
sns.boxplot(data=df, x='variable')
plt.title('Boxplot de variable')
plt.show()
```

---

## 3️⃣ Diagramas de dispersión (scatterplots)

```python
sns.scatterplot(data=df, x='var1', y='var2', hue='categoria')
plt.title('Scatterplot entre var1 y var2')
plt.show()
```

---

## 4️⃣ Matriz de correlación

```python
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlación')
plt.show()
```

---

## 5️⃣ Pairplot (distribuciones cruzadas)

```python
sns.pairplot(df[['var1', 'var2', 'var3']], hue='categoria')
plt.suptitle('Pairplot entre variables seleccionadas', y=1.02)
plt.show()
```

---

## 6️⃣ Barras para categorías

```python
sns.countplot(data=df, x='variable_categorica', order=df['variable_categorica'].value_counts().index)
plt.title('Frecuencias de categorías')
plt.xticks(rotation=45)
plt.show()
```

---

## 7️⃣ Gráfico de líneas

```python
sns.lineplot(data=df, x='tiempo', y='valor', hue='grupo')
plt.title('Evolución temporal')
plt.show()
```

---

## 8️⃣ Gráfico de violín

```python
sns.violinplot(data=df, x='grupo', y='valor')
plt.title('Distribución por grupo (violinplot)')
plt.show()
```

---

## 9️⃣ Mapa de valores nulos

```python
msno.matrix(df)
plt.title('Matriz de valores nulos')
plt.show()

msno.heatmap(df)
plt.title('Mapa de correlación de nulos')
plt.show()
```

---

## 🔟 Gráficos interactivos con Plotly

### 🔹 Dispersión 3D

```python
fig = px.scatter_3d(df, x='var1', y='var2', z='var3', color='categoria')
fig.show()
```

### 🔹 Box interactivo

```python
fig = px.box(df, x='grupo', y='valor', color='grupo')
fig.show()
```

### 🔹 Serie de tiempo

```python
fig = px.line(df, x='fecha', y='valor', color='categoria')
fig.show()
```

---

## 🔄 Mapa de calor de tablas cruzadas

```python
tabla = pd.crosstab(df['cat1'], df['cat2'])
sns.heatmap(tabla, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Crosstab heatmap entre cat1 y cat2')
plt.show()
```

---

## 📈 Catálogo de visualizaciones disponibles

| Tipo               | Librería     | Función principal            |
|--------------------|--------------|------------------------------|
| Histograma         | seaborn      | `histplot()`                 |
| Boxplot            | seaborn      | `boxplot()`                  |
| Dispersión         | seaborn      | `scatterplot()`              |
| Correlación        | seaborn      | `heatmap()`                  |
| Distribución múltiple | seaborn  | `pairplot()`                 |
| Barras             | seaborn      | `countplot()`                |
| Líneas             | seaborn      | `lineplot()`                 |
| Violinplot         | seaborn      | `violinplot()`               |
| Nulos              | missingno    | `matrix()` / `heatmap()`     |
| 3D, interactivo    | plotly       | `scatter_3d()`               |
| Series tiempo      | plotly       | `line()`                     |
| Crosstab heatmap   | seaborn/pandas | `crosstab()` + `heatmap()` |

---

Puedes combinar estas visualizaciones en notebooks, dashboards o reportes automáticos.
