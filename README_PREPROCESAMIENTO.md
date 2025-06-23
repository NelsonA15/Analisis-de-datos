
# 🔄 Preprocesamiento de Datos en Python

Este documento recopila un **catálogo extenso y detallado** de técnicas y herramientas de preprocesamiento en Python. El preprocesamiento es un paso esencial en cualquier proyecto de ciencia de datos: mejora la calidad de los datos y su preparación para modelos predictivos o descriptivos.

---

## 📦 Librerías necesarias

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
```

---

## 1️⃣ Limpieza básica

```python
# Eliminar duplicados
df = df.drop_duplicates()

# Renombrar columnas
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Convertir fechas
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

# Reemplazar texto
df['columna'] = df['columna'].replace({'N/A': np.nan, '-': np.nan})
```

---

## 2️⃣ Manejo de valores faltantes

```python
# Conteo de faltantes
df.isnull().sum()

# Imputación con la media, mediana o moda
imputer = SimpleImputer(strategy='mean')  # median, most_frequent
df[['col1']] = imputer.fit_transform(df[['col1']])

# Imputación por KNN
knn_imp = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(knn_imp.fit_transform(df), columns=df.columns)
```

---

## 3️⃣ Codificación de variables categóricas

```python
# Label Encoding
le = LabelEncoder()
df['col_label'] = le.fit_transform(df['col_categorica'])

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['col_categorica'])

# Usando sklearn
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['col_categorica']])
```

---

## 4️⃣ Escalamiento de variables

```python
# Estandarización (media=0, var=1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['var1', 'var2']])

# Normalización a [0, 1]
minmax = MinMaxScaler()
df_norm = minmax.fit_transform(df[['var1', 'var2']])

# Escalamiento robusto (menos sensible a outliers)
robust = RobustScaler()
df_robust = robust.fit_transform(df[['var1', 'var2']])
```

---

## 5️⃣ Transformaciones de variables

```python
# Log-transformación
df['log_var'] = np.log1p(df['var'])

# Raíz cuadrada
df['sqrt_var'] = np.sqrt(df['var'])

# Potencias o Box-Cox
from scipy.stats import boxcox
df['boxcox_var'], _ = boxcox(df['var'].clip(lower=1))
```

---

## 6️⃣ Detección y tratamiento de outliers

```python
# Usando el rango intercuartílico (IQR)
Q1 = df['var'].quantile(0.25)
Q3 = df['var'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['var'] < Q1 - 1.5 * IQR) | (df['var'] > Q3 + 1.5 * IQR)]

# Capping (winsorización)
df['var'] = np.where(df['var'] > Q3 + 1.5 * IQR, Q3 + 1.5 * IQR, df['var'])
```

---

## 7️⃣ Reducción de dimensionalidad

```python
# PCA: Análisis de Componentes Principales
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[['var1', 'var2', 'var3']])

# Filtrado por varianza
selector = VarianceThreshold(threshold=0.01)
df_reduced = selector.fit_transform(df)
```

---

## 8️⃣ Creación de nuevas variables (feature engineering)

```python
# Interacciones
df['interaccion'] = df['var1'] * df['var2']

# Agregaciones por grupo
df['promedio_por_grupo'] = df.groupby('categoria')['var'].transform('mean')

# Variables temporales
df['mes'] = df['fecha'].dt.month
df['anio'] = df['fecha'].dt.year
```

---

## 9️⃣ Balanceo de clases (para clasificación)

```python
# Oversampling con SMOTE
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

# Undersampling
from imblearn.under_sampling import RandomUnderSampler
X_under, y_under = RandomUnderSampler().fit_resample(X, y)
```

---

## 🔍 Consejos finales

- Documenta cada paso de preprocesamiento.
- Mantén una copia de los datos originales (`df_raw`).
- Usa pipelines (`sklearn.pipeline`) si vas a aplicar modelos.
- Aplica transformaciones solo en entrenamiento, no en test.

