
# 📥 Carga de Archivos en Python

Este documento proporciona un resumen práctico y completo de cómo **cargar distintos tipos de archivos de datos** en proyectos de análisis y ciencia de datos usando Python.

---

## 📦 Librerías necesarias

```python
import pandas as pd
import json
import pyreadstat
import sqlite3
import zipfile
```

---

## 1️⃣ Archivos CSV

```python
# CSV estándar
df = pd.read_csv("data/archivo.csv")

# Con separador personalizado (por ejemplo, punto y coma)
df = pd.read_csv("data/archivo.csv", sep=';')

# Con codificación
df = pd.read_csv("data/archivo.csv", encoding='latin1')
```

---

## 2️⃣ Archivos Excel

```python
# Leer hoja por nombre o índice
df = pd.read_excel("data/archivo.xlsx", sheet_name='Hoja1')  # o sheet_name=0
```

---

## 3️⃣ Archivos JSON

```python
# JSON como diccionario
with open("data/archivo.json", "r") as f:
    data = json.load(f)

# Convertir a DataFrame
df = pd.json_normalize(data)
```

---

## 4️⃣ Archivos de texto (TXT, TSV, delimitados)

```python
# Tabulado
df = pd.read_csv("data/archivo.txt", sep="\t")

# Otros delimitadores
df = pd.read_csv("data/archivo.txt", delimiter="|")
```

---

## 5️⃣ Archivos comprimidos (ZIP)

```python
with zipfile.ZipFile("data/archivo.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")
df = pd.read_csv("data/archivo_extraido.csv")
```

---

## 6️⃣ Archivos SPSS (.sav)

```python
df, meta = pyreadstat.read_sav("data/archivo.sav")
```

---

## 7️⃣ Archivos Stata (.dta)

```python
df = pd.read_stata("data/archivo.dta")
```

---

## 8️⃣ Archivos Parquet

```python
df = pd.read_parquet("data/archivo.parquet")
```

---

## 9️⃣ Bases de datos SQLite

```python
conn = sqlite3.connect("data/basedatos.sqlite")
df = pd.read_sql_query("SELECT * FROM tabla", conn)
```

---

## 🔟 Leer múltiples archivos en una carpeta

```python
import os

dfs = []
for file in os.listdir("data/"):
    if file.endswith(".csv"):
        temp_df = pd.read_csv(os.path.join("data", file))
        dfs.append(temp_df)

df_total = pd.concat(dfs, ignore_index=True)
```

---

## ✅ Buenas prácticas

- Crea carpetas `/data/raw/` y `/data/processed/` para organización
- Usa `try-except` para manejar errores de lectura
- Documenta la fuente y características del archivo
- No subas archivos >100 MB a GitHub; usa enlaces o `.gitignore`

