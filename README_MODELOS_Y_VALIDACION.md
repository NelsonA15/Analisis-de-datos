
# 🤖 Modelado Predictivo en Python

Este documento incluye una **guía extensa y práctica** para implementar modelos en Python con `scikit-learn`, cubriendo:

- Modelos de regresión, clasificación y ensambles
- Configuración base de modelos
- Búsqueda de hiperparámetros (Grid Search y Random Search)
- Validación cruzada
- Herramientas adicionales de evaluación

---

## 📦 Librerías necesarias

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, mean_squared_error, r2_score
```

---

## 1️⃣ Separar datos

```python
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 2️⃣ Modelos de clasificación (configuración base)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

modelos_clasificacion = {
    "LogReg": LogisticRegression(),
    "Árbol": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}
```

---

## 3️⃣ Modelos de regresión (configuración base)

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

modelos_regresion = {
    "Lineal": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Árbol": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "SVR": SVR(),
    "XGBoost": XGBRegressor(objective='reg:squarederror')
}
```

---

## 4️⃣ Búsqueda de hiperparámetros (Grid Search)

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)

print("Mejores parámetros:", grid.best_params_)
print("F1 score:", grid.best_score_)
```

---

## 5️⃣ Búsqueda aleatoria (Randomized Search)

```python
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10)
}

random_search = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=10, cv=5, random_state=42, scoring='roc_auc')
random_search.fit(X_train, y_train)

print("Mejores parámetros:", random_search.best_params_)
```

---

## 6️⃣ Validación cruzada

```python
from sklearn.model_selection import cross_val_score

model = GradientBoostingClassifier()
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Accuracy promedio:", scores.mean())
```

```python
# Validación cruzada extendida con múltiples métricas
res = cross_validate(model, X, y, cv=5,
                     scoring=['accuracy', 'f1', 'roc_auc'],
                     return_train_score=True)

print("F1 promedio:", res['test_f1'].mean())
```

---

## 7️⃣ Métricas de evaluación

### Clasificación

```python
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Matriz de confusión:
", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
```

### Regresión

```python
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))
```

---

## 8️⃣ Pipelines y preprocesamiento

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('modelo', RandomForestClassifier())
])

pipe.fit(X_train, y_train)
```

---

## 9️⃣ Comparación de modelos

```python
for nombre, modelo in modelos_clasificacion.items():
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    print(f"{nombre} - Accuracy: {accuracy_score(y_test, pred):.3f}")
```

---

## 🔍 Recomendaciones

- Usa `GridSearchCV` para problemas pequeños o medianos; `RandomizedSearchCV` para grandes.
- Evalúa con múltiples métricas, no solo accuracy.
- Usa `Pipelines` para combinar transformación y modelado.
- Guarda los mejores modelos con `joblib` o `pickle`.

