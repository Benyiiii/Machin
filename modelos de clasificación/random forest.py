# 📦 Setup: Cargar datos y preparar variables
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib

# Cargar CSV
df = pd.read_csv("AnexoET_RoundWinner_Limpio.csv")

# Variables seleccionadas
features = ['RoundHeadshots', 'TeamStartingEquipmentValue', 'PrimaryAssaultRifle', 'TimeAlive']
X = df[features]
y = df['RoundWinner']

# Separar y escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 🔍 Modelo: Random Forest Classifier + ROC AUC
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
model_rf.fit(X_train, y_train)

# Predicción y evaluación
y_pred = model_rf.predict(X_test)
y_proba = model_rf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("Precisión:", round(accuracy * 100, 2), "%")
print("ROC AUC:", round(roc_auc, 4))

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel("Falsos Positivos")
plt.ylabel("Verdaderos Positivos")
plt.title("Curva ROC - Random Forest")
plt.legend()
plt.grid()
plt.show()
