# 📦 Setup: Cargar datos y preparar variables
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Cargar CSV
df = pd.read_csv("AnexoET_RoundWinner_Limpio.csv")

# Variables seleccionadas
features = ['RoundHeadshots', 'TeamStartingEquipmentValue', 'PrimaryAssaultRifle', 'TimeAlive']
target = 'RoundKills'  # Asegurarse de que esta variable exista en el CSV

X = df[features]
y = df[target]

# Separar y escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 🔍 Modelo: Regresión Lineal
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Predicción y evaluación
y_pred = model_lr.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R^2:", round(r2, 4))
print("MSE:", round(mse, 4))

# Gráfica de resultados
plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Regresión Lineal - Predicciones vs Reales")
plt.grid()
plt.show()
