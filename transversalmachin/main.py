from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="CSGO Round Predictor")

# Cargar modelos y scalers
modelo_clasificacion = joblib.load("models/modelo_clasificacion.pkl")
modelo_regresion = joblib.load("models/modelo_regresion.pkl")
scaler_clf = joblib.load("models/scaler_clasificacion.pkl")
scaler_reg = joblib.load("models/scaler_regresion.pkl")

# Esquemas de entrada
class ClasificacionInput(BaseModel):
    RoundHeadshots: int
    TeamStartingEquipmentValue: int
    PrimaryAssaultRifle: int
    TimeAlive: float

class RegresionInput(BaseModel):
    RoundHeadshots: int
    TeamStartingEquipmentValue: int
    PrimaryAssaultRifle: int
    TimeAlive: float

@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Archivo index.html no encontrado")
    except UnicodeDecodeError:
        raise HTTPException(status_code=500, detail="Error de codificación del archivo HTML")

@app.post("/predict/clasificacion")
def predict_round_winner(data: ClasificacionInput):
    try:
        # Preparar datos
        input_data = pd.DataFrame([[data.RoundHeadshots, data.TeamStartingEquipmentValue, 
                                  data.PrimaryAssaultRifle, data.TimeAlive]],
                                columns=['RoundHeadshots', 'TeamStartingEquipmentValue', 
                                        'PrimaryAssaultRifle', 'TimeAlive'])
        
        # Escalar
        input_scaled = scaler_clf.transform(input_data)
        
        # Predecir
        prediction = modelo_clasificacion.predict(input_scaled)
        
        return {"RoundWinner_predicho": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/regresion")
def predict_round_kills(data: RegresionInput):
    try:
        # Preparar datos
        input_data = pd.DataFrame([[data.RoundHeadshots, data.TeamStartingEquipmentValue, 
                                  data.PrimaryAssaultRifle, data.TimeAlive]],
                                columns=['RoundHeadshots', 'TeamStartingEquipmentValue', 
                                        'PrimaryAssaultRifle', 'TimeAlive'])
        
        # Escalar
        input_scaled = scaler_reg.transform(input_data)
        
        # Predecir
        prediction = modelo_regresion.predict(input_scaled)
        
        return {"RoundKills_predicho": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)