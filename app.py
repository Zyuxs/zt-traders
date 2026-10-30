import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import time
import json
import os
from datetime import datetime
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

st.set_page_config(page_title="ZT TRADERS", layout="centered", page_icon="🎯")

ARCHIVO_HISTORIAL = "historial.json"
ARCHIVO_PATRONES = "patrones.json"
ARCHIVO_USUARIOS = "usuarios.json"

def cargar_usuarios():
    if os.path.exists(ARCHIVO_USUARIOS):
        try:
            with open(ARCHIVO_USUARIOS, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def verificar_usuario(usuario, clave):
    usuarios = cargar_usuarios()
    if usuario not in usuarios:
        return False, "Usuario no encontrado"
    datos = usuarios[usuario]
    if datos['clave'] != clave:
        return False, "Contraseña incorrecta"
    fecha_vence = datetime.strptime(datos['vence'], '%Y-%m-%d').date()
    hoy = datetime.now().date()
    if hoy > fecha_vence:
        return False, f"Tu membresia vencio el {datos['vence']}. Contacta al administrador."
    return True, datos.get('nombre', usuario)

if 'logueado' not in st.session_state:
    st.session_state.logueado = False
if 'usuario_actual' not in st.session_state:
    st.session_state.usuario_actual = ""

if not st.session_state.logueado:
    st.markdown("""
    <style>
    body { background: #000; color: #fff; font-family: Arial, sans-serif; }
    .login-container { max-width: 400px; margin: 80px auto; padding: 30px; background: linear-gradient(90deg, #1a1a2e, #16213e); border-radius: 15px; border: 2px solid #ffd700; box-shadow: 0 0 40px rgba(255, 215, 0, 0.3); }
    .login-title { color: #ffd700; text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 10px; letter-spacing: 3px; }
    .login-subtitle { color: #aaa; text-align: center; font-size: 13px; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">ZT TRADERS</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Sistema de Acceso Privado</div>', unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        usuario_input = st.text_input("Usuario", placeholder="Ingresa tu usuario")
        clave_input = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña")
        submit = st.form_submit_button("ENTRAR", use_container_width=True)
        if submit:
            if usuario_input and clave_input:
                ok, mensaje = verificar_usuario(usuario_input, clave_input)
                if ok:
                    st.session_state.logueado = True
                    st.session_state.usuario_actual = mensaje
                    st.rerun()
                else:
                    st.error(mensaje)
            else:
                st.warning("Por favor completa todos los campos")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

def cargar_patrones():
    if os.path.exists(ARCHIVO_PATRONES):
        try:
            with open(ARCHIVO_PATRONES, 'r') as f:
                return json.load(f)
        except:
            os.remove(ARCHIVO_PATRONES)
    return {'total_ops': 0, 'total_ganadas': 0, 'winrate': 0.0, 'total_saltos': 0, 'modelo_entrenado': False}

def guardar_patrones(pesos):
    try:
        with open(ARCHIVO_PATRONES, 'w') as f:
            json.dump(pesos, f, indent=2)
    except:
        pass

def cargar_historial():
    if os.path.exists(ARCHIVO_HISTORIAL):
        try:
            with open(ARCHIVO_HISTORIAL, 'r') as f:
                return json.load(f)
        except:
            os.remove(ARCHIVO_HISTORIAL)
    return []

def guardar_historial(hist):
    try:
        with open(ARCHIVO_HISTORIAL, 'w') as f:
            json.dump(hist, f, indent=2)
    except:
        pass

def aprender_en_segundo_plano(señal, resultado, indicadores):
    hist = cargar_historial()
    pesos = cargar_patrones()
    registro = {
        'fecha': datetime.now().strftime('%H:%M:%S'),
        'señal': señal,
        'resultado': resultado,
        'indicadores': indicadores
    }
    hist.append(registro)
    pesos['total_ops'] += 1
    if resultado == 'GANADA':
        pesos['total_ganadas'] += 1
    pesos['winrate'] = (pesos['total_ganadas'] / pesos['total_ops'] * 100) if pesos['total_ops'] > 0 else 0.0
    guardar_historial(hist)
    guardar_patrones(pesos)

def entrenar_modelo_ml():
    """Entrena un modelo Random Forest con el historial de operaciones"""
    hist = cargar_historial()
    
    # Necesitamos al menos 20 operaciones para entrenar
    if len(hist) < 20:
        return None
    
    # Preparar datos para entrenamiento
    X = []  # Características (indicadores)
    y = []  # Etiquetas (ganó o perdió)
    
    for op in hist:
        if 'indicadores' in op:
            ind = op['indicadores']
            # Extraer características
            features = [
                ind.get('rsi', 50),
                ind.get('stoch', 50),
                ind.get('macd', 0),
                ind.get('adx', 20),
                ind.get('bb_position', 0.5),
                ind.get('sma_cross', 0),
                1 if op['señal'] == 'CALL' else 0  # Dirección de la señal
            ]
            X.append(features)
            y.append(1 if op['resultado'] == 'GANADA' else 0)
    
    # Necesitamos al menos 20 muestras con indicadores completos
    if len(X) < 20:
        return None
    
    # Entrenar modelo Random Forest
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        modelo.fit(X_train, y_train)
        
        # Calcular precisión del modelo con datos de prueba (NO de entrenamiento)
        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'modelo': modelo,
            'accuracy': accuracy,
            'muestras': len(X)
        }
    except:
        return None

def predecir_con_ml(modelo, indicadores):
    """Usa el modelo ML para predecir probabilidad de éxito"""
    features = [
        indicadores.get('rsi', 50),
        indicadores.get('stoch', 50),
        indicadores.get('macd', 0),
        indicadores.get('adx', 20),
        indicadores.get('bb_position', 0.5),
        indicadores.get('sma_cross', 0),
        1 if indicadores.get('direccion', 'CALL') == 'CALL' else 0
    ]
    
    # Predecir probabilidad
    probabilidad = modelo.predict_proba([features])[0]
    prob_exito = probabilidad[1]  # Probabilidad de ganar (1)
    
    return prob_exito * 100  # Convertir a porcentaje

def señal_por_defecto(codigo_par, rsi=50.0):
    """Señal de respaldo infalible cuando falla el análisis"""
    dirección = 'CALL' if rsi < 50 else 'PUT'
    indicadores = {
        'rsi': rsi,
        'stoch': 50,
        'macd': 0,
        'adx': 20,
        'bb_position': 0.5,
        'sma_cross': 0,
        'direccion': dirección
    }
    return {
        'señal': dirección,
        'precio': 0.0,
        'rsi': rsi,
        'confianza': 10,
        'razon': f"Señal por defecto (RSI {rsi:.0f})",
        'indicadores': indicadores
    }

def analizar_par(codigo_par):
    """Analiza el mercado y usa ML para predecir - NUNCA devuelve None"""
    try:
        datos_1m = yf.download(codigo_par, period="2d", interval="1m", progress=False)
        datos_5m = yf.download(codigo_par, period="5d", interval="5m", progress=False)
        if isinstance(datos_1m.columns, pd.MultiIndex):
            datos_1m.columns = datos_1m.columns.get_level_values(0)
            datos_5m.columns = datos_5m.columns.get_level_values(0)
        datos_1m = datos_1m[['Open', 'High', 'Low', 'Close']].copy()
        datos_5m = datos_5m[['Open', 'High', 'Low', 'Close']].copy()
        
        if len(datos_1m) < 30 or len(datos_5m) < 30:
            return señal_por_defecto(codigo_par)

        # Calcular indicadores
        datos_1m['RSI'] = ta.momentum.RSIIndicator(datos_1m['Close'], window=14).rsi()
        datos_1m['STOCH'] = ta.momentum.StochasticOscillator(datos_1m['High'], datos_1m['Low'], datos_1m['Close']).stoch()
        datos_1m['MACD'] = ta.trend.MACD(datos_1m['Close']).macd()
        datos_1m['MACD_signal'] = ta.trend.MACD(datos_1m['Close']).macd_signal()
        datos_1m['ADX'] = ta.trend.ADXIndicator(datos_1m['High'], datos_1m['Low'], datos_1m['Close']).adx()
        bb = ta.volatility.BollingerBands(datos_1m['Close'], window=20, window_dev=2)
        datos_1m['BB_High'] = bb.bollinger_hband()
        datos_1m['BB_Low'] = bb.bollinger_lband()
        datos_1m['SMA_10'] = datos_1m['Close'].rolling(window=10).mean()
        datos_1m['SMA_50'] = datos_1m['Close'].rolling(window=50).mean()
        datos_5m['SMA_20'] = datos_5m['Close'].rolling(window=20).mean()
        datos_5m['RSI'] = ta.momentum.RSIIndicator(datos_5m['Close'], window=14).rsi()

        u1 = datos_1m.iloc[-1]
        a1 = datos_1m.iloc[-2]
        u5 = datos_5m.iloc[-1]
        rsi = 50.0 if pd.isna(u1['RSI']) else float(u1['RSI'])
        stoch = 50.0 if pd.isna(u1['STOCH']) else float(u1['STOCH'])
        macd = 0.0 if pd.isna(u1['MACD']) else float(u1['MACD'])
        macd_sig = 0.0 if pd.isna(u1['MACD_signal']) else float(u1['MACD_signal'])
        adx = 20.0 if pd.isna(u1['ADX']) else float(u1['ADX'])
        precio = float(u1['Close'])
        bb_high = float(u1['BB_High'])
        bb_low = float(u1['BB_Low'])
        sma_1m = float(u1['SMA_10'])
        sma_50 = float(u1['SMA_50'])
        sma_5m = float(u5['SMA_20'])
        precio_5m = float(u5['Close'])
        rsi_5m = 50.0 if pd.isna(u5['RSI']) else float(u5['RSI'])

        # Determinar dirección base
        dirección = 'CALL' if rsi < 50 else 'PUT'
        if precio <= bb_low:
            dirección = 'CALL'
        elif precio >= bb_high:
            dirección = 'PUT'
        if precio > sma_1m and a1['Close'] <= a1['SMA_10']:
            dirección = 'CALL'
        elif precio < sma_1m and a1['Close'] >= a1['SMA_10']:
            dirección = 'PUT'

        # Calcular posición en Bandas de Bollinger (0 = banda inferior, 1 = banda superior)
        bb_position = (precio - bb_low) / (bb_high - bb_low) if bb_high != bb_low else 0.5
        bb_position = max(0, min(1, bb_position))
        
        # Cruce de SMA (1 = cruzó arriba, -1 = cruzó abajo, 0 = sin cruce)
        sma_cross = 0
        if precio > sma_1m and a1['Close'] <= a1['SMA_10']:
            sma_cross = 1
        elif precio < sma_1m and a1['Close'] >= a1['SMA_10']:
            sma_cross = -1

        # Preparar indicadores para ML
        indicadores = {
            'rsi': rsi,
            'stoch': stoch,
            'macd': macd - macd_sig,  # Diferencia MACD
            'adx': adx,
            'bb_position': bb_position,
            'sma_cross': sma_cross,
            'direccion': dirección
        }

        # Intentar usar ML si hay modelo entrenado
        modelo_info = entrenar_modelo_ml()
        
        if modelo_info and modelo_info['modelo']:
            # Usar predicción ML
            probabilidad_ml = predecir_con_ml(modelo_info['modelo'], indicadores)
            
            # Solo dar señal si ML predice > 50% de probabilidad (más flexible para aprender)
            if probabilidad_ml >= 50:
                return {
                    'señal': dirección,
                    'precio': precio,
                    'rsi': rsi,
                    'confianza': probabilidad_ml,
                    'razon': f"ML: {probabilidad_ml:.1f}% (Precisión: {modelo_info['accuracy']*100:.0f}%, Muestras: {modelo_info['muestras']})",
                    'indicadores': indicadores
                }
        
        # Fallback: usar sistema de puntos si ML no está listo o predice bajo
        confianza = 0
        razones = []

        if dirección == 'CALL' and rsi <
