import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import requests
import ta
import time as time_module
import json
import os
import uuid
import textwrap
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# CONFIGURACIÓN Y SEGURIDAD
# ==========================================
st.set_page_config(page_title="ZT TRADERS INSTITUTIONAL PRO", layout="centered", page_icon="⚡")

# NOTA: Es recomendable mover esto a variables de entorno o st.secrets
TWELVE_DATA_API_KEY = "03f557e7efa24c3889042680d2039aee"

ARCHIVO_HISTORIAL = "historial.json"
ARCHIVO_PATRONES = "patrones.json"
ARCHIVO_LICENCIAS = "licencias.json"

def cargar_licencias():
    if os.path.exists(ARCHIVO_LICENCIAS):
        try:
            with open(ARCHIVO_LICENCIAS, 'r') as f: return json.load(f)
        except: return {}
    return {}

def guardar_licencias(licencias):
    try:
        with open(ARCHIVO_LICENCIAS, 'w') as f: json.dump(licencias, f, indent=2)
    except: pass

# Inicialización de estado
if 'licencia_activa' not in st.session_state: st.session_state.licencia_activa = False
if 'usuario_actual' not in st.session_state: st.session_state.usuario_actual = ""

params = st.query_params
device_token = params.get("zt_token", None)

# Lógica de Login (Simplificada para mantener el flujo)
if device_token and not st.session_state.licencia_activa:
    licencias = cargar_licencias()
    if device_token == "admin_master_device":
        st.session_state.licencia_activa = True
        st.session_state.usuario_actual = "👑 Admin Master"
    else:
        for k, v in licencias.items():
            if v.get("device_id") == device_token:
                try:
                    if datetime.now().date() <= datetime.strptime(v['vence'], '%Y-%m-%d').date():
                        st.session_state.licencia_activa = True
                        st.session_state.usuario_actual = f"Licencia ({k[:8]}...)"
                except: pass

if not st.session_state.licencia_activa:
    st.markdown("""
    <style>
    body { background: #020208; color: #fff; }
    .login-card { max-width: 350px; margin: 50px auto; padding: 30px; background: rgba(10, 14, 28, 0.95); border-radius: 16px; border: 1.5px solid #00ff88; text-align: center; }
    .login-title { color: #00ff88; font-size: 24px; font-weight: 900; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="login-card"><div class="login-title">⚡ ZT TRADERS</div><div style="color:#64748b; font-size:10px;">Terminal Institucional Privada</div></div>', unsafe_allow_html=True)
    with st.form("licencia_form"):
        key_input = st.text_input("License Key", type="password")
        submit_key = st.form_submit_button("🚀 CONECTAR TERMINAL", use_container_width=True)
        if submit_key:
            if key_input == "ADMIN-MASTER-2027":
                st.session_state.licencia_activa = True
                st.session_state.usuario_actual = "👑 Admin Master"
                st.query_params["zt_token"] = "admin_master_device"
                st.rerun()
            else:
                licencias = cargar_licencias()
                if key_input in licencias:
                    datos = licencias[key_input]
                    if datetime.now().date() <= datetime.strptime(datos['vence'], '%Y-%m-%d').date():
                        if datos['device_id'] is None:
                            datos['device_id'] = str(uuid.uuid4())
                            guardar_licencias(licencias)
                            st.query_params["zt_token"] = datos['device_id']
                            st.session_state.licencia_activa = True
                            st.session_state.usuario_actual = f"Licencia ({key_input[:8]}...)"
                            st.rerun()
                        elif device_token == datos['device_id']:
                            st.session_state.licencia_activa = True
                            st.session_state.usuario_actual = f"Licencia ({key_input[:8]}...)"
                            st.rerun()
                    else: st.error("❌ Key expirada o inválida.")
                else: st.error("❌ Key no encontrada.")
    st.stop()

# ==========================================
# 📂 GESTIÓN DE DATOS
# ==========================================
def cargar_patrones():
    if os.path.exists(ARCHIVO_PATRONES):
        try:
            with open(ARCHIVO_PATRONES, 'r') as f: return json.load(f)
        except: return {'total_ops': 0, 'total_ganadas': 0, 'winrate': 0.0}
    return {'total_ops': 0, 'total_ganadas': 0, 'winrate': 0.0}

def guardar_patrones(pesos):
    with open(ARCHIVO_PATRONES, 'w') as f: json.dump(pesos, f, indent=2)

def cargar_historial():
    if os.path.exists(ARCHIVO_HISTORIAL):
        try:
            with open(ARCHIVO_HISTORIAL, 'r') as f: return json.load(f)
        except: return []
    return []

def guardar_historial(hist):
    with open(ARCHIVO_HISTORIAL, 'w') as f: json.dump(hist, f, indent=2)

def resetear_memoria():
    if os.path.exists(ARCHIVO_HISTORIAL): os.remove(ARCHIVO_HISTORIAL)
    if os.path.exists(ARCHIVO_PATRONES): os.remove(ARCHIVO_PATRONES)
    st.session_state.pesos = {'total_ops': 0, 'total_ganadas': 0, 'winrate': 0.0}

# ==========================================
# 🧠 MODELO DE MACHINE LEARNING (MEJORADO)
# ==========================================
def entrenar_modelo_ml():
    try:
        hist = cargar_historial()
        if len(hist) < 15: return None
        
        ganadas = sum(1 for op in hist if op['resultado'] == 'GANADA')
        winrate_actual = (ganadas / len(hist)) * 100 if len(hist) > 0 else 0
        
        if len(hist) >= 30 and winrate_actual < 35:
            resetear_memoria()
            return None
        
        X, y = [], []
        for op in hist:
            ind = op.get('indicadores', {})
            # Mejoras en Features: Usar cambios relativos (deltas)
            features = [
                ind.get('rsi', 50), 
                ind.get('rsi_change', 0), # NUEVO: Cambio del RSI
                ind.get('stoch', 50),
                ind.get('macd_diff', 0),
                ind.get('bb_pos', 0.5),
                ind.get('vol_conf', 0.5), # NUEVO: Confirmación de volumen
                1 if ind.get('es_martillo', False) else 0,
                1 if ind.get('es_estrella_fugaz', False) else 0,
                1 if ind.get('es_envolvente_alcista', False) else 0,
                1 if ind.get('es_envolvente_bajista', False) else 0,
                1 if op['señal'] == 'CALL' else 0
            ]
            X.append(features)
            y.append(1 if op['resultado'] == 'GANADA' else 0)
        
        modelo = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
        modelo.fit(X, y)
        return modelo
    except: return None

def predecir_con_ml(modelo, indicadores, direccion):
    try:
        features = [
            indicadores.get('rsi', 50), 
            indicadores.get('rsi_change', 0),
            indicadores.get('stoch', 50),
            indicadores.get('macd_diff', 0),
            indicadores.get('bb_pos', 0.5),
            indicadores.get('vol_conf', 0.5),
            1 if indicadores.get('es_martillo', False) else 0,
            1 if indicadores.get('es_estrella_fugaz', False) else 0,
            1 if indicadores.get('es_envolvente_alcista', False) else 0,
            1 if indicadores.get('es_envolvente_bajista', False) else 0,
            1 if direccion == 'CALL' else 0
        ]
        prob = modelo.predict_proba([features])[0]
        idx = 1 if direccion == 'CALL' else 0
        return int(prob[idx] * 100)
    except: return None

# ==========================================
# 📊 OBTENCIÓN DE DATOS Y ANÁLISIS
# ==========================================
def obtener_datos_binance(symbol, interval='1m', limit=60):
    try:
        res = requests.get("https://api.binance.com/api/v3/klines", params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=5).json()
        df = pd.DataFrame(res, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        for c in ['open', 'high', 'low', 'close', 'volume']: df[c] = pd.to_numeric(df[c])
        return df
    except: return None

def obtener_datos_twelvedata(symbol, limit=60):
    try:
        res = requests.get(f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize={limit}&apikey={TWELVE_DATA_API_KEY}", timeout=5).json()
        if 'values' in res:
            df = pd.DataFrame(res['values']).iloc[::-1].reset_index(drop
