import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import ta
import time as time_module
import json
import os
import uuid
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ZT TRADERS", layout="centered", page_icon="🎯")

ARCHIVO_HISTORIAL = "historial.json"
ARCHIVO_PATRONES = "patrones.json"
ARCHIVO_LICENCIAS = "licencias.json"

def cargar_licencias():
    if os.path.exists(ARCHIVO_LICENCIAS):
        try:
            with open(ARCHIVO_LICENCIAS, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def guardar_licencias(licencias):
    try:
        with open(ARCHIVO_LICENCIAS, 'w') as f:
            json.dump(licencias, f, indent=2)
    except:
        pass

# --- SISTEMA DE LICENCIA Y SEGURIDAD POR DISPOSITIVO ---
if 'licencia_activa' not in st.session_state:
    st.session_state.licencia_activa = False
if 'usuario_actual' not in st.session_state:
    st.session_state.usuario_actual = ""

params = st.query_params
device_token = params.get("zt_token", None)

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
                except:
                    pass

if not st.session_state.licencia_activa:
    st.markdown("""
    <style>
    body { background: #000; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 0; }
    .login-container { max-width: 320px; margin: 50px auto; padding: 20px; background: linear-gradient(135deg, #0a0a1a, #1a0a2e); border-radius: 12px; border: 2px solid #ffd700; box-shadow: 0 0 30px rgba(255, 215, 0, 0.4); text-align: center; }
    .login-title { color: #ffd700; font-size: 22px; font-weight: bold; margin-bottom: 5px; letter-spacing: 2px; }
    .login-subtitle { color: #aaa; font-size: 11px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">⚡ ZT TRADERS</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Sistema de Licencia Protegido</div>', unsafe_allow_html=True)
    
    with st.form("licencia_form"):
        key_input = st.text_input("License Key", placeholder="Ingresa tu Key", type="password")
        submit_key = st.form_submit_button("⚡ ACTIVAR", use_container_width=True)
        
        if submit_key:
            if key_input:
                if key_input == "ADMIN-MASTER-2026":
                    st.session_state.licencia_activa = True
                    st.session_state.usuario_actual = "👑 Admin Master"
                    st.query_params["zt_token"] = "admin_master_device"
                    st.success("¡Acceso Admin concedido!")
                    st.rerun()
                else:
                    licencias = cargar_licencias()
                    if key_input in licencias:
                        datos = licencias[key_input]
                        try:
                            fecha_vence = datetime.strptime(datos['vence'], '%Y-%m-%d').date()
                            hoy = datetime.now().date()
                            
                            if hoy > fecha_vence:
                                st.error(f"❌ La Key venció el {datos['vence']}.")
                            else:
                                if datos['device_id'] is None:
                                    nuevo_token = str(uuid.uuid4())
                                    datos['device_id'] = nuevo_token
                                    guardar_licencias(licencias)
                                    st.query_params["zt_token"] = nuevo_token
                                    st.session_state.licencia_activa = True
                                    st.session_state.usuario_actual = f"Licencia ({key_input[:8]}...)"
                                    st.success("¡Licencia vinculada a este equipo con éxito!")
                                    st.rerun()
                                else:
                                    if device_token == datos['device_id']:
                                        st.session_state.licencia_activa = True
                                        st.session_state.usuario_actual = f"Licencia ({key_input[:8]}...)"
                                        st.rerun()
                                    else:
                                        st.error("❌ Esta Key ya está activada en otro dispositivo.")
                        except:
                            st.error("Error al validar fecha.")
                    else:
                        st.error("❌ Key inválida o inexistente.")
            else:
                st.warning("Ingresa una clave.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# --- FUNCIONALIDADES DE TRADING Y ML ---
def cargar_patrones():
    if os.path.exists(ARCHIVO_PATRONES):
        try:
            with open(ARCHIVO_PATRONES, 'r') as f:
                return json.load(f)
        except:
            os.remove(ARCHIVO_PATRONES)
    return {'total_ops': 0, 'total_ganadas': 0, 'winrate': 0.0}

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

def resetear_memoria():
    if os.path.exists(ARCHIVO_HISTORIAL):
        os.remove(ARCHIVO_HISTORIAL)
    if os.path.exists(ARCHIVO_PATRONES):
        os.remove(ARCHIVO_PATRONES)
    st.session_state.pesos = {'total_ops': 0, 'total_ganadas': 0, 'winrate': 0.0}

def entrenar_modelo_ml():
    try:
        hist = cargar_historial()
        if len(hist) < 5:
            return None
        X, y = [], []
        for op in hist:
            ind = op.get('indicadores', {})
            features = [
                ind.get('rsi', 50),
                ind.get('precio_relativo', 0.5),
                1 if op['señal'] == 'CALL' else 0
            ]
            X.append(features)
            y.append(1 if op['resultado'] == 'GANADA' else 0)
        modelo = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
        modelo.fit(X, y)
        return modelo
    except:
        return None

def aprender_en_segundo_plano(señal, resultado, indicadores, precio_real_entrada):
    hist = cargar_historial()
    pesos = cargar_patrones()
    registro = {
        'fecha': datetime.now().strftime('%H:%M:%S'),
        'señal': señal,
        'resultado': resultado,
        'precio_entrada_real': precio_real_entrada,
        'indicadores': indicadores
    }
    hist.append(registro)
    pesos['total_ops'] += 1
    if resultado == 'GANADA':
        pesos['total_ganadas'] += 1
    pesos['winrate'] = (pesos['total_ganadas'] / pesos['total_ops'] * 100) if pesos['total_ops'] > 0 else 0.0
    guardar_historial(hist)
    guardar_patrones(pesos)

def obtener_datos_binance(symbol, interval='1m', limit=30):
    try:
        url = f"https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        response = requests.get(url, params=params, timeout=4)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        return df
    except:
        return None

def obtener_datos_yahoo(symbol, period="1d", interval="1m"):
    try:
        datos = yf.download(symbol, period=period, interval=interval, progress=False, timeout=4)
        if datos is None or len(datos) == 0:
            return None
        if isinstance(datos.columns, pd.MultiIndex):
            datos.columns = datos.columns.get_level_values(0)
        datos = datos[['Open', 'High', 'Low', 'Close']].dropna().copy()
        return datos
    except:
        return None

def obtener_precio_actual(codigo_par, tipo='crypto'):
    if tipo == 'crypto':
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={codigo_par}"
            response = requests.get(url, timeout=3)
            data = response.json()
            return float(data['price'])
        except:
            return None
    else:
        try:
            datos = yf.download(codigo_par, period="1d", interval="1m", progress=False, timeout=3)
            if datos is None or len(datos) < 1: return None
            if isinstance(datos.columns, pd.MultiIndex):
                datos.columns = datos.columns.get_level_values(0)
            return float(datos['Close'].iloc[-1])
        except:
            return None

def analizar_par(codigo_par, tipo='crypto'):
    try:
        if tipo == 'crypto':
            df = obtener_datos_binance(codigo_par, interval='1m', limit=30)
        else:
            df = obtener_datos_yahoo(codigo_par, period="1d", interval="1m")
        
        if df is None or len(df) < 10:
            return {'señal': 'CALL', 'confianza': 60, 'razon': "Mercado cerrado / Sin datos", 'indicadores': {'rsi': 50, 'precio_relativo': 0.5}}
        
        c_close = 'close' if tipo == 'crypto' else 'Close'
        c_open = 'open' if tipo == 'crypto' else 'Open'
        
        df['RSI_7'] = ta.momentum.RSIIndicator(df[c_close], window=7).rsi()
        df['EMA_9'] = ta.trend.EMAIndicator(df[c_close], window=9).ema_indicator()
        df['EMA_21'] = ta.trend.EMAIndicator(df[c_close], window=21).ema_indicator()
        
        u1 = df.iloc[-1]
        precio = float(u1[c_close])
        apertura = float(u1[c_open])
        rsi_7 = float(u1['RSI_7'])
        ema_9 = float(u1['EMA_9'])
        ema_21 = float(u1['EMA_21'])
        
        vela_verde = precio >= apertura
        razones = []
        
        if (rsi_7 < 45 or ema_9 > ema_21) and vela_verde:
            dirección = 'CALL'
            razones.append("Momentum 1M Alcista")
        elif (rsi_7 > 55 or ema_9 < ema_21) and not vela_verde:
            dirección = 'PUT'
            razones.append("Momentum 1M Bajista")
        else:
            dirección = 'CALL' if ema_9 >= ema_21 else 'PUT'
            razones.append("Tendencia EMA 9/21")

        confianza_base = 75 if (rsi_7 < 35 or rsi_7 > 65) else 65
        modelo_ml = entrenar_modelo_ml()
        indicadores = {'rsi': rsi_7, 'precio_relativo': 1.0 if vela_verde else 0.0}
        
        if modelo_ml is not None:
            try:
                prediccion = modelo_ml.predict([[indicadores['rsi'], indicadores['precio_relativo'], 1 if dirección == 'CALL' else 0]])
                probabilidad = modelo_ml.predict_proba([[indicadores['rsi'], indicadores['precio_relativo'], 1 if dirección == 'CALL' else 0]])
                confianza_ml = int(probabilidad[0][prediccion[0]] * 100)
                confianza = int((confianza_base + confianza_ml) / 2)
                razones.append(f"ML ({confianza}%)")
            except:
                confianza = confianza_base
        else:
            confianza = confianza_base

        return {'señal': dirección, 'confianza': max(50, min(confianza, 98)), 'razon': " | ".join(razones), 'indicadores': indicadores}
    except Exception as e:
        return {'señal': 'CALL', 'confianza': 60, 'razon': "Modo seguro activo", 'indicadores': {'rsi': 50, 'precio_relativo': 0.5}}

def reproducir_voz(mensaje):
    mensaje_seguro = mensaje.replace("'", "\\'").replace('"', '\\"')
    js_code = f"""
    <script>
    (function() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance('{mensaje_seguro}');
            msg.lang = 'es-ES';
            msg.rate = 1.0;
            msg.volume = 1.0;
            window.speechSynthesis.speak(msg);
        }}
    }})();
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

# 23 PARES ORIGINALES COMPLETOS
PARES = {
    'Bitcoin': {'codigo': 'BTCUSDT', 'tipo': 'crypto'},
    'Ethereum': {'codigo': 'ETHUSDT', 'tipo': 'crypto'},
    'Solana': {'codigo': 'SOLUSDT', 'tipo': 'crypto'},
    'Polkadot': {'codigo': 'DOTUSDT', 'tipo': 'crypto'},
    'Worldcoin': {'codigo': 'WLDUSDT', 'tipo': 'crypto'},
    'Binance Coin': {'codigo': 'BNBUSDT', 'tipo': 'crypto'},
    'Cardano': {'codigo': 'ADAUSDT', 'tipo': 'crypto'},
    'Ripple': {'codigo': 'XRPUSDT', 'tipo': 'crypto'},
    'Dogecoin': {'codigo': 'DOGEUSDT', 'tipo': 'crypto'},
    'Polygon': {'codigo': 'MATICUSDT', 'tipo': 'crypto'},
    'Litecoin': {'codigo': 'LTCUSDT', 'tipo': 'crypto'},
    'EUR/USD': {'codigo': 'EURUSD=X', 'tipo': 'forex'},
    'GBP/USD': {'codigo': 'GBPUSD=X', 'tipo': 'forex'},
    'USD/JPY': {'codigo': 'USDJPY=X', 'tipo': 'forex'},
    'AUD/USD': {'codigo': 'AUDUSD=X', 'tipo': 'forex'},
    'EUR/JPY': {'codigo': 'EURJPY=X', 'tipo': 'forex'},
    'AUD/CAD': {'codigo': 'AUDCAD=X', 'tipo': 'forex'},
    'EUR/GBP': {'codigo': 'EURGBP=X', 'tipo': 'forex'},
    'GBP/JPY': {'codigo': 'GBPJPY=X', 'tipo': 'forex'},
    'NZD/USD': {'codigo': 'NZDUSD=X', 'tipo': 'forex'},
    'USD/CAD': {'codigo': 'USDCAD=X', 'tipo': 'forex'},
    'USD/CHF': {'codigo': 'USDCHF=X', 'tipo': 'forex'},
    'AUD/JPY': {'codigo': 'AUDJPY=X', 'tipo': 'forex'}
}

if 'estado' not in st.session_state: st.session_state.estado = 'analizando'
if 'par_sel' not in st.session_state: st.session_state.par_sel = 'Bitcoin'
if 'senal' not in st.session_state: st.session_state.senal = None
if 'pesos' not in st.session_state: st.session_state.pesos = cargar_patrones()
if 'resultado_guardado' not in st.session_state: st.session_state.resultado_guardado = False
if 'minuto_anterior' not in st.session_state: st.session_state.minuto_anterior = None
if 'precio_real_entrada' not in st.session_state: st.session_state.precio_real_entrada = None
if 'voz_reproducida' not in st.session_state: st.session_state.voz_reproducida = False

CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container { padding-top: 5px !important; padding-bottom: 0px !important; padding-left: 8px !important; padding-right: 8px !important; max-width: 100% !important; }

body { background: #000; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 0; }
.main-container { width: 100%; max-width: 450px; margin: 0 auto; display: flex; flex-direction: column; gap: 4px; }

.header { background: linear-gradient(90deg, #1a0a2e, #0a1a2e); padding: 6px; border-radius: 8px; border: 1.5px solid #ffd700; text-align: center; }
.header h1 { color: #ffd700; margin: 0; font-size: 16px; letter-spacing: 2px; }
.header p { color: #aaa; margin: 2px 0 0 0; font-size: 9px; }

.stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; }
.stat-box { background: linear-gradient(135deg, #0a1a2e, #1a0a2e); border: 1px solid #ffd700; border-radius: 6px; padding: 4px; text-align: center; }
.stat-box strong { color: #ffd700; font-size: 8px; display: block; margin-bottom: 1px; }
.stat-box span { color: #00ff88; font-size: 12px; font-weight: bold; }

.hora-box { background: linear-gradient(90deg, #0a2a1a, #1a1a2e); border: 2px solid #00ff88; border-radius: 6px; padding: 4px; text-align: center; }
.hora-box span { color: #00ff88; font-size: 18px; font-weight: bold; text-shadow: 0 0 8px #00ff88; }

.signal-box { padding: 8px; border-radius: 8px; text-align: center; font-size: 22px; font-weight: bold; letter-spacing: 2px; margin: 2px 0; }
.signal-call { background: linear-gradient(135deg, #00cc00, #00ff00); border: 2px solid #00ff00; color: white; box-shadow: 0 0 15px rgba(0, 255, 0, 0.6); }
.signal-put { background: linear-gradient(135deg, #cc0000, #ff0000); border: 2px solid #ff0000; color: white; box-shadow: 0 0 15px rgba(255, 0, 0, 0.6); }
.signal-wait { background: linear-gradient(135deg, #1a1a2e, #0a1a2e); border: 2px solid #ffd700; color: #ffd700; font-size: 14px; padding: 6px; }

.radar-container { position: relative; width: 140px; height: 140px; margin: 6px auto; }
.radar { width: 140px; height: 140px; border-radius: 50%; position: relative; overflow: hidden; border: 3px solid; }
.radar-green { border-color: #00ff00; background: radial-gradient(circle, #0a2a1a, #000); box-shadow: 0 0 15px rgba(0, 255, 0, 0.5); }
.radar-red { border-color: #ff0000; background: radial-gradient(circle, #2a0a0a, #000); box-shadow: 0 0 15px rgba(255, 0, 0, 0.5); }
.radar-yellow { border-color: #ffd700; background: radial-gradient(circle, #2a2a1a, #000); box-shadow: 0 0 15px rgba(255, 215, 0, 0.5); }
.radar-center { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; z-index: 10; }
.radar-number { font-size: 42px; font-weight: bold; line-height: 1; text-shadow: 0 0 10px currentColor; }
.radar-number-green { color: #00ff00; }
.radar-number-red { color: #ff0000; }
.radar-number-yellow { color: #ffd700; font-size: 15px; }
.radar-label { color: #fff; font-size: 8px; margin-top: 2px; letter-spacing: 1px; }

.info-panel { background: linear-gradient(135deg, #0a1a2e, #1a0a2e); border: 1px solid #00ff88; border-radius: 6px; padding: 5px 8px; font-size: 10px; text-align: center; }
.info-panel strong { color: #00ff88; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# CONTENEDOR ESTÁTICO SUPERIOR
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown(f'<div class="header"><h1>⚡ ZT TRADERS</h1><p>User: {st.session_state.usuario_actual} | 23 PARES ACTIVOS</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    par_nombre = st.selectbox("", list(PARES.keys()), index=list(PARES.keys()).index(st.session_state.par_sel), label_visibility="collapsed")
    st.session_state.par_sel = par_nombre
with col2:
    if st.button("🗑️ Reset", key="reset_btn", use_container_width=True):
        resetear_memoria()
        st.session_state.senal = None
        st.session_state.estado = 'analizando'
        st.rerun()

codigo_info = PARES[par_nombre]
codigo = codigo_info['codigo']
tipo = codigo_info['tipo']

hist = cargar_historial()
ops = st.session_state.pesos['total_ops']
ganadas = st.session_state.pesos['total_ganadas']
winrate = round(st.session_state.pesos['winrate'], 1)

st.markdown(f"""
<div class="stats-row">
    <div class="stat-box"><strong>Winrate</strong><span>{winrate}%</span></div>
    <div class="stat-box"><strong>Ops</strong><span>{ops}</span></div>
    <div class="stat-box"><strong>Win</strong><span>{ganadas}</span></div>
    <div class="stat-box"><strong>ML</strong><span style="font-size:10px;">{len(hist)} ops</span></div>
</div>
""", unsafe_allow_html=True)

# CONTENEDOR DINÁMICO (Limpia los elementos duplicados antes de refrescar)
pantalla_dinamica = st.empty()

ahora = datetime.now()
hora_actual = ahora.strftime('%H:%M:%S')

minuto_actual = ahora.minute
segundo_actual = ahora.second
minuto_es_par = (minuto_actual % 2 == 0)

if st.session_state.minuto_anterior is None:
    st.session_state.minuto_anterior = minuto_actual
elif st.session_state.minuto_anterior != minuto_actual:
    st.session_state.resultado_guardado = False
    st.session_state.voz_reproducida = False
    st.session_state.minuto_anterior = minuto_actual

with pantalla_dinamica.container():
    st.markdown(f'<div class="hora-box"><span>🕐 {hora_actual}</span></div>', unsafe_allow_html=True)

    if minuto_es_par:
        if segundo_actual < 55:
            st.markdown('<div class="signal-box signal-wait">📊 ESCANEANDO MERCADO...</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-panel">⏱️ Próxima señal en: <strong>{55 - segundo_actual}s</strong></div>', unsafe_allow_html=True)
            st.markdown('<div class="radar-container"><div class="radar radar-yellow"><div class="radar-center"><div class="radar-number radar-number-yellow">IA</div><div class="radar-label">ACTIVA</div></div></div></div>', unsafe_allow_html=True)
            time_module.sleep(1)
            st.rerun()
        else:
            if st.session_state.senal is None or st.session_state.estado != 'senal':
                st.session_state.senal = analizar_par(codigo, tipo)
                st.session_state.estado = 'senal'
            
            señal = st.session_state.senal
            es_call = señal['señal'] == 'CALL'
            segundos_restantes = 59 - segundo_actual
            
            if not st.session_state.voz_reproducida:
                if es_call:
                    reproducir_voz(f"Atención. Señal de compra. Confianza {int(señal['confianza'])} por ciento.")
                else:
                    reproducir_voz(f"Atención. Señal de venta. Confianza {int(señal['confianza'])} por ciento.")
                st.session_state.voz_reproducida = True
            
            clase_box = "signal-call" if es_call else "signal-put"
            clase_radar = "radar-green" if es_call else "radar-red"
            clase_numero = "radar-number-green" if es_call else "radar-number-red"
            tipo_signal = "🟢 COMPRA (CALL)" if es_call else "🔴 VENTA (PUT)"
            
            st.markdown(f'<div class="signal-box {clase_box}">{tipo_signal}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-panel">Confianza IA: <strong>{señal["confianza"]:.0f}%</strong> | <strong>{señal["razon"]}</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="radar-container"><div class="radar {clase_radar}"><div class="radar-center"><div class="radar-number {clase_numero}">{segundos_restantes}</div><div class="radar-label">SEGUNDOS</div></div></div></div>', unsafe_allow_html=True)
            time_module.sleep(0.3)
            st.rerun()
    else:
        if st.session_state.estado != 'operando':
            st.session_state.estado = 'operando'
            st.session_state.precio_real_entrada = obtener_precio_actual(codigo, tipo)
        
        señal = st.session_state.senal
        if señal is None:
            señal = {'señal': 'CALL', 'confianza': 60, 'razon': "Esperando"}
        
        es_call = señal['señal'] == 'CALL'
        segundos_restantes_op = 59 - segundo_actual
        
        clase_box = "signal-call" if es_call else "signal-put"
        clase_radar = "radar-green" if es_call else "radar-red"
        clase_numero = "radar-number-green" if es_call else "radar-number-red"
        tipo_signal = "🟢 EN CURSO (CALL)" if es_call else "🔴 EN CURSO (PUT)"
        
        st.markdown(f'<div class="signal-box {clase_box}">{tipo_signal}</div>', unsafe_allow_html=True)
        if st.session_state.precio_real_entrada:
            st.markdown(f'<div class="info-panel">Entrada: <strong>{round(st.session_state.precio_real_entrada, 5)}</strong></div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="radar-container"><div class="radar {clase_radar}"><div class="radar-center"><div class="radar-number {clase_numero}">{segundos_restantes_op}</div><div class="radar-label">EXPIRACIÓN</div></div></div></div>', unsafe_allow_html=True)
        
        if segundo_actual >= 58 and not st.session_state.resultado_guardado:
            if st.session_state.precio_real_entrada and st.session_state.precio_real_entrada > 0:
                precio_actual = obtener_precio_actual(codigo, tipo)
                resultado_interno = 'PERDIDA'
                if precio_actual is not None:
                    if es_call and precio_actual > st.session_state.precio_real_entrada:
                        resultado_interno = 'GANADA'
                    elif not es_call and precio_actual < st.session_state.precio_real_entrada:
                        resultado_interno = 'GANADA'
                aprender_en_segundo_plano(señal['señal'], resultado_interno, señal.get('indicadores', {}), st.session_state.precio_real_entrada)
                st.session_state.pesos = cargar_patrones()
            st.session_state.resultado_guardado = True
        time_module.sleep(0.5)
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
