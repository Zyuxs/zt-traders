import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import ta
import time as time_module
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

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
    body { background: #000; color: #fff; font-family: 'Arial Black', Arial, sans-serif; margin: 0; padding: 0; }
    .login-container { max-width: 350px; margin: 60px auto; padding: 25px; background: linear-gradient(135deg, #0a0a1a, #1a0a2e); border-radius: 15px; border: 2px solid #ffd700; box-shadow: 0 0 40px rgba(255, 215, 0, 0.4); }
    .login-title { color: #ffd700; text-align: center; font-size: 26px; font-weight: bold; margin-bottom: 8px; letter-spacing: 3px; }
    .login-subtitle { color: #aaa; text-align: center; font-size: 12px; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">⚡ ZT TRADERS</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Sistema de Acceso Privado</div>', unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        usuario_input = st.text_input("Usuario", placeholder="Ingresa tu usuario")
        clave_input = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña")
        submit = st.form_submit_button("⚡ ENTRAR", use_container_width=True)
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

def entrenar_modelo_ml():
    hist = cargar_historial()
    X = []
    y = []
    for op in hist:
        if 'indicadores' in op:
            ind = op['indicadores']
            features = [
                ind.get('rsi', 50), ind.get('stoch', 50), ind.get('macd', 0),
                ind.get('adx', 20), ind.get('bb_position', 0.5),
                ind.get('sma_cross', 0), ind.get('tendencia_5m', 0),
                1 if op['señal'] == 'CALL' else 0
            ]
            X.append(features)
            y.append(1 if op['resultado'] == 'GANADA' else 0)
    if len(X) < 10:
        return None
    try:
        modelo = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        modelo.fit(X, y)
        return {'modelo': modelo, 'accuracy': 0.5, 'muestras': len(X)}
    except:
        return None

def obtener_datos_binance(symbol, interval='1m', limit=50):
    try:
        url = f"https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        response = requests.get(url, params=params, timeout=10)
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
        datos = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(datos.columns, pd.MultiIndex):
            datos.columns = datos.columns.get_level_values(0)
        datos = datos[['Open', 'High', 'Low', 'Close']].copy()
        return datos
    except:
        return None

def obtener_precio_actual(codigo_par, tipo='crypto'):
    if tipo == 'crypto':
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={codigo_par}"
            response = requests.get(url, timeout=5)
            data = response.json()
            return float(data['price'])
        except:
            return None
    else:
        try:
            datos = yf.download(codigo_par, period="1d", interval="1m", progress=False)
            if isinstance(datos.columns, pd.MultiIndex):
                datos.columns = datos.columns.get_level_values(0)
            if len(datos) < 1: return None
            return float(datos['Close'].iloc[-1])
        except:
            return None

def analizar_par(codigo_par, tipo='crypto'):
    if tipo == 'crypto':
        datos_1m = obtener_datos_binance(codigo_par, interval='1m', limit=50)
    else:
        datos_1m = obtener_datos_yahoo(codigo_par, period="1d", interval="1m")
    
    if datos_1m is None or len(datos_1m) < 20:
        return {'señal': 'CALL', 'confianza': 50, 'razon': "Señal básica", 'indicadores': {}}
    
    try:
        if tipo == 'crypto':
            datos_1m['RSI'] = ta.momentum.RSIIndicator(datos_1m['close'], window=14).rsi()
            datos_1m['STOCH'] = ta.momentum.StochasticOscillator(datos_1m['high'], datos_1m['low'], datos_1m['close']).stoch()
            datos_1m['MACD'] = ta.trend.MACD(datos_1m['close']).macd()
            datos_1m['MACD_signal'] = ta.trend.MACD(datos_1m['close']).macd_signal()
            bb = ta.volatility.BollingerBands(datos_1m['close'], window=20, window_dev=2)
            datos_1m['BB_High'] = bb.bollinger_hband()
            datos_1m['BB_Low'] = bb.bollinger_lband()
            datos_1m['SMA_10'] = datos_1m['close'].rolling(window=10).mean()
            u1 = datos_1m.iloc[-1]
            a1 = datos_1m.iloc[-2]
            precio = float(u1['close'])
            bb_high = float(u1['BB_High'])
            bb_low = float(u1['BB_Low'])
            sma_1m = float(u1['SMA_10'])
            a1_close = float(a1['close'])
        else:
            datos_1m['RSI'] = ta.momentum.RSIIndicator(datos_1m['Close'], window=14).rsi()
            datos_1m['STOCH'] = ta.momentum.StochasticOscillator(datos_1m['High'], datos_1m['Low'], datos_1m['Close']).stoch()
            datos_1m['MACD'] = ta.trend.MACD(datos_1m['Close']).macd()
            datos_1m['MACD_signal'] = ta.trend.MACD(datos_1m['Close']).macd_signal()
            bb = ta.volatility.BollingerBands(datos_1m['Close'], window=20, window_dev=2)
            datos_1m['BB_High'] = bb.bollinger_hband()
            datos_1m['BB_Low'] = bb.bollinger_lband()
            datos_1m['SMA_10'] = datos_1m['Close'].rolling(window=10).mean()
            u1 = datos_1m.iloc[-1]
            a1 = datos_1m.iloc[-2]
            precio = float(u1['Close'])
            bb_high = float(u1['BB_High'])
            bb_low = float(u1['BB_Low'])
            sma_1m = float(u1['SMA_10'])
            a1_close = float(a1['Close'])
        
        rsi = 50.0 if pd.isna(u1['RSI']) else float(u1['RSI'])
        stoch = 50.0 if pd.isna(u1['STOCH']) else float(u1['STOCH'])
        macd = 0.0 if pd.isna(u1['MACD']) else float(u1['MACD'])
        macd_sig = 0.0 if pd.isna(u1['MACD_signal']) else float(u1['MACD_signal'])

        if rsi < 50 or precio <= bb_low:
            dirección = 'CALL'
        else:
            dirección = 'PUT'

        confianza = 50
        razones = []
        
        if dirección == 'CALL':
            if rsi < 40: confianza += 20; razones.append("RSI bajo")
            if precio <= bb_low: confianza += 20; razones.append("Banda inferior")
            if macd > macd_sig: confianza += 10; razones.append("MACD+")
        else:
            if rsi > 60: confianza += 20; razones.append("RSI alto")
            if precio >= bb_high: confianza += 20; razones.append("Banda superior")
            if macd < macd_sig: confianza += 10; razones.append("MACD-")
        
        if not razones:
            razones.append("Tendencia")

        indicadores = {
            'rsi': rsi, 'stoch': stoch, 'macd': macd - macd_sig,
            'bb_position': 0.5, 'sma_cross': 0, 
            'tendencia_5m': 1 if precio > sma_1m else -1, 'direccion': dirección
        }

        return {'señal': dirección, 'confianza': confianza, 
               'razon': " | ".join(razones), 'indicadores': indicadores}
               
    except Exception as e:
        return {'señal': 'CALL', 'confianza': 50, 'razon': "Señal básica", 'indicadores': {}}

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

PARES = {
    'Bitcoin': {'codigo': 'BTCUSDT', 'tipo': 'crypto'},
    'Ethereum': {'codigo': 'ETHUSDT', 'tipo': 'crypto'},
    'Polkadot': {'codigo': 'DOTUSDT', 'tipo': 'crypto'},
    'Worldcoin': {'codigo': 'WLDUSDT', 'tipo': 'crypto'},
    'Binance Coin': {'codigo': 'BNBUSDT', 'tipo': 'crypto'},
    'Cardano': {'codigo': 'ADAUSDT', 'tipo': 'crypto'},
    'Solana': {'codigo': 'SOLUSDT', 'tipo': 'crypto'},
    'Ripple': {'codigo': 'XRPUSDT', 'tipo': 'crypto'},
    'Dogecoin': {'codigo': 'DOGEUSDT', 'tipo': 'crypto'},
    'Polygon': {'codigo': 'MATICUSDT', 'tipo': 'crypto'},
    'Litecoin': {'codigo': 'LTCUSDT', 'tipo': 'crypto'}
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
body { background: #000; color: #fff; font-family: 'Arial Black', Arial, sans-serif; margin: 0; padding: 0; }
.main-container { max-width: 450px; margin: 0 auto; padding: 8px; }
.header { background: linear-gradient(90deg, #1a0a2e, #0a1a2e); padding: 10px; border-radius: 12px; border: 2px solid #ffd700; text-align: center; margin-bottom: 8px; box-shadow: 0 0 20px rgba(255, 215, 0, 0.3); }
.header h1 { color: #ffd700; margin: 0; font-size: 22px; letter-spacing: 3px; text-shadow: 0 0 10px #ffd700; }
.header p { color: #aaa; margin: 3px 0 0 0; font-size: 10px; }
.stats-row { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 4px; margin-bottom: 8px; }
.stat-box { background: linear-gradient(135deg, #0a1a2e, #1a0a2e); border: 2px solid #ffd700; border-radius: 8px; padding: 6px; text-align: center; box-shadow: 0 0 10px rgba(255, 215, 0, 0.2); }
.stat-box strong { color: #ffd700; font-size: 10px; display: block; }
.stat-box span { color: #00ff88; font-size: 16px; font-weight: bold; }
.hora-box { background: linear-gradient(90deg, #0a2a1a, #1a1a2e); border: 2px solid #00ff88; border-radius: 10px; padding: 10px; text-align: center; margin-bottom: 8px; box-shadow: 0 0 15px rgba(0, 255, 136, 0.4); }
.hora-box span { color: #00ff88; font-size: 26px; font-weight: bold; text-shadow: 0 0 10px #00ff88; }
.signal-box { padding: 18px; border-radius: 12px; text-align: center; font-size: 30px; font-weight: bold; margin: 8px 0; letter-spacing: 3px; animation: pulse 1.5s infinite; }
.signal-call { background: linear-gradient(135deg, #00cc00, #00ff00); border: 3px solid #00ff00; color: white; box-shadow: 0 0 40px rgba(0, 255, 0, 0.8); }
.signal-put { background: linear-gradient(135deg, #cc0000, #ff0000); border: 3px solid #ff0000; color: white; box-shadow: 0 0 40px rgba(255, 0, 0, 0.8); }
.signal-wait { background: linear-gradient(135deg, #1a1a2e, #0a1a2e); border: 2px solid #ffd700; color: #ffd700; animation: none; }
@keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.02); } }
.radar-container { position: relative; width: 200px; height: 200px; margin: 10px auto; }
.radar { width: 200px; height: 200px; border-radius: 50%; position: relative; overflow: hidden; border: 3px solid; }
.radar-green { border-color: #00ff00; background: radial-gradient(circle, #0a2a1a, #000); box-shadow: 0 0 40px rgba(0, 255, 0, 0.6), inset 0 0 40px rgba(0, 255, 0, 0.2); }
.radar-red { border-color: #ff0000; background: radial-gradient(circle, #2a0a0a, #000); box-shadow: 0 0 40px rgba(255, 0, 0, 0.6), inset 0 0 40px rgba(255, 0, 0, 0.2); }
.radar-yellow { border-color: #ffd700; background: radial-gradient(circle, #2a2a1a, #000); box-shadow: 0 0 40px rgba(255, 215, 0, 0.6), inset 0 0 40px rgba(255, 215, 0, 0.2); }
.radar-sweep { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: conic-gradient(from 0deg, transparent 0deg, rgba(0, 255, 136, 0.4) 60deg, transparent 120deg); animation: sweep 3s linear infinite; }
.radar-sweep-red { background: conic-gradient(from 0deg, transparent 0deg, rgba(255, 0, 0, 0.4) 60deg, transparent 120deg); }
.radar-sweep-yellow { background: conic-gradient(from 0deg, transparent 0deg, rgba(255, 215, 0, 0.4) 60deg, transparent 120deg); }
@keyframes sweep { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.radar-center { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; z-index: 10; }
.radar-number { font-size: 70px; font-weight: bold; text-shadow: 0 0 20px currentColor; line-height: 1; }
.radar-number-green { color: #00ff00; }
.radar-number-red { color: #ff0000; }
.radar-number-yellow { color: #ffd700; font-size: 20px; }
.radar-label { color: #fff; font-size: 10px; margin-top: 5px; letter-spacing: 2px; }
.info-panel { background: linear-gradient(135deg, #0a1a2e, #1a0a2e); border: 2px solid #00ff88; border-radius: 10px; padding: 10px; margin: 8px 0; font-size: 12px; box-shadow: 0 0 15px rgba(0, 255, 136, 0.3); }
.info-panel strong { color: #00ff88; }
.controls { display: flex; gap: 5px; margin-bottom: 8px; }
.controls select { flex: 1; background: #1a1a2e; color: #fff; border: 2px solid #ffd700; border-radius: 8px; padding: 8px; font-size: 12px; font-weight: bold; }
.controls button { background: #ff4444; color: white; border: none; border-radius: 8px; padding: 8px 12px; font-size: 11px; font-weight: bold; cursor: pointer; }
.footer { text-align: center; color: #666; font-size: 10px; margin-top: 10px; padding: 8px; border-top: 1px solid #333; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown(f'<div class="header"><h1>⚡ ZT TRADERS</h1><p>Bienvenido: {st.session_state.usuario_actual} | IA ACTIVA</p></div>', unsafe_allow_html=True)

# Controles
col1, col2 = st.columns([3, 1])
with col1:
    par_nombre = st.selectbox("", list(PARES.keys()), index=list(PARES.keys()).index(st.session_state.par_sel), label_visibility="collapsed")
    st.session_state.par_sel = par_nombre
with col2:
    if st.button("🗑️ Reset", key="reset_btn"):
        resetear_memoria()
        st.session_state.senal = None
        st.session_state.estado = 'analizando'
        st.rerun()

codigo_info = PARES[par_nombre]
codigo = codigo_info['codigo']
tipo = codigo_info['tipo']

# Estadísticas
hist = cargar_historial()
ops = st.session_state.pesos['total_ops']
ganadas = st.session_state.pesos['total_ganadas']
winrate = round(st.session_state.pesos['winrate'], 1)

st.markdown(f"""
<div class="stats-row">
    <div class="stat-box"><strong>Winrate</strong><span>{winrate}%</span></div>
    <div class="stat-box"><strong>Operaciones</strong><span>{ops}</span></div>
    <div class="stat-box"><strong>Ganadas</strong><span>{ganadas}</span></div>
    <div class="stat-box"><strong>ML</strong><span style="font-size:12px;">{len(hist)}/10</span></div>
</div>
""", unsafe_allow_html=True)

# Reloj
ahora = datetime.now()
hora_actual = ahora.strftime('%H:%M:%S')
st.markdown(f'<div class="hora-box"><span>🕐 {hora_actual}</span></div>', unsafe_allow_html=True)

minuto_actual = ahora.minute
segundo_actual = ahora.second
minuto_es_par = (minuto_actual % 2 == 0)

if st.session_state.minuto_anterior is None:
    st.session_state.minuto_anterior = minuto_actual
elif st.session_state.minuto_anterior != minuto_actual:
    st.session_state.resultado_guardado = False
    st.session_state.voz_reproducida = False
    st.session_state.minuto_anterior = minuto_actual

# CICLO
if minuto_es_par:
    if segundo_actual < 55:
        st.markdown('<div class="signal-box signal-wait">📊 ANALIZANDO MERCADO...</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-panel" style="text-align:center;"><strong>⏱️ Señal en: {55 - segundo_actual}s</strong></div>', unsafe_allow_html=True)
        st.markdown('<div class="radar-container"><div class="radar radar-yellow"><div class="radar-sweep radar-sweep-yellow"></div><div class="radar-center"><div class="radar-number radar-number-yellow">IA</div><div class="radar-label">ACTIVA</div></div></div></div>', unsafe_allow_html=True)
        time_module.sleep(1)
        st.rerun()
   
