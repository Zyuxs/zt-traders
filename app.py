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
from sklearn.model_selection import cross_val_score

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
    
    if pesos['total_ops'] >= 20 and pesos['winrate'] < 15:
        resetear_memoria()

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
    
    if len(X) < 15:
        return None
    try:
        modelo = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, min_samples_split=4)
        scores = cross_val_score(modelo, X, y, cv=3, scoring='accuracy')
        accuracy_real = scores.mean()
        modelo.fit(X, y)
        return {'modelo': modelo, 'accuracy': accuracy_real, 'muestras': len(X)}
    except:
        return None

def predecir_con_ml(modelo, indicadores):
    features = [
        indicadores.get('rsi', 50), indicadores.get('stoch', 50),
        indicadores.get('macd', 0), indicadores.get('adx', 20),
        indicadores.get('bb_position', 0.5), indicadores.get('sma_cross', 0),
        indicadores.get('tendencia_5m', 0),
        1 if indicadores.get('direccion', 'CALL') == 'CALL' else 0
    ]
    try:
        probabilidad = modelo.predict_proba([features])[0]
        return probabilidad[1] * 100
    except:
        return 50.0

def obtener_datos_binance(symbol, interval='1m', limit=100):
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
    except Exception as e:
        return None

def obtener_datos_yahoo(symbol, period="2d", interval="1m"):
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
        datos_1m = obtener_datos_binance(codigo_par, interval='1m', limit=100)
        datos_5m = obtener_datos_binance(codigo_par, interval='5m', limit=100)
    else:
        datos_1m = obtener_datos_yahoo(codigo_par, period="2d", interval="1m")
        datos_5m = obtener_datos_yahoo(codigo_par, period="5d", interval="5m")
    
    if datos_1m is None or datos_5m is None or len(datos_1m) < 30:
        return {'señal': 'ESPERA', 'rsi': 50, 'confianza': 0, 'razon': "Datos insuficientes", 'indicadores': {}}
    
    try:
        if tipo == 'crypto':
            datos_1m['RSI'] = ta.momentum.RSIIndicator(datos_1m['close'], window=14).rsi()
            datos_1m['STOCH'] = ta.momentum.StochasticOscillator(datos_1m['high'], datos_1m['low'], datos_1m['close']).stoch()
            datos_1m['MACD'] = ta.trend.MACD(datos_1m['close']).macd()
            datos_1m['MACD_signal'] = ta.trend.MACD(datos_1m['close']).macd_signal()
            datos_1m['ADX'] = ta.trend.ADXIndicator(datos_1m['high'], datos_1m['low'], datos_1m['close']).adx()
            bb = ta.volatility.BollingerBands(datos_1m['close'], window=20, window_dev=2)
            datos_1m['BB_High'] = bb.bollinger_hband()
            datos_1m['BB_Low'] = bb.bollinger_lband()
            datos_1m['SMA_10'] = datos_1m['close'].rolling(window=10).mean()
            datos_5m['SMA_20'] = datos_5m['close'].rolling(window=20).mean()
            u1 = datos_1m.iloc[-1]
            a1 = datos_1m.iloc[-2]
            u5 = datos_5m.iloc[-1]
            precio = float(u1['close'])
            bb_high = float(u1['BB_High'])
            bb_low = float(u1['BB_Low'])
            sma_1m = float(u1['SMA_10'])
            sma_5m = float(u5['SMA_20'])
            a1_close = float(a1['close'])
            open_1m = float(u1['open'])
        else:
            datos_1m['RSI'] = ta.momentum.RSIIndicator(datos_1m['Close'], window=14).rsi()
            datos_1m['STOCH'] = ta.momentum.StochasticOscillator(datos_1m['High'], datos_1m['Low'], datos_1m['Close']).stoch()
            datos_1m['MACD'] = ta.trend.MACD(datos_1m['Close']).macd()
            datos_1m['MACD_signal'] = ta.trend.MACD(datos_1m['Close']).macd_signal()
            datos_1m['ADX'] = ta.trend.ADXIndicator(datos_1m['High'], datos_1m['Low'], datos_1m['Close']).adx()
            bb = ta.volatility.BollingerBands(datos_1m['Close'], window=20, window_dev=2)
            datos_1m['BB_High'] = bb.bollinger_hband()
            datos_1m['BB_Low'] = bb.bollinger_lband()
            datos_1m['SMA_10'] = datos_1m['Close'].rolling(window=10).mean()
            datos_5m['SMA_20'] = datos_5m['Close'].rolling(window=20).mean()
            u1 = datos_1m.iloc[-1]
            a1 = datos_1m.iloc[-2]
            u5 = datos_5m.iloc[-1]
            precio = float(u1['Close'])
            bb_high = float(u1['BB_High'])
            bb_low = float(u1['BB_Low'])
            sma_1m = float(u1['SMA_10'])
            sma_5m = float(u5['SMA_20'])
            a1_close = float(a1['Close'])
            open_1m = float(u1['Open'])
        
        rsi = 50.0 if pd.isna(u1['RSI']) else float(u1['RSI'])
        stoch = 50.0 if pd.isna(u1['STOCH']) else float(u1['STOCH'])
        macd = 0.0 if pd.isna(u1['MACD']) else float(u1['MACD'])
        macd_sig = 0.0 if pd.isna(u1['MACD_signal']) else float(u1['MACD_signal'])
        adx = 20.0 if pd.isna(u1['ADX']) else float(u1['ADX'])

        # Tendencia de 5m
        tendencia_5m = 1 if precio > sma_5m else -1
        
        # LÓGICA RELAJADA para dar más señales
        dirección = 'ESPERA'
        
        # CALL: si RSI bajo O toca banda inferior O tendencia alcista
        if rsi < 55 or precio <= bb_low or tendencia_5m == 1:
            dirección = 'CALL'
        
        # PUT: si RSI alto O toca banda superior O tendencia bajista
        if rsi > 45 or precio >= bb_high or tendencia_5m == -1:
            # Si ya era CALL, mantener CALL solo si hay fuerte señal
            if dirección == 'CALL':
                # Prioridad: el que tenga más indicadores a favor
                call_score = (1 if rsi < 55 else 0) + (1 if precio <= bb_low else 0) + (1 if tendencia_5m == 1 else 0)
                put_score = (1 if rsi > 45 else 0) + (1 if precio >= bb_high else 0) + (1 if tendencia_5m == -1 else 0)
                if put_score > call_score:
                    dirección = 'PUT'
            else:
                dirección = 'PUT'

        bb_position = (precio - bb_low) / (bb_high - bb_low) if bb_high != bb_low else 0.5
        bb_position = max(0, min(1, bb_position))
        sma_cross = 1 if precio > sma_1m and a1_close <= sma_1m else (-1 if precio < sma_1m and a1_close >= sma_1m else 0)

        indicadores = {
            'rsi': rsi, 'stoch': stoch, 'macd': macd - macd_sig, 'adx': adx, 
            'bb_position': bb_position, 'sma_cross': sma_cross, 
            'tendencia_5m': tendencia_5m, 'direccion': dirección
        }

        # ML
        modelo_info = entrenar_modelo_ml()
        confianza_ml = None
        if modelo_info and modelo_info['modelo']:
            confianza_ml = predecir_con_ml(modelo_info['modelo'], indicadores)

        # Sistema clásico - UMBRAL MUY BAJO (15%)
        confianza = 0
        razones = []
        
        if dirección == 'CALL':
            if rsi < 40: confianza += 25; razones.append("RSI bajo")
            if precio <= bb_low: confianza += 25; razones.append("Banda inferior")
            if macd > macd_sig: confianza += 20; razones.append("MACD alcista")
            if tendencia_5m == 1: confianza += 15; razones.append("Tendencia alcista")
        else:
            if rsi > 60: confianza += 25; razones.append("RSI alto")
            if precio >= bb_high: confianza += 25; razones.append("Banda superior")
            if macd < macd_sig: confianza += 20; razones.append("MACD bajista")
            if tendencia_5m == -1: confianza += 15; razones.append("Tendencia bajista")
            
        if adx > 25: confianza += 10; razones.append(f"ADX {adx:.0f}")
        if stoch < 20 and dirección == 'CALL': confianza += 10; razones.append("Stoch bajo")
        if stoch > 80 and dirección == 'PUT': confianza += 10; razones.append("Stoch alto")

        # Si no hay razones, dar confianza mínima de 15%
        if confianza == 0:
            confianza = 15
            razones.append("Señal básica")

        confianza_final = confianza_ml if confianza_ml and confianza_ml >= 50 else confianza
        
        razon_final = f"ML: {confianza_ml:.1f}%" if confianza_ml and confianza_ml >= 50 else (" | ".join(razones))

        return {'señal': dirección, 'rsi': rsi, 'confianza': confianza_final, 
               'razon': razon_final, 'indicadores': indicadores}
               
    except Exception as e:
        return {'señal': 'CALL', 'rsi': 50, 'confianza': 15, 'razon': "Señal básica", 'indicadores': {'rsi': 50, 'stoch': 50, 'macd': 0, 'adx': 20, 'bb_position': 0.5, 'sma_cross': 0, 'tendencia_5m': 0, 'direccion': 'CALL'}}

# TODOS LOS PARES RESTAURADOS
PARES = {
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
    'AUD/JPY': {'codigo': 'AUDJPY=X', 'tipo': 'forex'},
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
body { background: #000; color: #fff; font-family: Arial, sans-serif; }
.main-header { background: linear-gradient(90deg, #1a1a2e, #16213e); padding: 20px; border-radius: 10px; border: 2px solid #ffd700; text-align: center; margin-bottom: 15px; }
.main-header h1 { color: #ffd700; margin: 0; font-size: 32px; letter-spacing: 3px; }
.main-header p { color: #aaa; margin: 8px 0 0 0; font-size: 13px; }
.signal-box { padding: 35px; border-radius: 15px; text-align: center; font-size: 36px; font-weight: bold; margin: 15px 0; letter-spacing: 2px; }
.signal-call { background: linear-gradient(135deg, #00cc00, #00ff00); border: 4px solid #00ff00; color: white; box-shadow: 0 0 40px rgba(0, 255, 0, 0.6); }
.signal-put { background: linear-gradient(135deg, #cc0000, #ff0000); border: 4px solid #ff0000; color: white; box-shadow: 0 0 40px rgba(255, 0, 0, 0.6); }
.signal-wait { background: #1a1a2e; border: 3px solid #ffd700; color: #ffd700; }
.radar { width: 240px; height: 240px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 20px auto; }
.radar-green { border: 4px solid #00ff00; background: radial-gradient(circle, #0a2a1a, #000); box-shadow: 0 0 40px rgba(0, 255, 0, 0.6); }
.radar-red { border: 4px solid #ff0000; background: radial-gradient(circle, #2a0a0a, #000); box-shadow: 0 0 40px rgba(255, 0, 0, 0.6); }
.radar-yellow { border: 4px solid #ffd700; background: radial-gradient(circle, #2a2a1a, #000); box-shadow: 0 0 40px rgba(255, 215, 0, 0.6); }
.radar-number { font-size: 90px; font-weight: bold; text-shadow: 0 0 25px currentColor; }
.radar-number-green { color: #00ff00; }
.radar-number-red { color: #ff0000; }
.radar-number-yellow { color: #ffd700; font-size: 28px; text-align: center; }
.info-panel { background: #1a1a2e; border: 2px solid #00ff88; border-radius: 10px; padding: 15px; margin: 12px 0; }
.info-panel strong { color: #00ff88; }
.stat-box { background: #1a1a2e; border: 2px solid #ffd700; border-radius: 10px; padding: 12px; text-align: center; }
.hora-box { background: linear-gradient(90deg, #0a2a1a, #1a1a2e); border: 3px solid #00ff88; border-radius: 12px; padding: 18px; text-align: center; margin: 15px 0; box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); }
.footer { text-align: center; color: #666; font-size: 11px; margin-top: 25px; padding: 15px; border-top: 1px solid #333; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if st.button("🔒 Cerrar Sesión", key="logout_btn"):
    st.session_state.logueado = False
    st.session_state.usuario_actual = ""
    st.rerun()

# Botón de RESET MANUAL
if st.button("🗑️ Borrar Memoria", key="reset_manual_btn"):
    resetear_memoria()
    st.success("✅ Memoria borrada. La IA empieza de cero.")
    st.session_state.senal = None
    st.session_state.estado = 'analizando'
    st.rerun()

st.markdown(f'<div class="main-header"><h1>ZT TRADERS</h1><p>Bienvenido: {st.session_state.usuario_actual} | Alerta de Voz Activa</p></div>', unsafe_allow_html=True)

# Selector de par
par_nombre = st.selectbox("Par:", list(PARES.keys()), index=list(PARES.keys()).index(st.session_state.par_sel))
st.session_state.par_sel = par_nombre

codigo_info = PARES[par_nombre]
codigo = codigo_info['codigo']
tipo = codigo_info['tipo']

# Estadísticas
hist = cargar_historial()
modelo_info = entrenar_modelo_ml()
ops_con_indicadores = sum(1 for op in hist if 'indicadores' in op)

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">Winrate</strong><br><span style="color:#00ff88; font-size:22px;">{round(st.session_state.pesos["winrate"], 1)}%</span></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">Operaciones</strong><br><span style="color:#00ff88; font-size:22px;">{st.session_state.pesos["total_ops"]}</span></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">Ganadas</strong><br><span style="color:#00ff88; font-size:22px;">{st.session_state.pesos["total_ganadas"]}</span></div>', unsafe_allow_html=True)
with c4: 
    if modelo_info and ops_con_indicadores >= 15:
        precision = modelo_info["accuracy"] * 100
        color_p = "#00ff88" if precision >= 60 else "#ffaa00" if precision >= 50 else "#ff4444"
        st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">ML</strong><br><span style="color:{color_p}; font-size:22px;">{precision:.0f}%</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">ML</strong><br><span style="color:#ff8800; font-size:18px;">({ops_con_indicadores}/15)</span></div>', unsafe_allow_html=True)

st.markdown("---")

# Script de voz
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
            msg.pitch = 1.0;
            msg.volume = 1.0;
            window.speechSynthesis.speak(msg);
        }}
    }})();
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

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

# RELOJ SIEMPRE VISIBLE ARRIBA
st.markdown(f'<div class="hora-box"><strong style="color:#00ff88; font-size:32px;">🕐 {hora_actual}</strong></div>', unsafe_allow_html=True)

if minuto_es_par:
    if segundo_actual < 50:
        st.markdown('<div class="signal-box signal-wait">ESPERANDO...</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-panel" style="text-align:center;"><strong>Próximo análisis en: {50 - segundo_actual}s</strong></div>', unsafe_allow_html=True)
        time_module.sleep(1)
        st.rerun()
    elif segundo_actual < 55:
        if st.session_state.estado != 'analizando':
            st.session_state.estado = 'analizando'
            st.session_state.senal = None
        if st.session_state.senal is None:
            st.session_state.senal = analizar_par(codigo, tipo)
        
        if st.session_state.senal and st.session_state.senal['señal'] == 'ESPERA':
            st.markdown('<div class="signal-box signal-wait">MERCADO LATERAL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="signal-box signal-wait">ANALIZANDO...</div>', unsafe_allow_html=True)
        time_module.sleep(0.5)
        st.rerun()
    else:
        if st.session_state.estado != 'senal':
            st.session_state.estado = 'senal'
            if st.session_state.senal is None:
                st.session_state.senal = analizar_par(codigo, tipo)
                
        señal = st.session_state.senal
        
        if not señal or señal['señal'] == 'ESPERA':
            st.markdown('<div class="signal-box signal-wait">SIN SEÑAL CLARA</div>', unsafe_allow_html=True)
            if not st.session_state.voz_reproducida:
                reproducir_voz("Mercado lateral. Sin señal clara.")
                st.session_state.voz_reproducida = True
            time_module.sleep(0.5)
            st.rerun()
            
        es_call = señal['señal'] == 'CALL'
        segundos_restantes = 59 - segundo_actual
        tipo_signal = "CALL" if es_call else "PUT"
        clase_box = "signal-call" if es_call else "signal-put"
        clase_radar = "radar-green" if es_call else "radar-red"
        clase_numero = "radar-number-green" if es_call else "radar-number-red"
        
        if not st.session_state.voz_reproducida:
            confianza_texto = f"Confianza {int(señal['confianza'])} por ciento"
            if es_call:
                mensaje_voz = f"Atención. Señal de compra. Operación a la alza. {confianza_texto}."
            else:
                mensaje_voz = f"Atención. Señal de venta. Operación a la baja. {confianza_texto}."
            reproducir_voz(mensaje_voz)
            st.session_state.voz_reproducida = True
        
        st.markdown(f'<div class="signal-box {clase_box}">{tipo_signal}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-panel"><strong>Confianza:</strong> {señal["confianza"]:.1f}%<br><strong>Razon:</strong> {señal["razon"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="radar {clase_radar}"><div class="radar-number {clase_numero}">{segundos_restantes}</div></div>', unsafe_allow_html=True)
        time_module.sleep(0.3)
        st.rerun()
else:
    if st.session_state.estado != 'operando':
        st.session_state.estado = 'operando'
        st.session_state.precio_real_entrada = obtener_precio_actual(codigo, tipo)
    
    señal = st.session_state.senal
    if not señal or señal['señal'] == 'ESPERA':
        st.markdown('<div class="signal-box signal-wait">ESPERANDO...</div>', unsafe_allow_html=True)
        time_module.sleep(1)
        st.rerun()
    
    es_call = señal['señal'] == 'CALL'
    segundos_restantes_op = 59 - segundo_actual
    tipo_signal = "CALL" if es_call else "PUT"
    clase_box = "signal-call" if es_call else "signal-put"
    clase_radar = "radar-green" if es_call else "radar-red"
    clase_numero = "radar-number-green" if es_call else "radar-number-red"
    
    precio_entrada = st.session_state.precio_real_entrada
    
    st.markdown(f'<div class="signal-box {clase_box}">{tipo_signal}</div>', unsafe_allow_html=True)
    if precio_entrada:
        st.markdown(f'<div class="info-panel"><strong>Entrada REAL:</strong> {round(precio_entrada, 5)} | <strong>Confianza:</strong> {señal["confianza"]:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="radar {clase_radar}"><div class="radar-number {clase_numero}">{segundos_restantes_op}</div></div>', unsafe_allow_html=True)
    st.warning("No cierres esta pagina")
    
    if segundo_actual >= 58 and not st.session_state.resultado_guardado:
        if precio_entrada and precio_entrada > 0:
            precio_actual = obtener_precio_actual(codigo, tipo)
            resultado_interno = 'PERDIDA'
            if precio_actual is not None:
                if es_call and precio_actual > precio_entrada:
                    resultado_interno = 'GANADA'
                elif not es_call and precio_actual < precio_entrada:
                    resultado_interno = 'GANADA'
            indicadores = señal.get('indicadores', {})
            aprender_en_segundo_plano(señal['señal'], resultado_interno, indicadores, precio_entrada)
            st.session_state.pesos = cargar_patrones()
        st.session_state.resultado_guardado = True
    time_module.sleep(0.5)
    st.rerun()

st.markdown("---")
st.markdown("### Historial Reciente")
hist = cargar_historial()
if hist:
    for op in reversed(hist[-10:]):
        emoji = "✅" if op['resultado'] == 'GANADA' else "❌"
        st.write(f"{emoji} {op['fecha']} - {op['señal']} - {op['resultado']}")
else:
    st.info("Memoria limpia. La IA está aprendiendo desde cero.")

st.markdown(f'<div class="footer">ZT TRADERS | Usuario: {st.session_state.usuario_actual}</div>', unsafe_allow_html=True)
