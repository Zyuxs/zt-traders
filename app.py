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
    """SIEMPRE devuelve CALL o PUT, nunca ESPERA"""
    if tipo == 'crypto':
        datos_1m = obtener_datos_binance(codigo_par, interval='1m', limit=100)
        datos_5m = obtener_datos_binance(codigo_par, interval='5m', limit=100)
    else:
        datos_1m = obtener_datos_yahoo(codigo_par, period="2d", interval="1m")
        datos_5m = obtener_datos_yahoo(codigo_par, period="5d", interval="5m")
    
    # Si no hay datos, dar señal por defecto
    if datos_1m is None or datos_5m is None or len(datos_1m) < 30:
        return {'señal': 'CALL', 'rsi': 50, 'confianza': 50, 'razon': "Señal por defecto", 
                'indicadores': {'rsi': 50, 'stoch': 50, 'macd': 0, 'adx': 20, 
                               'bb_position': 0.5, 'sma_cross': 0, 'tendencia_5m': 0, 'direccion': 'CALL'}}
    
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

        # LÓGICA SIMPLE: SIEMPRE da CALL o PUT
        # Basado en RSI y tendencia de 5m
        tendencia_5m = 1 if precio > sma_5m else -1
        
        # Puntuación CALL vs PUT
        call_score = 0
        put_score = 0
        
        if rsi < 50: call_score += 2
        if rsi > 50: put_score += 2
        if precio <= bb_low: call_score += 2
        if precio >= bb_high: put_score += 2
        if tendencia_5m == 1: call_score += 1
        if tendencia_5m == -1: put_score += 1
        if macd > macd_sig: call_score += 1
        if macd < macd_sig: put_score += 1
        if stoch < 20: call_score += 1
        if stoch > 80: put_score += 1
        
        # Siempre elegir el ganador
        if call_score >= put_score:
            dirección = 'CALL'
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

        # Confianza basada en diferencia de scores
        diferencia = abs(call_score - put_score)
        confianza = min(30 + diferencia * 10, 95)
        
        razones = []
        if dirección == 'CALL':
            if rsi < 50: razones.append("RSI bajo")
            if precio <= bb_low: razones.append("Banda inferior")
            if tendencia_5m == 1: razones.append("Tendencia alcista")
        else:
            if rsi > 50: razones.append("RSI alto")
            if precio >= bb_high: razones.append("Banda superior")
            if tendencia_5m == -1: razones.append("Tendencia bajista")
        
        if not razones:
            razones.append("Señal básica")

        confianza_final = confianza_ml if confianza_ml and confianza_ml >= 50 else confianza
        razon_final = f"ML: {confianza_ml:.1f}%" if confianza_ml and confianza_ml >= 50 else (" | ".join(razones))

        return {'señal': dirección, 'rsi': rsi, 'confianza': confianza_final, 
               'razon': razon_final, 'indicadores': indicadores, 'precio': precio}
               
    except Exception as e:
        return {'señal': 'CALL', 'rsi': 50, 'confianza': 50, 'razon': "Señal básica", 
                'indicadores': {'rsi': 50, 'stoch': 50, 'macd': 0, 'adx': 20, 
                               'bb_position': 0.5, 'sma_cross': 0, 'tendencia_5m': 0, 'direccion': 'CALL'}, 'precio': 0}

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
body { background: #0a0a1a; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 0; }
.container { max-width: 480px; margin: 0 auto; padding: 10px; }
.header { background: linear-gradient(90deg, #1a1a2e, #16213e); padding: 12px; border-radius: 10px; border: 2px solid #ffd700; text-align: center; margin-bottom: 10px; }
.header h1 { color: #ffd700; margin: 0; font-size: 22px; letter-spacing: 2px; }
.header p { color: #aaa; margin: 4px 0 0 0; font-size: 11px; }
.signal-box { padding: 25px; border-radius: 12px; text-align: center; font-size: 32px; font-weight: bold; margin: 8px 0; letter-spacing: 2px; }
.signal-call { background: linear-gradient(135deg, #00cc00, #00ff00); border: 3px solid #00ff00; color: white; box-shadow: 0 0 30px rgba(0, 255, 0, 0.5); }
.signal-put { background: linear-gradient(135deg, #cc0000, #ff0000); border: 3px solid #ff0000; color: white; box-shadow: 0 0 30px rgba(255, 0, 0, 0.5); }
.signal-wait { background: #1a1a2e; border: 2px solid #ffd700; color: #ffd700; font-size: 24px; }
.radar-container { text-align: center; margin: 10px 0; }
.radar { width: 180px; height: 180px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; }
.radar-green { border: 3px solid #00ff00; background: radial-gradient(circle, #0a2a1a, #000); box-shadow: 0 0 30px rgba(0, 255, 0, 0.5); }
.radar-red { border: 3px solid #ff0000; background: radial-gradient(circle, #2a0a0a, #000); box-shadow: 0 0 30px rgba(255, 0, 0, 0.5); }
.radar-yellow { border: 3px solid #ffd700; background: radial-gradient(circle, #2a2a1a, #000); box-shadow: 0 0 30px rgba(255, 215, 0, 0.5); }
.radar-number { font-size: 70px; font-weight: bold; text-shadow: 0 0 20px currentColor; }
.radar-number-green { color: #00ff00; }
.radar-number-red { color: #ff0000; }
.radar-number-yellow { color: #ffd700; font-size: 22px; text-align: center; }
.info-row { display: flex; justify-content: space-between; margin: 6px 0; }
.info-item { background: #1a1a2e; border: 2px solid #00ff88; border-radius: 8px; padding: 10px; flex: 1; margin: 0 4px; text-align: center; }
.info-item strong { color: #00ff88; font-size: 11px; display: block; }
.info-item span { color: #ffd700; font-size: 18px; font-weight: bold; }
.stats-row { display: flex; justify-content: space-between; margin: 6px 0; }
.stat-item { background: #1a1a2e; border: 2px solid #ffd700; border-radius: 8px; padding: 8px; flex: 1; margin: 0 3px; text-align: center; }
.stat-item strong { color: #ffd700; font-size: 10px; display: block; }
.stat-item span { color: #00ff88; font-size: 16px; font-weight: bold; }
.hora-box { background: linear-gradient(90deg, #0a2a1a, #1a1a2e); border: 2px solid #00ff88; border-radius: 8px; padding: 10px; text-align: center; margin: 8px 0; }
.footer { text-align: center; color: #666; font-size: 10px; margin-top: 15px; padding: 10px; border-top: 1px solid #333; }
.btn-reset { background: #ff4444; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 12px; margin: 5px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

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

# Contenedor principal compacto
st.markdown('<div class="container">', unsafe_allow_html=True)

# Header compacto
st.markdown(f'<div class="header"><h1>ZT TRADERS</h1><p>{st.session_state.usuario_actual} | Voz Activa</p></div>', unsafe_allow_html=True)

# Botones pequeños
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    par_nombre = st.selectbox("", list(PARES.keys()), index=list(PARES.keys()).index(st.session_state.par_sel), label_visibility="collapsed")
    st.session_state.par_sel = par_nombre
with col_btn2:
    if st.button("🗑️", key="reset_btn"):
        resetear_memoria()
        st.session_state.senal = None
        st.session_state.estado = 'analizando'
        st.rerun()

codigo_info = PARES[par_nombre]
codigo = codigo_info['codigo']
tipo = codigo_info['tipo']

# Estadísticas compactas
hist = cargar_historial()
modelo_info = entrenar_modelo_ml()
ops_con_indicadores = sum(1 for op in hist if 'indicadores' in op)

st.markdown('<div class="stats-row">', unsafe_allow_html=True)
st.markdown(f'<div class="stat-item"><strong>Winrate</strong><span>{round(st.session_state.pesos["winrate"], 1)}%</span></div>', unsafe_allow_html=True)
st.markdown(f'<div class="stat-item"><strong>Ops</strong><span>{st.session_state.pesos["total_ops"]}</span></div>', unsafe_allow_html=True)
st.markdown(f'<div class="stat-item"><strong>Ganadas</strong><span>{st.session_state.pesos["total_ganadas"]}</span></div>', unsafe_allow_html=True)
if modelo_info and ops_con_indicadores >= 15:
    precision = modelo_info["accuracy"] * 100
    st.markdown(f'<div class="stat-item"><strong>ML</strong><span>{precision:.0f}%</span></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="stat-item"><strong>ML</strong><span>({ops_con_indicadores}/15)</span></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

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

# Reloj
st.markdown(f'<div class="hora-box"><strong style="color:#00ff88; font-size:28px;">🕐 {hora_actual}</strong></div>', unsafe_allow_html=True)

if minuto_es_par:
    if segundo_actual < 50:
        st.markdown('<div class="signal-box signal-wait">ESCANEANDO MERCADO...</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-row"><div class="info-item"><strong>ANÁLISIS EN</strong><span>{50 - segundo_actual}s</span></div></div>', unsafe_allow_html=True)
        time_module.sleep(1)
        st.rerun()
    elif segundo_actual < 55:
        if st.session_state.estado != 'analizando':
            st.session_state.estado = 'analizando'
            st.session_state.senal = None
        if st.session_state.senal is None:
            st.session_state.senal = analizar_par(codigo, tipo)
        st.markdown('<div class="signal-box signal-wait">ANALIZANDO IA...</div>', unsafe_allow_html=True)
        time_module.sleep(0.5)
        st.rerun()
    else:
        if st.session_state.estado != 'senal':
            st.session_state.estado = 'senal'
            if st.session_state.senal is None:
                st.session_state.senal = analizar_par(codigo, tipo)
                
        señal = st.session_state.senal
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
        
        # Info compacta tipo Super Profit Sniper
        st.markdown('<div class="info-row">', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><strong>ENTRADA EXACTA</strong><span>-:00</span></div>', unsafe_allow_html=True)
        precio_señal = señal.get('precio', 0)
        st.markdown(f'<div class="info-item"><strong>PUNTO ENTRADA</strong><span>{round(precio_señal, 5) if precio_señal else "--"}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-row">', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><strong>EXPIRACIÓN</strong><span>1 MIN</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><strong>CONFIANZA</strong><span>{señal["confianza"]:.0f}%</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Radar con countdown
        st.markdown(f'<div class="radar-container"><div class="radar {clase_radar}"><div class="radar-number {clase_numero}">{segundos_restantes}</div></div></div>', unsafe_allow_html=True)
        time_module.sleep(0.3)
        st.rerun()
else:
    if st.session_state.estado != 'operando':
        st.session_state.estado = 'operando'
        st.session_state.precio_real_entrada = obtener_precio_actual(codigo, tipo)
    
    señal = st.session_state.senal
    if señal is None:
        señal = analizar_par(codigo, tipo)
        st.session_state.senal = señal
    
    es_call = señal['señal'] == 'CALL'
    segundos_restantes_op = 59 - segundo_actual
    tipo_signal = "CALL" if es_call else "PUT"
    clase_box = "signal-call" if es_call else "signal-put"
    clase_radar = "radar-green" if es_call else "radar-red"
    clase_numero = "radar-number-green" if es_call else "radar-number-red"
    
    precio_entrada = st.session_state.precio_real_entrada
    
    st.markdown(f'<div class="signal-box {clase_box}">{tipo_signal}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-row">', unsafe_allow_html=True)
    st.markdown(f'<div class="info-item"><strong>ENTRADA REAL</strong><span>{round(precio_entrada, 5) if precio_entrada else "--"}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-item"><strong>CONFIANZA</strong><span>{señal["confianza"]:.0f}%</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="radar-container"><div class="radar {clase_radar}"><div class="radar-number {clase_numero}">{segundos_restantes_op}</div></div></div>', unsafe_allow_html=True)
    st.warning("⚠️ No cierres esta pagina", icon="⚠️")
    
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

st.markdown('</div>', unsafe_allow_html=True)

# Historial compacto
st.markdown("---")
st.markdown("### 📊 Historial")
hist = cargar_historial()
if hist:
    for op in reversed(hist[-5:]):
        emoji = "✅" if op['resultado'] == 'GANADA' else "❌"
        st.write(f"{emoji} {op['fecha']} - {op['señal']} - {op['resultado']}")
else:
    st.info("Memoria limpia.")

st.markdown(f'<div class="footer">ZT TRADERS | {st.session_state.usuario_actual}</div>', unsafe_allow_html=True)
