import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import ta
import time as time_module
import json
import os
import uuid
import textwrap
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ZT TRADERS INSTITUTIONAL PRO", layout="centered", page_icon="⚡")

# 🔑 TU CLAVE REAL DE TWELVE DATA (La que generaste antes)
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

if 'licencia_activa' not in st.session_state: st.session_state.licencia_activa = False
if 'usuario_actual' not in st.session_state: st.session_state.usuario_actual = ""

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
                except: pass

if not st.session_state.licencia_activa:
    st.markdown("""
    <style>
    body { background: #020208; color: #fff; font-family: 'Inter', sans-serif; }
    .login-card { max-width: 350px; margin: 50px auto; padding: 30px; background: rgba(10, 14, 28, 0.95); border-radius: 16px; border: 1.5px solid #00ff88; box-shadow: 0 0 40px rgba(0, 255, 136, 0.2); text-align: center; }
    .login-title { color: #00ff88; font-size: 24px; font-weight: 900; letter-spacing: 3px; text-shadow: 0 0 10px rgba(0,255,136,0.4); }
    .login-sub { color: #64748b; font-size: 10px; margin-bottom: 25px; text-transform: uppercase; letter-spacing: 1.5px; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="login-card"><div class="login-title">⚡ ZT TRADERS</div><div class="login-sub">Terminal Institucional Privada</div></div>', unsafe_allow_html=True)
    with st.form("licencia_form"):
        key_input = st.text_input("License Key", placeholder="Ingresa tu clave de acceso", type="password")
        submit_key = st.form_submit_button("🚀 CONECTAR TERMINAL", use_container_width=True)
        if submit_key:
            if key_input:
                if key_input == "ADMIN-MASTER-2026":
                    st.session_state.licencia_activa = True
                    st.session_state.usuario_actual = "👑 Admin Master"
                    st.query_params["zt_token"] = "admin_master_device"
                    st.success("¡Acceso Admin Concedido!")
                    st.rerun()
                else:
                    licencias = cargar_licencias()
                    if key_input in licencias:
                        datos = licencias[key_input]
                        try:
                            if datetime.now().date() > datetime.strptime(datos['vence'], '%Y-%m-%d').date():
                                st.error(f"❌ Key vencida el {datos['vence']}.")
                            else:
                                if datos['device_id'] is None:
                                    nuevo_token = str(uuid.uuid4())
                                    datos['device_id'] = nuevo_token
                                    guardar_licencias(licencias)
                                    st.query_params["zt_token"] = nuevo_token
                                    st.session_state.licencia_activa = True
                                    st.session_state.usuario_actual = f"Licencia ({key_input[:8]}...)"
                                    st.rerun()
                                else:
                                    if device_token == datos['device_id']:
                                        st.session_state.licencia_activa = True
                                        st.session_state.usuario_actual = f"Licencia ({key_input[:8]}...)"
                                        st.rerun()
                                    else:
                                        st.error("❌ Key activa en otro dispositivo.")
                        except: st.error("Error al validar fecha.")
                    else: st.error("❌ Key inválida.")
    st.stop()

def cargar_patrones():
    if os.path.exists(ARCHIVO_PATRONES):
        try:
            with open(ARCHIVO_PATRONES, 'r') as f: return json.load(f)
        except: os.remove(ARCHIVO_PATRONES)
    return {'total_ops': 0, 'total_ganadas': 0, 'winrate': 0.0}

def guardar_patrones(pesos):
    try:
        with open(ARCHIVO_PATRONES, 'w') as f: json.dump(pesos, f, indent=2)
    except: pass

def cargar_historial():
    if os.path.exists(ARCHIVO_HISTORIAL):
        try:
            with open(ARCHIVO_HISTORIAL, 'r') as f: return json.load(f)
        except: os.remove(ARCHIVO_HISTORIAL)
    return []

def guardar_historial(hist):
    try:
        with open(ARCHIVO_HISTORIAL, 'w') as f: json.dump(hist, f, indent=2)
    except: pass

def resetear_memoria():
    if os.path.exists(ARCHIVO_HISTORIAL): os.remove(ARCHIVO_HISTORIAL)
    if os.path.exists(ARCHIVO_PATRONES): os.remove(ARCHIVO_PATRONES)
    st.session_state.pesos = {'total_ops': 0, 'total_ganadas': 0, 'winrate': 0.0}

def entrenar_modelo_ml():
    try:
        hist = cargar_historial()
        if len(hist) < 10: return None
        X, y = [], []
        for op in hist:
            ind = op.get('indicadores', {})
            X.append([
                ind.get('rsi', 50), ind.get('stoch', 50), ind.get('macd_diff', 0), 
                ind.get('adx', 20), ind.get('bb_pos', 0.5), ind.get('tendencia_5m', 0), 
                1 if op['señal'] == 'CALL' else 0
            ])
            y.append(1 if op['resultado'] == 'GANADA' else 0)
        modelo = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42)
        modelo.fit(X, y)
        return modelo
    except: return None

def aprender_en_segundo_plano(señal, resultado, indicadores, precio_real_entrada):
    hist = cargar_historial()
    pesos = cargar_patrones()
    hist.append({'fecha': datetime.now().strftime('%H:%M:%S'), 'señal': señal, 'resultado': resultado, 'precio_entrada_real': precio_real_entrada, 'indicadores': indicadores})
    pesos['total_ops'] += 1
    if resultado == 'GANADA': pesos['total_ganadas'] += 1
    pesos['winrate'] = (pesos['total_ganadas'] / pesos['total_ops'] * 100) if pesos['total_ops'] > 0 else 0.0
    guardar_historial(hist)
    guardar_patrones(pesos)

def obtener_datos_binance(symbol, interval='1m', limit=50):
    try:
        url = "https://api.binance.com/api/v3/klines"
        res = requests.get(url, params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=5).json()
        df = pd.DataFrame(res, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        for c in ['open', 'high', 'low', 'close', 'volume']: df[c] = pd.to_numeric(df[c])
        return df
    except: return None

def obtener_datos_twelvedata(symbol, limit=50):
    try:
        # Twelve Data usa el formato EUR/USD, BTC/USD, etc.
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize={limit}&apikey={TWELVE_DATA_API_KEY}"
        res = requests.get(url, timeout=5).json()
        if 'values' in res:
            df = pd.DataFrame(res['values'])
            df = df.iloc[::-1].reset_index(drop=True) # Invertir para que el más reciente esté al final
            for c in ['open', 'high', 'low', 'close']:
                df[c] = pd.to_numeric(df[c])
            return df
        return None
    except: return None

def obtener_precio_actual(codigo_par, tipo='crypto', df_actual=None):
    if df_actual is not None and len(df_actual) > 0:
        return float(df_actual['close'].iloc[-1])
    
    if tipo == 'crypto':
        try:
            return float(requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={codigo_par}", timeout=3).json()['price'])
        except: return None
    else:
        try:
            df = obtener_datos_twelvedata(codigo_par, limit=1)
            if df is not None and len(df) > 0:
                return float(df['close'].iloc[-1])
        except: pass
        return None

def analizar_par(codigo_par, tipo='crypto', td_symbol=None):
    try:
        if tipo == 'crypto':
            df = obtener_datos_binance(codigo_par, limit=60)
        else:
            # Usar Twelve Data para Forex (sincronizado con IQ Option)
            df = obtener_datos_twelvedata(td_symbol, limit=60)
            
        if df is None or len(df) < 30: 
            return {'señal': 'ESPERAR', 'confianza': 0, 'razon': "Sincronizando datos...", 'indicadores': {}, 'df': None}
        
        c_close, c_open, c_high, c_low = 'close', 'open', 'high', 'low'
        
        # INDICADORES ROBUSTOS (RSI 14)
        df['RSI'] = ta.momentum.RSIIndicator(df[c_close], window=14).rsi()
        df['STOCH'] = ta.momentum.StochasticOscillator(df[c_high], df[c_low], df[c_close]).stoch()
        macd = ta.trend.MACD(df[c_close])
        df['MACD_diff'] = macd.macd_diff()
        df['MACD_signal'] = macd.macd_signal()
        df['ADX'] = ta.trend.ADXIndicator(df[c_high], df[c_low], df[c_close]).adx()
        
        bollinger = ta.volatility.BollingerBands(df[c_close], window=20, window_dev=2)
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_low'] = bollinger.bollinger_lband()
        df['SMA_10'] = df[c_close].rolling(window=10).mean()
        
        # Datos de 5m para tendencia
        if tipo == 'crypto':
            df_5m = obtener_datos_binance(codigo_par, interval='5m', limit=20)
        else:
            df_5m = obtener_datos_twelvedata(td_symbol, limit=20) # Twelve Data también para 5m
            
        sma_5m = 0
        if df_5m is not None and len(df_5m) > 0:
            df_5m['SMA_20'] = df_5m[c_close].rolling(window=20).mean()
            sma_5m = float(df_5m['SMA_20'].iloc[-1])
        
        u1 = df.iloc[-1]
        precio, apertura = float(u1[c_close]), float(u1[c_open])
        rsi = float(u1['RSI']) if not pd.isna(u1['RSI']) else 50
        stoch = float(u1['STOCH']) if not pd.isna(u1['STOCH']) else 50
        macd_diff = float(u1['MACD_diff']) if not pd.isna(u1['MACD_diff']) else 0
        macd_sig = float(u1['MACD_signal']) if not pd.isna(u1['MACD_signal']) else 0
        adx = float(u1['ADX']) if not pd.isna(u1['ADX']) else 20
        bb_high, bb_low = float(u1['BB_high']), float(u1['BB_low'])
        sma_10 = float(u1['SMA_10'])
        
        bb_pos = (precio - bb_low) / (bb_high - bb_low) if (bb_high - bb_low) > 0 else 0.5
        vela_verde = precio >= apertura
        tendencia_5m = 1 if precio > sma_5m else -1
        
        razones = []
        if rsi < 45 and macd_diff > macd_sig and vela_verde:
            dirección = 'CALL'
            razones.append("Rebote Alcista Confirmado")
        elif rsi > 55 and macd_diff < macd_sig and not vela_verde:
            dirección = 'PUT'
            razones.append("Rebote Bajista Confirmado")
        else:
            dirección = 'CALL' if (macd_diff >= 0 or rsi < 50) else 'PUT'
            razones.append("Tendencia General")

        confianza = 50
        if dirección == 'CALL':
            if rsi < 40: confianza += 15
            if precio <= bb_low: confianza += 15
            if macd_diff > macd_sig: confianza += 10
            if stoch < 30: confianza += 10
        else:
            if rsi > 60: confianza += 15
            if precio >= bb_high: confianza += 15
            if macd_diff < macd_sig: confianza += 10
            if stoch > 70: confianza += 10
            
        if adx > 25: confianza += 10

        modelo_ml = entrenar_modelo_ml()
        indicadores = {
            'rsi': rsi, 'stoch': stoch, 'macd_diff': macd_diff, 
            'adx': adx, 'bb_pos': bb_pos, 'tendencia_5m': tendencia_5m
        }
        
        if modelo_ml is not None:
            try:
                features = [rsi, stoch, macd_diff, adx, bb_pos, tendencia_5m, 1 if dirección == 'CALL' else 0]
                pred = modelo_ml.predict([features])
                prob = modelo_ml.predict_proba([features])
                confianza_ml = int(prob[0][pred[0]] * 100)
                if confianza_ml > 55:
                    confianza = int((confianza + confianza_ml) / 2)
                    razones.append(f"IA: {confianza_ml}%")
            except: pass

        if confianza < 55:
            return {'señal': 'ESPERAR', 'confianza': confianza, 'razon': "Condiciones no óptimas", 'indicadores': indicadores, 'df': df}

        return {'señal': dirección, 'confianza': min(confianza, 95), 'razon': " | ".join(razones), 'indicadores': indicadores, 'df': df}
    except Exception as e:
        return {'señal': 'ESPERAR', 'confianza': 0, 'razon': f"Error: {str(e)[:20]}", 'indicadores': {}, 'df': None}

def reproducir_voz(mensaje):
    m = mensaje.replace("'", "\\'").replace('"', '\\"')
    st.markdown(f"<script>if('speechSynthesis' in window){{window.speechSynthesis.cancel();const msg=new SpeechSynthesisUtterance('{m}');msg.lang='es-ES';msg.volume=1.0;window.speechSynthesis.speak(msg);}}</script>", unsafe_allow_html=True)

# MAPEO DE PARES: Nombre visible : {'codigo_binance', 'tipo', 'twelvedata_symbol'}
PARES = {
    'Bitcoin': {'codigo': 'BTCUSDT', 'tipo': 'crypto', 'td': 'BTC/USD'},
    'Ethereum': {'codigo': 'ETHUSDT', 'tipo': 'crypto', 'td': 'ETH/USD'},
    'Solana': {'codigo': 'SOLUSDT', 'tipo': 'crypto', 'td': 'SOL/USD'},
    'Polkadot': {'codigo': 'DOTUSDT', 'tipo': 'crypto', 'td': 'DOT/USD'},
    'Worldcoin': {'codigo': 'WLDUSDT', 'tipo': 'crypto', 'td': 'WLD/USD'},
    'Binance Coin': {'codigo': 'BNBUSDT', 'tipo': 'crypto', 'td': 'BNB/USD'},
    'Cardano': {'codigo': 'ADAUSDT', 'tipo': 'crypto', 'td': 'ADA/USD'},
    'Ripple': {'codigo': 'XRPUSDT', 'tipo': 'crypto', 'td': 'XRP/USD'},
    'Dogecoin': {'codigo': 'DOGEUSDT', 'tipo': 'crypto', 'td': 'DOGE/USD'},
    'Polygon': {'codigo': 'MATICUSDT', 'tipo': 'crypto', 'td': 'MATIC/USD'},
    'Litecoin': {'codigo': 'LTCUSDT', 'tipo': 'crypto', 'td': 'LTC/USD'},
    'EUR/USD': {'codigo': 'EURUSD=X', 'tipo': 'forex', 'td': 'EUR/USD'},
    'GBP/USD': {'codigo': 'GBPUSD=X', 'tipo': 'forex', 'td': 'GBP/USD'},
    'USD/JPY': {'codigo': 'USDJPY=X', 'tipo': 'forex', 'td': 'USD/JPY'},
    'AUD/USD': {'codigo': 'AUDUSD=X', 'tipo': 'forex', 'td': 'AUD/USD'},
    'EUR/JPY': {'codigo': 'EURJPY=X', 'tipo': 'forex', 'td': 'EUR/JPY'},
    'AUD/CAD': {'codigo': 'AUDCAD=X', 'tipo': 'forex', 'td': 'AUD/CAD'},
    'EUR/GBP': {'codigo': 'EURGBP=X', 'tipo': 'forex', 'td': 'EUR/GBP'},
    'GBP/JPY': {'codigo': 'GBPJPY=X', 'tipo': 'forex', 'td': 'GBP/JPY'},
    'NZD/USD': {'codigo': 'NZDUSD=X', 'tipo': 'forex', 'td': 'NZD/USD'},
    'USD/CAD': {'codigo': 'USDCAD=X', 'tipo': 'forex', 'td': 'USD/CAD'},
    'USD/CHF': {'codigo': 'USDCHF=X', 'tipo': 'forex', 'td': 'USD/CHF'},
    'AUD/JPY': {'codigo': 'AUDJPY=X', 'tipo': 'forex', 'td': 'AUD/JPY'}
}

if 'estado' not in st.session_state: st.session_state.estado = 'analizando'
if 'par_sel' not in st.session_state: st.session_state.par_sel = 'Bitcoin'
if 'senal' not in st.session_state: st.session_state.senal = None
if 'pesos' not in st.session_state: st.session_state.pesos = cargar_patrones()
if 'resultado_guardado' not in st.session_state: st.session_state.resultado_guardado = False
if 'minuto_anterior' not in st.session_state: st.session_state.minuto_anterior = None
if 'precio_real_entrada' not in st.session_state: st.session_state.precio_real_entrada = None
if 'voz_reproducida' not in st.session_state: st.session_state.voz_reproducida = False
if 'df_cache' not in st.session_state: st.session_state.df_cache = None

CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 4px !important; padding-bottom: 0px !important; padding-left: 6px !important; padding-right: 6px !important; max-width: 100% !important; }
body { background: #02040a; color: #f1f5f9; font-family: 'Inter', -apple-system, sans-serif; }
.terminal-container { width: 100%; max-width: 430px; margin: 0 auto; display: flex; flex-direction: column; gap: 4px; }
.term-header { background: linear-gradient(135deg, #090f24 0%, #030712 100%); border: 1.5px solid rgba(0, 255, 136, 0.4); border-radius: 12px; padding: 8px; text-align: center; box-shadow: 0 4px 25px rgba(0,0,0,0.9); }
.term-title { color: #00ff88; font-size: 16px; font-weight: 900; letter-spacing: 3px; margin: 0; text-shadow: 0 0 12px rgba(0,255,136,0.5); }
.term-sub { color: #94a3b8; font-size: 9px; font-weight: 600; letter-spacing: 1px; margin-top: 2px; text-transform: uppercase; }
.metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; margin: 4px 0; }
.metric-card { background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(0, 255, 136, 0.25); border-radius: 8px; padding: 5px 2px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.5); }
.metric-card small { color: #64748b; font-size: 8px; font-weight: 800; display: block; text-transform: uppercase; }
.metric-card span { color: #00ff88; font-size: 13px; font-weight: 900; text-shadow: 0 0 8px rgba(0,255,136,0.4); }
.clock-display { background: rgba(3, 30, 18, 0.9); border: 1px solid #00ff88; border-radius: 8px; padding: 4px; text-align: center; box-shadow: 0 0 15px rgba(0,255,136,0.2); }
.clock-display span { color: #00ff88; font-size: 17px; font-weight: 900; text-shadow: 0 0 10px rgba(0,255,136,0.6); font-family: monospace; letter-spacing: 1px; }
.signal-banner { padding: 9px; border-radius: 10px; text-align: center; font-size: 21px; font-weight: 900; letter-spacing: 2px; margin: 4px 0; text-transform: uppercase; }
.signal-call { background: linear-gradient(135deg, #065f46, #059669); border: 1.5px solid #34d399; color: #fff; box-shadow: 0 0 25px rgba(52,211,153,0.5); }
.signal-put { background: linear-gradient(135deg, #991b1b, #dc2626); border: 1.5px solid #f87171; color: #fff; box-shadow: 0 0 25px rgba(248,113,113,0.5); }
.signal-wait { background: rgba(30, 41, 59, 0.9); border: 1.5px solid #fbbf24; color: #fbbf24; font-size: 13px; }
.radar-box { position: relative; width: 165px; height: 165px; margin: 8px auto; }
.radar-ring { width: 165px; height: 165px; border-radius: 50%; border: 3.5px solid; position: relative; }
.ring-green { border-color: #34d399; background: radial-gradient(circle, rgba(5,150,105,0.35) 0%, #02040a 75%); box-shadow: 0 0 30px rgba(52,211,153,0.5); }
.ring-red { border-color: #f87171; background: radial-gradient(circle, rgba(220,38,38,0.35) 0%, #02040a 75%); box-shadow: 0 0 30px rgba(248,113,113,0.5); }
.ring-yellow { border-color: #fbbf24; background: radial-gradient(circle, rgba(251,191,36,0.35) 0%, #02040a 75%); box-shadow: 0 0 30px rgba(251,191,36,0.5); }
.radar-content { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; }
.radar-num { font-size: 48px; font-weight: 900; line-height: 1; font-family: monospace; }
.num-green { color: #34d399; text-shadow: 0 0 15px rgba(52,211,153,0.8); }
.num-red { color: #f87171; text-shadow: 0 0 15px rgba(248,113,113,0.8); }
.num-yellow { color: #fbbf24; font-size: 16px; letter-spacing: 1px; }
.meta-panel { background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(0, 255, 136, 0.3); border-radius: 6px; padding: 5px; font-size: 10px; text-align: center; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="terminal-container">', unsafe_allow_html=True)
st.markdown(f'<div class="term-header"><div class="term-title">⚡ ZT TRADERS PRO</div><div class="term-sub">OPERADOR: {st.session_state.usuario_actual} | 23 PARES EN VIVO</div></div>', unsafe_allow_html=True)

es_admin = (st.session_state.usuario_actual == "👑 Admin Master")
if es_admin:
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
else:
    par_nombre = st.selectbox("", list(PARES.keys()), index=list(PARES.keys()).index(st.session_state.par_sel), label_visibility="collapsed")
    st.session_state.par_sel = par_nombre

codigo_info = PARES[par_nombre]
codigo = codigo_info['codigo']
tipo = codigo_info['tipo']
td_symbol = codigo_info['td']

# Widget de TradingView
tv_symbol = f"BINANCE:{codigo}" if tipo == 'crypto' else f"FX:{codigo.replace('=X', '')}"
widget_html = f"""
<div class="tradingview-widget-container" style="height:190px;width:100%">
  <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {{
  "width": "100%", "height": "190", "symbol": "{tv_symbol}", "interval": "1", "timezone": "Etc/UTC",
  "theme": "dark", "style": "1", "locale": "es", "enable_publishing": false, "hide_top_toolbar": true,
  "hide_legend": true, "save_image": false, "calendar": false, "hide_volume": true, "support_host": "https://www.tradingview.com"
  }}
  </script>
</div>
"""
components.html(widget_html, height=195)

hist = cargar_historial()
ops, ganadas = st.session_state.pesos['total_ops'], st.session_state.pesos['total_ganadas']
winrate = round(st.session_state.pesos['winrate'], 1)

st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card"><small>Winrate</small><span>{winrate}%</span></div>
    <div class="metric-card"><small>Ops</small><span>{ops}</span></div>
    <div class="metric-card"><small>Win</small><span>{ganadas}</span></div>
    <div class="metric-card"><small>ML</small><span>{len(hist)} ops</span></div>
</div>
""", unsafe_allow_html=True)

pantalla_dinamica = st.empty()
ahora = datetime.now()
hora_actual = ahora.strftime('%H:%M:%S')
minuto_actual, segundo_actual = ahora.minute, ahora.second
minuto_es_par = (minuto_actual % 2 == 0)

if st.session_state.minuto_anterior is None:
    st.session_state.minuto_anterior = minuto_actual
elif st.session_state.minuto_anterior != minuto_actual:
    st.session_state.resultado_guardado = False
    st.session_state.voz_reproducida = False
    st.session_state.minuto_anterior = minuto_actual

html_dinamico = f'<div class="clock-display"><span>🕐 {hora_actual}</span></div>'

if minuto_es_par:
    if segundo_actual < 55:
        html_dinamico += textwrap.dedent(f"""
        <div class="signal-banner signal-wait">📊 ANALIZANDO TENDENCIA...</div>
        <div class="meta-panel">⏱️ Sincronización IA en: <strong>{55 - segundo_actual}s</strong></div>
        <div class="radar-box"><div class="radar-ring ring-yellow"><div class="radar-content"><div class="radar-num num-yellow">IA ACTIVA</div><small style="font-size:8px;color:#94a3b8;letter-spacing:1px;">ESCANEO</small></div></div></div>
        """)
        pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
        time_module.sleep(1)
        st.rerun()
    else:
        if st.session_state.senal is None or st.session_state.estado != 'senal':
            res_analisis = analizar_par(codigo, tipo, td_symbol)
            st.session_state.df_cache = res_analisis.get('df')
            
            if res_analisis['señal'] == 'ESPERAR' or res_analisis['confianza'] < 55:
                st.session_state.senal = {'señal': 'ESPERAR', 'confianza': res_analisis['confianza'], 'razon': "Filtro estricto: Buscando alta probabilidad", 'indicadores': res_analisis['indicadores']}
            else:
                st.session_state.senal = res_analisis
            st.session_state.estado = 'senal'
        
        señal = st.session_state.senal
        if señal['señal'] == 'ESPERAR':
            html_dinamico += textwrap.dedent(f"""
            <div class="signal-banner signal-wait">⏳ FILTRANDO RUIDO DE MERCADO</div>
            <div class="meta-panel">Confianza: <strong>{señal["confianza"]}%</strong> | {señal["razon"]}</div>
            <div class="radar-box"><div class="radar-ring ring-yellow"><div class="radar-content"><div class="radar-num num-yellow">ESPERA</div><small style="font-size:8px;color:#94a3b8;letter-spacing:1px;">BAJA CERTEZA</small></div></div></div>
            """)
            pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
            time_module.sleep(0.5)
            st.rerun()
        
        es_call = señal['señal'] == 'CALL'
        segundos_restantes = 59 - segundo_actual
        
        if not st.session_state.voz_reproducida:
            reproducir_voz(f"Atención. Señal de {'compra' if es_call else 'venta'}. Confianza {int(señal['confianza'])} por ciento.")
            st.session_state.voz_reproducida = True
        
        c_box = "signal-call" if es_call else "signal-put"
        c_radar = "ring-green" if es_call else "ring-red"
        c_val = "num-green" if es_call else "num-red"
        t_sig = "🟢 COMPRA (CALL)" if es_call else "🔴 VENTA (PUT)"
        
        html_dinamico += textwrap.dedent(f"""
        <div class="signal-banner {c_box}">{t_sig}</div>
        <div class="meta-panel">Confianza IA: <strong>{señal["confianza"]:.0f}%</strong> | <strong>{señal["razon"]}</strong></div>
        <div class="radar-box"><div class="radar-ring {c_radar}"><div class="radar-content"><div class="radar-num {c_val}">{segundos_restantes}</div><small style="font-size:8px;color:#94a3b8;letter-spacing:1px;">SEGUNDOS</small></div></div></div>
        """)
        pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
        time_module.sleep(0.3)
        st.rerun()
else:
    if st.session_state.estado != 'operando':
        st.session_state.estado = 'operando'
        st.session_state.precio_real_entrada = obtener_precio_actual(codigo, tipo, df_actual=st.session_state.df_cache)
    
    señal = st.session_state.senal
    if señal is None or señal.get('señal') == 'ESPERAR':
        html_dinamico += textwrap.dedent(f"""
        <div class="signal-banner signal-wait">⏸️ SIN OPERACIÓN EN ESTE CICLO</div>
        <div class="radar-box"><div class="radar-ring ring-yellow"><div class="radar-content"><div class="radar-num num-yellow">--</div><small style="font-size:8px;color:#94a3b8;letter-spacing:1px;">ESPERA</small></div></div></div>
        """)
        pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
        time_module.sleep(0.5)
        st.rerun()
    
    es_call = señal['señal'] == 'CALL'
    seg_op = 59 - segundo_actual
    
    c_box = "signal-call" if es_call else "signal-put"
    c_radar = "ring-green" if es_call else "ring-red"
    c_val = "num-green" if es_call else "num-red"
    t_sig = "🟢 EN CURSO (CALL)" if es_call else "🔴 EN CURSO (PUT)"
    
    html_dinamico += textwrap.dedent(f"""
    <div class="signal-banner {c_box}">{t_sig}</div>
    """)
    if st.session_state.precio_real_entrada:
        html_dinamico += textwrap.dedent(f"""
        <div class="meta-panel">Precio Sincronizado: <strong>{round(st.session_state.precio_real_entrada, 5)}</strong></div>
        """)
    html_dinamico += textwrap.dedent(f"""
    <div class="radar-box"><div class="radar-ring {c_radar}"><div class="radar-content"><div class="radar-num {c_val}">{seg_op}</div><small style="font-size:8px;color:#94a3b8;letter-spacing:1px;">EXPIRACIÓN</small></div></div></div>
    """)
    
    pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
    
    if segundo_actual >= 58 and not st.session_state.resultado_guardado:
        if st.session_state.precio_real_entrada and st.session_state.precio_real_entrada > 0:
            precio_actual = obtener_precio_actual(codigo, tipo, df_actual=st.session_state.df_cache)
            resultado_interno = 'PERDIDA'
            if precio_actual is not None:
                if (es_call and precio_actual > st.session_state.precio_real_entrada) or (not es_call and precio_actual < st.session_state.precio_real_entrada):
                    resultado_interno = 'GANADA'
            aprender_en_segundo_plano(señal['señal'], resultado_interno, señal.get('indicadores', {}), st.session_state.precio_real_entrada)
            st.session_state.pesos = cargar_patrones()
        st.session_state.resultado_guardado = True
    time_module.sleep(0.5)
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
