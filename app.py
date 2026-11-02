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
            if key_input == "ADMIN-MASTER-2026":
                st.session_state.licencia_activa = True
                st.session_state.usuario_actual = "👑 Admin Master"
                st.query_params["zt_token"] = "admin_master_device"
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
            df = pd.DataFrame(res['values']).iloc[::-1].reset_index(drop=True)
            for c in ['open', 'high', 'low', 'close']: df[c] = pd.to_numeric(df[c])
            return df
        return None
    except: return None

def obtener_precio_actual(codigo_par, tipo='crypto', df_actual=None):
    if df_actual is not None and len(df_actual) > 0:
        return float(df_actual['close'].iloc[-1])
    if tipo == 'crypto':
        try: return float(requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={codigo_par}", timeout=3).json()['price'])
        except: return None
    else:
        try:
            df = obtener_datos_twelvedata(codigo_par, limit=1)
            if df is not None and len(df) > 0: return float(df['close'].iloc[-1])
        except: pass
    return None

# ==========================================
# 🧠 MOTOR DE APRENDIZAJE CON DETECCIÓN DE VELAS + CONFLUENCIA
# ==========================================
def analizar_par(codigo_par, tipo='crypto', td_symbol=None):
    try:
        df_1m = obtener_datos_binance(codigo_par, limit=60) if tipo == 'crypto' else obtener_datos_twelvedata(td_symbol, limit=60)
        df_5m = obtener_datos_binance(codigo_par, interval='5m', limit=20) if tipo == 'crypto' else obtener_datos_twelvedata(td_symbol, limit=20)
        
        if df_1m is None or len(df_1m) < 30 or df_5m is None or len(df_5m) < 5: 
            return {'señal': 'CALL', 'confianza': 50, 'razon': "Sincronizando datos...", 'indicadores': {}, 'df': None}
        
        c = 'close'
        # INDICADORES TÉCNICOS
        df_1m['RSI'] = ta.momentum.RSIIndicator(df_1m[c], window=14).rsi()
        df_1m['STOCH'] = ta.momentum.StochasticOscillator(df_1m['high'], df_1m['low'], df_1m[c]).stoch()
        macd = ta.trend.MACD(df_1m[c])
        df_1m['MACD_diff'] = macd.macd_diff()
        df_1m['MACD_signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df_1m[c], window=20, window_dev=2)
        df_1m['BB_high'] = bb.bollinger_hband()
        df_1m['BB_low'] = bb.bollinger_lband()
        
        df_5m['SMA_20'] = df_5m[c].rolling(window=20).mean()
        
        u1 = df_1m.iloc[-1]
        u2 = df_1m.iloc[-2]
        
        precio, open_p = float(u1[c]), float(u1['open'])
        open_prev, close_prev = float(u2['open']), float(u2['close'])
        
        rsi = float(u1['RSI']) if not pd.isna(u1['RSI']) else 50
        stoch = float(u1['STOCH']) if not pd.isna(u1['STOCH']) else 50
        macd_diff = float(u1['MACD_diff']) if not pd.isna(u1['MACD_diff']) else 0
        macd_sig = float(u1['MACD_signal']) if not pd.isna(u1['MACD_signal']) else 0
        bb_high, bb_low = float(u1['BB_high']), float(u1['BB_low'])
        sma_5m = float(df_5m['SMA_20'].iloc[-1])
        
        bb_pos = (precio - bb_low) / (bb_high - bb_low) if (bb_high - bb_low) > 0 else 0.5
        tendencia_5m = 1 if precio > sma_5m else -1
        
        # DETECCIÓN DE VELAS (PRICE ACTION)
        cuerpo_actual = abs(precio - open_p)
        mecha_inf = min(open_p, precio) - float(u1['low'])
        mecha_sup = float(u1['high']) - max(open_p, precio)
        
        es_martillo = (mecha_inf > cuerpo_actual * 2) and (cuerpo_actual > 0) and (precio > open_p)
        es_estrella_fugaz = (mecha_sup > cuerpo_actual * 2) and (cuerpo_actual > 0) and (precio < open_p)
        es_envolvente_alcista = (open_prev > close_prev) and (precio > open_prev) and (open_p < close_prev)
        es_envolvente_bajista = (open_prev < close_prev) and (precio < open_prev) and (open_p > close_prev)
        
        # LÓGICA BASE: Siempre da una dirección (CALL o PUT)
        if rsi < 50 or precio <= bb_low or tendencia_5m == 1:
            direccion = 'CALL'
        else:
            direccion = 'PUT'
        
        # CONFIANZA BASE
        confianza = 50
        razones = []
        
        # CONFLUENCIA (Aumenta confianza)
        if direccion == 'CALL':
            if tendencia_5m == 1: confianza += 10; razones.append("Tendencia 5m+")
            if rsi < 40: confianza += 15; razones.append("RSI sobreventa")
            if precio <= bb_low: confianza += 15; razones.append("Banda inferior")
            if macd_diff > macd_sig: confianza += 10; razones.append("MACD+")
            if stoch < 30: confianza += 5; razones.append("Stoch bajo")
            
            # DETECCIÓN DE VELAS (BONUS)
            if es_martillo: confianza += 15; razones.append("🔨 Martillo")
            if es_envolvente_alcista: confianza += 15; razones.append("📈 Envolvente+")
        else:
            if tendencia_5m == -1: confianza += 10; razones.append("Tendencia 5m-")
            if rsi > 60: confianza += 15; razones.append("RSI sobrecompra")
            if precio >= bb_high: confianza += 15; razones.append("Banda superior")
            if macd_diff < macd_sig: confianza += 10; razones.append("MACD-")
            if stoch > 70: confianza += 5; razones.append("Stoch alto")
            
            # DETECCIÓN DE VELAS (BONUS)
            if es_estrella_fugaz: confianza += 15; razones.append("⭐ Estrella Fugaz")
            if es_envolvente_bajista: confianza += 15; razones.append("📉 Envolvente-")
        
        if not razones:
            razones.append("Señal básica")
        
        # APRENDIZAJE: Ajustar confianza basado en historial
        hist = cargar_historial()
        if len(hist) >= 10:
            # Calcular winrate de las últimas 20 operaciones
            ultimas_20 = hist[-20:]
            ganadas_20 = sum(1 for op in ultimas_20 if op['resultado'] == 'GANADA')
            winrate_reciente = (ganadas_20 / len(ultimas_20)) * 100
            
            # Si el winrate es bueno, mantener confianza. Si es malo, reducir ligeramente
            if winrate_reciente < 40:
                confianza = max(45, confianza - 5)  # Reducir si estamos perdiendo mucho
            elif winrate_reciente > 60:
                confianza = min(95, confianza + 5)  # Aumentar si estamos ganando
        
        indicadores = {
            'rsi': rsi, 'stoch': stoch, 'macd_diff': macd_diff, 
            'bb_pos': bb_pos, 'tendencia_5m': tendencia_5m,
            'es_martillo': es_martillo, 'es_estrella_fugaz': es_estrella_fugaz,
            'es_envolvente_alcista': es_envolvente_alcista, 'es_envolvente_bajista': es_envolvente_bajista
        }
        
        return {'señal': direccion, 'confianza': min(confianza, 95), 'razon': " | ".join(razones), 'indicadores': indicadores, 'df': df_1m}
        
    except Exception as e:
        return {'señal': 'CALL', 'confianza': 50, 'razon': "Señal básica", 'indicadores': {}, 'df': None}

def reproducir_voz(mensaje):
    m = mensaje.replace("'", "\\'").replace('"', '\\"')
    st.markdown(f"<script>if('speechSynthesis' in window){{window.speechSynthesis.cancel();const msg=new SpeechSynthesisUtterance('{m}');msg.lang='es-ES';msg.volume=1.0;window.speechSynthesis.speak(msg);}}</script>", unsafe_allow_html=True)

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
st.markdown(f'<div class="term-header"><div class="term-title">⚡ ZT TRADERS PRO</div><div class="term-sub">OPERADOR: {st.session_state.usuario_actual} | IA APRENDIZAJE ACTIVO</div></div>', unsafe_allow_html=True)

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
codigo, tipo, td_symbol = codigo_info['codigo'], codigo_info['tipo'], codigo_info['td']

tv_symbol = f"BINANCE:{codigo}" if tipo == 'crypto' else f"FX:{codigo.replace('=X', '')}"
widget_html = f"""
<div class="tradingview-widget-container" style="height:190px;width:100%">
  <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {{"width": "100%", "height": "190", "symbol": "{tv_symbol}", "interval": "1", "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "es", "enable_publishing": false, "hide_top_toolbar": true, "hide_legend": true, "save_image": false, "calendar": false, "hide_volume": true, "support_host": "https://www.tradingview.com"}}
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
    <div class="metric-card"><small>IA</small><span>{len(hist)} ops</span></div>
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
        <div class="signal-banner signal-wait">📊 ANALIZANDO PRICE ACTION...</div>
        <div class="meta-panel">⏱️ Escaneo en: <strong>{55 - segundo_actual}s</strong></div>
        <div class="radar-box"><div class="radar-ring ring-yellow"><div class="radar-content"><div class="radar-num num-yellow">IA ACTIVA</div><small style="font-size:8px;color:#94a3b8;letter-spacing:1px;">ESCANEO</small></div></div></div>
        """)
        pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
        time_module.sleep(1)
        st.rerun()
    else:
        if st.session_state.senal is None or st.session_state.estado != 'senal':
            res_analisis = analizar_par(codigo, tipo, td_symbol)
            st.session_state.df_cache = res_analisis.get('df')
            st.session_state.senal = res_analisis
            st.session_state.estado = 'senal'
        
        señal = st.session_state.senal
        es_call = señal['señal'] == 'CALL'
        segundos_restantes = 59 - segundo_actual
        
        if not st.session_state.voz_reproducida:
            reproducir_voz(f"Señal de {'compra' if es_call else 'venta'}. Confianza {int(señal['confianza'])} por ciento.")
            st.session_state.voz_reproducida = True
        
        c_box = "signal-call" if es_call else "signal-put"
        c_radar = "ring-green" if es_call else "ring-red"
        c_val = "num-green" if es_call else "num-red"
        t_sig = "🟢 COMPRA (CALL)" if es_call else "🔴 VENTA (PUT)"
        
        html_dinamico += textwrap.dedent(f"""
        <div class="signal-banner {c_box}">{t_sig}</div>
        <div class="meta-panel">Confianza: <strong>{señal["confianza"]:.0f}%</strong> | {señal["razon"]}</div>
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
    if señal is None:
        señal = {'señal': 'CALL', 'confianza': 50, 'razon': "Esperando"}
    
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
        <div class="meta-panel">Precio: <strong>{round(st.session_state.precio_real_entrada, 5)}</strong></div>
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
            
            hist = cargar_historial()
            pesos = cargar_patrones()
            hist.append({'fecha': datetime.now().strftime('%H:%M:%S'), 'señal': señal['señal'], 'resultado': resultado_interno, 'precio_entrada_real': st.session_state.precio_real_entrada, 'indicadores': señal.get('indicadores', {})})
            pesos['total_ops'] += 1
            if resultado_interno == 'GANADA': pesos['total_ganadas'] += 1
            pesos['winrate'] = (pesos['total_ganadas'] / pesos['total_ops'] * 100) if pesos['total_ops'] > 0 else 0.0
            
            with open(ARCHIVO_HISTORIAL, 'w') as f: json.dump(hist, f, indent=2)
            with open(ARCHIVO_PATRONES, 'w') as f: json.dump(pesos, f, indent=2)
            st.session_state.pesos = pesos
            
        st.session_state.resultado_guardado = True
    time_module.sleep(0.5)
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
