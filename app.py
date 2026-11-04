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
            if key_input == "ADMIN-MASTER-2027":
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

# ==========================================
# 🧠 ML CON RESET AUTOMÁTICO SI WINRATE < 40%
# ==========================================
def entrenar_modelo_ml():
    try:
        hist = cargar_historial()
        if len(hist) < 10: return None
        
        # AUTO-RESET: Si el winrate es muy bajo, borrar todo
        ganadas = sum(1 for op in hist if op['resultado'] == 'GANADA')
        winrate_actual = (ganadas / len(hist)) * 100 if len(hist) > 0 else 0
        
        if len(hist) >= 20 and winrate_actual < 40:
            resetear_memoria()
            return None
        
        X, y = [], []
        for op in hist:
            ind = op.get('indicadores', {})
            features = [
                ind.get('rsi', 50), ind.get('stoch', 50), ind.get('macd_diff', 0),
                ind.get('bb_pos', 0.5), ind.get('tendencia_5m', 0),
                ind.get('momentum_1m', 0), ind.get('momentum_3m', 0),
                1 if ind.get('es_martillo', False) else 0,
                1 if ind.get('es_estrella_fugaz', False) else 0,
                1 if ind.get('es_envolvente_alcista', False) else 0,
                1 if ind.get('es_envolvente_bajista', False) else 0,
                1 if op['señal'] == 'CALL' else 0
            ]
            X.append(features)
            y.append(1 if op['resultado'] == 'GANADA' else 0)
        modelo = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, min_samples_split=5)
        modelo.fit(X, y)
        return modelo
    except: return None

def predecir_con_ml(modelo, indicadores, direccion):
    try:
        features = [
            indicadores.get('rsi', 50), indicadores.get('stoch', 50),
            indicadores.get('macd_diff', 0), indicadores.get('bb_pos', 0.5),
            indicadores.get('tendencia_5m', 0),
            indicadores.get('momentum_1m', 0), indicadores.get('momentum_3m', 0),
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
# 🎯 LÓGICA DE FRANCOTIRADOR (Solo con confirmación real)
# ==========================================
def analizar_par(codigo_par, tipo='crypto', td_symbol=None):
    try:
        df_1m = obtener_datos_binance(codigo_par, limit=60) if tipo == 'crypto' else obtener_datos_twelvedata(td_symbol, limit=60)
        df_5m = obtener_datos_binance(codigo_par, interval='5m', limit=20) if tipo == 'crypto' else obtener_datos_twelvedata(td_symbol, limit=20)
        
        if df_1m is None or len(df_1m) < 30 or df_5m is None or len(df_5m) < 5: 
            return {'señal': 'ESPERAR', 'confianza': 0, 'razon': "Datos insuficientes", 'indicadores': {}, 'df': None}
        
        c = 'close'
        df_1m['RSI'] = ta.momentum.RSIIndicator(df_1m[c], window=14).rsi()
        df_1m['STOCH'] = ta.momentum.StochasticOscillator(df_1m['high'], df_1m['low'], df_1m[c]).stoch()
        macd = ta.trend.MACD(df_1m[c])
        df_1m['MACD_diff'] = macd.macd_diff()
        df_1m['MACD_signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df_1m[c], window=20, window_dev=2)
        df_1m['BB_high'] = bb.bollinger_hband()
        df_1m['BB_low'] = bb.bollinger_lband()
        df_1m['SMA_10'] = df_1m[c].rolling(window=10).mean()
        
        df_5m['SMA_20'] = df_5m[c].rolling(window=20).mean()
        
        # Calcular momentum (cambio de precio en últimas velas)
        if len(df_1m) >= 4:
            momentum_1m = float(df_1m[c].iloc[-1] - df_1m[c].iloc[-2])  # Cambio última vela
            momentum_3m = float(df_1m[c].iloc[-1] - df_1m[c].iloc[-4])  # Cambio últimas 3 velas
        else:
            momentum_1m = 0
            momentum_3m = 0
        
        u1 = df_1m.iloc[-1]
        u2 = df_1m.iloc[-2]
        u3 = df_1m.iloc[-3] if len(df_1m) >= 3 else u2
        
        precio, open_p = float(u1[c]), float(u1['open'])
        open_prev, close_prev = float(u2['open']), float(u2['close'])
        precio_anterior = float(u2[c])
        
        rsi = float(u1['RSI']) if not pd.isna(u1['RSI']) else 50
        stoch = float(u1['STOCH']) if not pd.isna(u1['STOCH']) else 50
        macd_diff = float(u1['MACD_diff']) if not pd.isna(u1['MACD_diff']) else 0
        macd_sig = float(u1['MACD_signal']) if not pd.isna(u1['MACD_signal']) else 0
        bb_high, bb_low = float(u1['BB_high']), float(u1['BB_low'])
        sma_5m = float(df_5m['SMA_20'].iloc[-1])
        sma_1m = float(u1['SMA_10'])
        
        bb_pos = (precio - bb_low) / (bb_high - bb_low) if (bb_high - bb_low) > 0 else 0.5
        tendencia_5m = 1 if precio > sma_5m else -1
        
        # Detectar velas
        cuerpo_actual = abs(precio - open_p)
        mecha_inf = min(open_p, precio) - float(u1['low'])
        mecha_sup = float(u1['high']) - max(open_p, precio)
        
        es_martillo = (mecha_inf > cuerpo_actual * 2) and (cuerpo_actual > 0) and (precio > open_p)
        es_estrella_fugaz = (mecha_sup > cuerpo_actual * 2) and (cuerpo_actual > 0) and (precio < open_p)
        es_envolvente_alcista = (open_prev > close_prev) and (precio > open_prev) and (open_p < close_prev)
        es_envolvente_bajista = (open_prev < close_prev) and (precio < open_prev) and (open_p > close_prev)
        
        # ==========================================
        # 🎯 NUEVA LÓGICA: Solo operar con CONFIRMACIÓN REAL
        # ==========================================
        
        # Para CALL: Necesitamos ver que el precio ESTÁ SUBIENDO ahora
        # No basta con RSI bajo, necesitamos momentum positivo
        
        call_score = 0
        put_score = 0
        
        # Momentum positivo (precio subiendo en últimas velas)
        if momentum_1m > 0:
            call_score += 20
        elif momentum_1m < 0:
            put_score += 20
        
        if momentum_3m > 0:
            call_score += 15
        elif momentum_3m < 0:
            put_score += 15
        
        # Tendencia de 5m
        if tendencia_5m == 1:
            call_score += 15
        else:
            put_score += 15
        
        # RSI en zona de reversión + momentum
        if rsi < 35 and momentum_1m > 0:  # RSI muy bajo + empezando a subir
            call_score += 25
        elif rsi > 65 and momentum_1m < 0:  # RSI muy alto + empezando a bajar
            put_score += 25
        
        # Bandas de Bollinger + momentum
        if precio <= bb_low and momentum_1m > 0:
            call_score += 20
        elif precio >= bb_high and momentum_1m < 0:
            put_score += 20
        
        # MACD cruzando + momentum
        if macd_diff > macd_sig and momentum_1m > 0:
            call_score += 15
        elif macd_diff < macd_sig and momentum_1m < 0:
            put_score += 15
        
        # Velas de reversión
        if es_martillo and momentum_1m > 0:
            call_score += 20
        if es_envolvente_alcista and momentum_1m > 0:
            call_score += 20
        if es_estrella_fugaz and momentum_1m < 0:
            put_score += 20
        if es_envolvente_bajista and momentum_1m < 0:
            put_score += 20
        
        # Decisión final
        if call_score > put_score and call_score >= 60:
            direccion = 'CALL'
            confianza = min(call_score, 95)
        elif put_score > call_score and put_score >= 60:
            direccion = 'PUT'
            confianza = min(put_score, 95)
        else:
            return {'señal': 'ESPERAR', 'confianza': 0, 'razon': "Sin confirmación clara", 'indicadores': {}, 'df': None}
        
        razones = []
        if direccion == 'CALL':
            if momentum_1m > 0: razones.append("Momentum +")
            if tendencia_5m == 1: razones.append("Tendencia 5m+")
            if rsi < 35: razones.append("RSI sobreventa")
            if precio <= bb_low: razones.append("Banda inferior")
            if macd_diff > macd_sig: razones.append("MACD+")
            if es_martillo: razones.append("🔨 Martillo")
            if es_envolvente_alcista: razones.append("📈 Envolvente+")
        else:
            if momentum_1m < 0: razones.append("Momentum -")
            if tendencia_5m == -1: razones.append("Tendencia 5m-")
            if rsi > 65: razones.append("RSI sobrecompra")
            if precio >= bb_high: razones.append("Banda superior")
            if macd_diff < macd_sig: razones.append("MACD-")
            if es_estrella_fugaz: razones.append("⭐ Estrella Fugaz")
            if es_envolvente_bajista: razones.append("📉 Envolvente-")
        
        indicadores = {
            'rsi': rsi, 'stoch': stoch, 'macd_diff': macd_diff, 
            'bb_pos': bb_pos, 'tendencia_5m': tendencia_5m,
            'momentum_1m': momentum_1m, 'momentum_3m': momentum_3m,
            'es_martillo': es_martillo, 'es_estrella_fugaz': es_estrella_fugaz,
            'es_envolvente_alcista': es_envolvente_alcista, 'es_envolvente_bajista': es_envolvente_bajista
        }
        
        # ML boost
        modelo_ml = entrenar_modelo_ml()
        if modelo_ml is not None:
            prob_ml = predecir_con_ml(modelo_ml, indicadores, direccion)
            if prob_ml is not None and prob_ml > 55:
                confianza = int((confianza + prob_ml) / 2)
                razones.append(f"🧠 IA: {prob_ml}%")
        
        return {'señal': direccion, 'confianza': max(60, min(confianza, 95)), 'razon': " | ".join(razones), 'indicadores': indicadores, 'df': df_1m}
        
    except Exception as e:
        return {'señal': 'ESPERAR', 'confianza': 0, 'razon': f"Error: {str(e)[:20]}", 'indicadores': {}, 'df': None}

# ==========================================
# 🔊 SISTEMA DE VOZ
# ==========================================
def reproducir_voz(mensaje):
    m = mensaje.replace("'", "\\'").replace('"', '\\"')
    html_voz = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="margin:0;padding:10px;background:#1a1a2e;font-family:Arial;">
        <div style="color:#00ff88;font-size:14px;font-weight:bold;text-align:center;">
            🔊 {mensaje}
        </div>
        <script>
        (function() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                setTimeout(function() {{
                    const msg = new SpeechSynthesisUtterance('{m}');
                    msg.lang = 'es-ES';
                    msg.rate = 0.9;
                    msg.pitch = 1.2;
                    msg.volume = 1.0;
                    window.speechSynthesis.speak(msg);
                }}, 300);
            }}
        }})();
        </script>
    </body>
    </html>
    """
    components.html(html_voz, height=80, width=150)

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
if 'voz_activada' not in st.session_state: st.session_state.voz_activada = False

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
.confianza-display { background: linear-gradient(135deg, #1e3a8a, #0f172a); border: 2px solid #3b82f6; border-radius: 8px; padding: 8px; margin: 6px 0; text-align: center; }
.confianza-display strong { color: #60a5fa; font-size: 14px; }
.confianza-display span { color: #fff; font-size: 20px; font-weight: 900; text-shadow: 0 0 10px rgba(59, 130, 246, 0.8); }
.ml-status { background: rgba(15, 23, 42, 0.9); border: 1px solid #a855f7; border-radius: 6px; padding: 5px; font-size: 10px; text-align: center; margin: 4px 0; }
.ml-status small { color: #a855f7; font-size: 8px; font-weight: 800; display: block; text-transform: uppercase; }
.ml-status span { color: #d8b4fe; font-size: 12px; font-weight: 900; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="terminal-container">', unsafe_allow_html=True)
st.markdown(f'<div class="term-header"><div class="term-title">⚡ ZT TRADERS PRO</div><div class="term-sub">OPERADOR: {st.session_state.usuario_actual} | FRANCOTIRADOR v2.0</div></div>', unsafe_allow_html=True)

if not st.session_state.voz_activada:
    if st.button("🔊 ACTIVAR VOZ (TOCAR 1 VEZ)", key="btn_activar_voz", use_container_width=True):
        reproducir_voz("Sistema de voz activado. Z T Traders listo.")
        st.session_state.voz_activada = True
        st.rerun()

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
modelo_ml = entrenar_modelo_ml()
ml_status = "🟢 APRENDIENDO" if modelo_ml else f"🟡 Recopilando ({len(hist)}/10)"

st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card"><small>Winrate</small><span>{winrate}%</span></div>
    <div class="metric-card"><small>Ops</small><span>{ops}</span></div>
    <div class="metric-card"><small>Win</small><span>{ganadas}</span></div>
    <div class="metric-card"><small>IA</small><span>{ml_status}</span></div>
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
        <div class="signal-banner signal-wait">📊 ANALIZANDO CONFLUENCIA...</div>
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
        
        if señal['señal'] == 'ESPERAR':
            html_dinamico += textwrap.dedent(f"""
            <div class="signal-banner signal-wait">⏳ SIN CONFLUENCIA</div>
            <div class="meta-panel">Esperando confirmación real...</div>
            <div class="radar-box"><div class="radar-ring ring-yellow"><div class="radar-content"><div class="radar-num num-yellow">ESPERA</div><small style="font-size:8px;color:#94a3b8;letter-spacing:1px;">SIN SEÑAL</small></div></div></div>
            """)
            pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
            time_module.sleep(0.5)
            st.rerun()
        
        es_call = señal['señal'] == 'CALL'
        segundos_restantes = 59 - segundo_actual
        confianza_actual = int(señal['confianza'])
        
        if not st.session_state.voz_reproducida and st.session_state.voz_activada:
            if confianza_actual >= 70:
                tipo_texto = "COMPRA" if es_call else "VENTA"
                mensaje = f"Atención. Señal confirmada. {tipo_texto}. {confianza_actual} por ciento. Entrar ahora."
                reproducir_voz(mensaje)
            st.session_state.voz_reproducida = True
        
        c_box = "signal-call" if es_call else "signal-put"
        c_radar = "ring-green" if es_call else "ring-red"
        c_val = "num-green" if es_call else "num-red"
        t_sig = "🟢 COMPRA (CALL)" if es_call else "🔴 VENTA (PUT)"
        
        if confianza_actual >= 70:
            color_confianza = "#00ff88"
            icono_alerta = "🔊"
        else:
            color_confianza = "#fbbf24"
            icono_alerta = "🔇"
        
        html_dinamico += textwrap.dedent(f"""
        <div class="signal-banner {c_box}">{t_sig}</div>
        <div class="confianza-display">
            <strong>{icono_alerta} CONFIANZA CONFIRMADA</strong><br>
            <span style="color: {color_confianza}; font-size: 28px;">{confianza_actual}%</span>
        </div>
        <div class="meta-panel"><strong>Confirmación:</strong> {señal["razon"]}</div>
        <div class="radar-box"><div class="radar-ring {c_radar}"><div class="radar-content"><div class="radar-num {c_val}">{segundos_restantes}</div><small style="font-size:8px;color:#94a3b8;letter-spacing:1px;">SEGUNDOS</small></div></div></div>
        """)
        pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
        time_module.sleep(0.5)
        st.rerun()
else:
    if st.session_state.estado != 'operando':
        st.session_state.estado = 'operando'
        st.session_state.precio_real_entrada = obtener_precio_actual(codigo, tipo, df_actual=st.session_state.df_cache)
    
    señal = st.session_state.senal
    if señal is None or señal.get('señal') == 'ESPERAR':
        html_dinamico += textwrap.dedent(f"""
        <div class="signal-banner signal-wait">⏸️ SIN OPERACIÓN</div>
        <div class="radar-box"><div class="radar-ring ring-yellow"><div class="radar-content"><div class="radar-num num-yellow">--</div><small style="font-size:8px;color:#94a3b8;letter-spacing:1px;">ESPERA</small></div></div></div>
        """)
        pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
        time_module.sleep(0.5)
        st.rerun()
    
    es_call = señal['señal'] == 'CALL'
    seg_op = 59 - segundo_actual
    confianza_actual = int(señal['confianza'])
    
    c_box = "signal-call" if es_call else "signal-put"
    c_radar = "ring-green" if es_call else "ring-red"
    c_val = "num-green" if es_call else "num-red"
    t_sig = "🟢 EN CURSO (CALL)" if es_call else "🔴 EN CURSO (PUT)"
    
    if confianza_actual >= 70:
        color_confianza = "#00ff88"
        icono_alerta = "🔊"
    else:
        color_confianza = "#fbbf24"
        icono_alerta = "🔇"
    
    html_dinamico += textwrap.dedent(f"""
    <div class="signal-banner {c_box}">{t_sig}</div>
    <div class="confianza-display">
        <strong>{icono_alerta} CONFIANZA</strong><br>
        <span style="color: {color_confianza}; font-size: 24px;">{confianza_actual}%</span>
    </div>
    """)
    if st.session_state.precio_real_entrada:
        html_dinamico += textwrap.dedent(f"""
        <div class="meta-panel">Precio Entrada: <strong>{round(st.session_state.precio_real_entrada, 5)}</strong></div>
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
