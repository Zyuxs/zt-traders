import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import ta
import time as time_module
import json
import os
import uuid
import textwrap
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ZT TRADERS PRO", layout="centered", page_icon="⚡")

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
    body { background: #05050f; color: #fff; font-family: 'Segoe UI', Roboto, sans-serif; }
    .login-card { max-width: 330px; margin: 40px auto; padding: 25px; background: rgba(15, 20, 35, 0.9); border-radius: 16px; border: 1.5px solid #ffd700; box-shadow: 0 0 30px rgba(255, 215, 0, 0.25); text-align: center; }
    .login-title { color: #ffd700; font-size: 24px; font-weight: 800; letter-spacing: 2px; }
    .login-sub { color: #8892b0; font-size: 11px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-card"><div class="login-title">⚡ ZT TRADERS</div><div class="login-sub">Terminal de Alta Frecuencia</div></div>', unsafe_allow_html=True)
    
    with st.form("licencia_form"):
        key_input = st.text_input("License Key", placeholder="Ingresa tu Key", type="password")
        submit_key = st.form_submit_button("🚀 INICIAR SESIÓN", use_container_width=True)
        
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
                            fecha_vence = datetime.strptime(datos['vence'], '%Y-%m-%d').date()
                            if datetime.now().date() > fecha_vence:
                                st.error(f"❌ La Key venció el {datos['vence']}.")
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
                                        st.error("❌ Key asignada a otro equipo.")
                        except:
                            st.error("Error al validar fecha.")
                    else:
                        st.error("❌ Key no encontrada.")
    st.stop()

# --- FUNCIONALIDADES Y ANALISIS ML ---
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
        if len(hist) < 5: return None
        X, y = [], []
        for op in hist:
            ind = op.get('indicadores', {})
            X.append([ind.get('rsi', 50), ind.get('macd_diff', 0), ind.get('bb_pos', 0.5), 1 if op['señal'] == 'CALL' else 0])
            y.append(1 if op['resultado'] == 'GANADA' else 0)
        modelo = RandomForestClassifier(n_estimators=40, max_depth=4, random_state=42)
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
        res = requests.get(url, params={'symbol': symbol, 'interval': interval, 'limit': limit}, timeout=4).json()
        df = pd.DataFrame(res, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        for c in ['open', 'high', 'low', 'close', 'volume']: df[c] = pd.to_numeric(df[c])
        return df
    except: return None

def obtener_datos_yahoo(symbol, period="1d", interval="1m"):
    try:
        datos = yf.download(symbol, period=period, interval=interval, progress=False, timeout=4)
        if datos is None or len(datos) == 0: return None
        if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
        return datos[['Open', 'High', 'Low', 'Close']].dropna().copy()
    except: return None

def obtener_precio_actual(codigo_par, tipo='crypto'):
    if tipo == 'crypto':
        try:
            return float(requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={codigo_par}", timeout=3).json()['price'])
        except: return None
    else:
        try:
            datos = yf.download(codigo_par, period="1d", interval="1m", progress=False, timeout=3)
            if isinstance(datos.columns, pd.MultiIndex): datos.columns = datos.columns.get_level_values(0)
            return float(datos['Close'].iloc[-1])
        except: return None

def analizar_par(codigo_par, tipo='crypto'):
    try:
        df = obtener_datos_binance(codigo_par, limit=50) if tipo == 'crypto' else obtener_datos_yahoo(codigo_par)
        if df is None or len(df) < 20: return {'señal': 'CALL', 'confianza': 60, 'razon': "Sin datos suficientes", 'indicadores': {'rsi': 50, 'macd_diff': 0, 'bb_pos': 0.5}}
        
        c_close = 'close' if tipo == 'crypto' else 'Close'
        c_open = 'open' if tipo == 'crypto' else 'Open'
        
        df['RSI_7'] = ta.momentum.RSIIndicator(df[c_close], window=7).rsi()
        macd = ta.trend.MACD(df[c_close])
        df['MACD_diff'] = macd.macd_diff()
        bollinger = ta.volatility.BollingerBands(df[c_close], window=20, window_dev=2)
        df['BB_high'] = bollinger.bollinger_hband()
        df['BB_low'] = bollinger.bollinger_lband()
        
        u1 = df.iloc[-1]
        precio, apertura = float(u1[c_close]), float(u1[c_open])
        rsi_7, macd_diff = float(u1['RSI_7']), float(u1['MACD_diff'])
        bb_high, bb_low = float(u1['BB_high']), float(u1['BB_low'])
        
        bb_pos = (precio - bb_low) / (bb_high - bb_low) if (bb_high - bb_low) > 0 else 0.5
        vela_verde = precio >= apertura
        razones = []
        
        if rsi_7 < 45 and macd_diff > 0 and vela_verde:
            dirección = 'CALL'
            razones.append("Rebote Alcista RSI+MACD")
        elif rsi_7 > 55 and macd_diff < 0 and not vela_verde:
            dirección = 'PUT'
            razones.append("Rebote Bajista RSI+MACD")
        else:
            dirección = 'CALL' if macd_diff >= 0 else 'PUT'
            razones.append("Tendencia MACD")

        confianza_base = 78 if (rsi_7 < 30 or rsi_7 > 70 or bb_pos < 0.1 or bb_pos > 0.9) else 68
        modelo_ml = entrenar_modelo_ml()
        indicadores = {'rsi': rsi_7, 'macd_diff': macd_diff, 'bb_pos': bb_pos}
        
        if modelo_ml is not None:
            try:
                pred = modelo_ml.predict([[indicadores['rsi'], indicadores['macd_diff'], indicadores['bb_pos'], 1 if dirección == 'CALL' else 0]])
                prob = modelo_ml.predict_proba([[indicadores['rsi'], indicadores['macd_diff'], indicadores['bb_pos'], 1 if dirección == 'CALL' else 0]])
                confianza_ml = int(prob[0][pred[0]] * 100)
                confianza = int((confianza_base + confianza_ml) / 2)
                razones.append(f"IA Pro ({confianza}%)")
            except: confianza = confianza_base
        else: confianza = confianza_base

        return {'señal': dirección, 'confianza': max(45, min(confianza, 98)), 'razon': " | ".join(razones), 'indicadores': indicadores}
    except: return {'señal': 'CALL', 'confianza': 60, 'razon': "Modo Seguro", 'indicadores': {'rsi': 50, 'macd_diff': 0, 'bb_pos': 0.5}}

def reproducir_voz(mensaje):
    m = mensaje.replace("'", "\\'").replace('"', '\\"')
    st.markdown(f"<script>if('speechSynthesis' in window){{window.speechSynthesis.cancel();const msg=new SpeechSynthesisUtterance('{m}');msg.lang='es-ES';window.speechSynthesis.speak(msg);}}</script>", unsafe_allow_html=True)

# 23 PARES
PARES = {
    'Bitcoin': {'codigo': 'BTCUSDT', 'tipo': 'crypto'}, 'Ethereum': {'codigo': 'ETHUSDT', 'tipo': 'crypto'},
    'Solana': {'codigo': 'SOLUSDT', 'tipo': 'crypto'}, 'Polkadot': {'codigo': 'DOTUSDT', 'tipo': 'crypto'},
    'Worldcoin': {'codigo': 'WLDUSDT', 'tipo': 'crypto'}, 'Binance Coin': {'codigo': 'BNBUSDT', 'tipo': 'crypto'},
    'Cardano': {'codigo': 'ADAUSDT', 'tipo': 'crypto'}, 'Ripple': {'codigo': 'XRPUSDT', 'tipo': 'crypto'},
    'Dogecoin': {'codigo': 'DOGEUSDT', 'tipo': 'crypto'}, 'Polygon': {'codigo': 'MATICUSDT', 'tipo': 'crypto'},
    'Litecoin': {'codigo': 'LTCUSDT', 'tipo': 'crypto'}, 'EUR/USD': {'codigo': 'EURUSD=X', 'tipo': 'forex'},
    'GBP/USD': {'codigo': 'GBPUSD=X', 'tipo': 'forex'}, 'USD/JPY': {'codigo': 'USDJPY=X', 'tipo': 'forex'},
    'AUD/USD': {'codigo': 'AUDUSD=X', 'tipo': 'forex'}, 'EUR/JPY': {'codigo': 'EURJPY=X', 'tipo': 'forex'},
    'AUD/CAD': {'codigo': 'AUDCAD=X', 'tipo': 'forex'}, 'EUR/GBP': {'codigo': 'EURGBP=X', 'tipo': 'forex'},
    'GBP/JPY': {'codigo': 'GBPJPY=X', 'tipo': 'forex'}, 'NZD/USD': {'codigo': 'NZDUSD=X', 'tipo': 'forex'},
    'USD/CAD': {'codigo': 'USDCAD=X', 'tipo': 'forex'}, 'USD/CHF': {'codigo': 'USDCHF=X', 'tipo': 'forex'},
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

# NUEVA HOJA DE ESTILOS CSS REVISADA Y MODERNA
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 5px !important; padding-bottom: 0px !important; padding-left: 8px !important; padding-right: 8px !important; max-width: 100% !important; }

body { background: #030409; color: #e2e8f0; font-family: 'Inter', -apple-system, sans-serif; }
.main-app { width: 100%; max-width: 420px; margin: 0 auto; }

/* CABECERA NEÓN */
.pro-header { background: linear-gradient(180deg, rgba(20,25,45,0.9) 0%, rgba(10,12,24,0.9) 100%); border: 1px solid rgba(255, 215, 0, 0.3); border-radius: 12px; padding: 10px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.8); }
.pro-title { color: #ffd700; font-size: 17px; font-weight: 900; letter-spacing: 3px; margin: 0; text-shadow: 0 0 12px rgba(255,215,0,0.4); }
.pro-sub { color: #64748b; font-size: 9px; font-weight: 600; letter-spacing: 1px; margin-top: 2px; }

/* RECUADROS DE ESTADÍSTICAS GLASS */
.grid-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px; margin: 6px 0; }
.card-stat { background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 215, 0, 0.2); border-radius: 8px; padding: 6px 2px; text-align: center; }
.card-stat small { color: #94a3b8; font-size: 8px; font-weight: 700; display: block; text-transform: uppercase; }
.card-stat span { color: #00ff88; font-size: 13px; font-weight: 800; text-shadow: 0 0 8px rgba(0,255,136,0.3); }

/* RELOJ DIGITAL */
.clock-banner { background: rgba(6, 78, 59, 0.3); border: 1px solid #00ff88; border-radius: 8px; padding: 5px; text-align: center; box-shadow: 0 0 15px rgba(0,255,136,0.15); }
.clock-banner span { color: #00ff88; font-size: 18px; font-weight: 800; text-shadow: 0 0 10px rgba(0,255,136,0.5); font-family: monospace; }

/* SEÑAL Y RADAR */
.box-signal { padding: 10px; border-radius: 10px; text-align: center; font-size: 21px; font-weight: 900; letter-spacing: 2px; margin: 6px 0; }
.signal-call { background: linear-gradient(135deg, #15803d, #22c55e); border: 1.5px solid #4ade80; color: #fff; box-shadow: 0 0 20px rgba(34,197,94,0.4); }
.signal-put { background: linear-gradient(135deg, #b91c1c, #ef4444); border: 1.5px solid #f87171; color: #fff; box-shadow: 0 0 20px rgba(239,68,68,0.4); }
.signal-wait { background: rgba(30, 41, 59, 0.8); border: 1.5px solid #ffd700; color: #ffd700; font-size: 13px; }

.radar-wrap { position: relative; width: 140px; height: 140px; margin: 8px auto; }
.radar-circle { width: 140px; height: 140px; border-radius: 50%; border: 3px solid; position: relative; }
.radar-green { border-color: #22c55e; background: radial-gradient(circle, rgba(34,197,94,0.2) 0%, transparent 70%); box-shadow: 0 0 20px rgba(34,197,94,0.3); }
.radar-red { border-color: #ef4444; background: radial-gradient(circle, rgba(239,68,68,0.2) 0%, transparent 70%); box-shadow: 0 0 20px rgba(239,68,68,0.3); }
.radar-yellow { border-color: #ffd700; background: radial-gradient(circle, rgba(255,215,0,0.2) 0%, transparent 70%); box-shadow: 0 0 20px rgba(255,215,0,0.3); }

.radar-text { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; }
.radar-val { font-size: 42px; font-weight: 900; line-height: 1; font-family: monospace; }
.val-green { color: #4ade80; text-shadow: 0 0 12px rgba(74,222,128,0.6); }
.val-red { color: #f87171; text-shadow: 0 0 12px rgba(248,113,113,0.6); }
.val-yellow { color: #ffd700; font-size: 15px; }

.panel-meta { background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(0, 255, 136, 0.3); border-radius: 6px; padding: 6px; font-size: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# CONTENEDOR SUPERIOR DE LA APLICACIÓN
st.markdown('<div class="main-app">', unsafe_allow_html=True)
st.markdown(f'<div class="pro-header"><div class="pro-title">⚡ ZT TRADERS PRO</div><div class="pro-sub">USUARIO: {st.session_state.usuario_actual} | 23 PARES ACTIVOS</div></div>', unsafe_allow_html=True)

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
codigo, tipo = codigo_info['codigo'], codigo_info['tipo']

hist = cargar_historial()
ops, ganadas = st.session_state.pesos['total_ops'], st.session_state.pesos['total_ganadas']
winrate = round(st.session_state.pesos['winrate'], 1)

st.markdown(f"""
<div class="grid-stats">
    <div class="card-stat"><small>Winrate</small><span>{winrate}%</span></div>
    <div class="card-stat"><small>Ops</small><span>{ops}</span></div>
    <div class="card-stat"><small>Win</small><span>{ganadas}</span></div>
    <div class="card-stat"><small>ML</small><span>{len(hist)} ops</span></div>
</div>
""", unsafe_allow_html=True)

# PANTALLA DINÁMICA ÚNICA (Actualización instantánea)
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

html_dinamico = f'<div class="clock-banner"><span>🕐 {hora_actual}</span></div>'

if minuto_es_par:
    if segundo_actual < 55:
        html_dinamico += textwrap.dedent(f"""
        <div class="box-signal signal-wait">📊 ESCANEANDO MERCADO...</div>
        <div class="panel-meta">⏱️ Próxima señal en: <strong>{55 - segundo_actual}s</strong></div>
        <div class="radar-wrap"><div class="radar-circle radar-yellow"><div class="radar-text"><div class="radar-val val-yellow">IA ACTIVA</div><small style="font-size:8px;color:#aaa;">ANALIZANDO</small></div></div></div>
        """)
        pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
        time_module.sleep(1)
        st.rerun()
    else:
        if st.session_state.senal is None or st.session_state.estado != 'senal':
            res_analisis = analizar_par(codigo, tipo)
            es_ciclo_4min = (minuto_actual % 4 == 0)
            
            if es_ciclo_4min or res_analisis['confianza'] >= 70:
                st.session_state.senal = res_analisis
            else:
                st.session_state.senal = {'señal': 'ESPERAR', 'confianza': res_analisis['confianza'], 'razon': "Filtro estricto: Buscando alta probabilidad", 'indicadores': res_analisis['indicadores']}
            st.session_state.estado = 'senal'
        
        señal = st.session_state.senal
        if señal['señal'] == 'ESPERAR':
            html_dinamico += textwrap.dedent(f"""
            <div class="box-signal signal-wait">⏳ ESPERANDO ALTA PROBABILIDAD</div>
            <div class="panel-meta">Confianza: <strong>{señal["confianza"]}%</strong> | {señal["razon"]}</div>
            <div class="radar-wrap"><div class="radar-circle radar-yellow"><div class="radar-text"><div class="radar-val val-yellow">FILTRANDO</div><small style="font-size:8px;color:#aaa;">ESPERA</small></div></div></div>
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
        c_radar = "radar-green" if es_call else "radar-red"
        c_val = "val-green" if es_call else "val-red"
        t_sig = "🟢 COMPRA (CALL)" if es_call else "🔴 VENTA (PUT)"
        
        html_dinamico += textwrap.dedent(f"""
        <div class="box-signal {c_box}">{t_sig}</div>
        <div class="panel-meta">Confianza IA: <strong>{señal["confianza"]:.0f}%</strong> | <strong>{señal["razon"]}</strong></div>
        <div class="radar-wrap"><div class="radar-circle {c_radar}"><div class="radar-text"><div class="radar-val {c_val}">{segundos_restantes}</div><small style="font-size:8px;color:#aaa;">SEGUNDOS</small></div></div></div>
        """)
        pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
        time_module.sleep(0.3)
        st.rerun()
else:
    if st.session_state.estado != 'operando':
        st.session_state.estado = 'operando'
        st.session_state.precio_real_entrada = obtener_precio_actual(codigo, tipo)
    
    señal = st.session_state.senal
    if señal is None or señal.get('señal') == 'ESPERAR':
        html_dinamico += textwrap.dedent(f"""
        <div class="box-signal signal-wait">⏸️ SIN OPERACIÓN EN ESTE CICLO</div>
        <div class="radar-wrap"><div class="radar-circle radar-yellow"><div class="radar-text"><div class="radar-val val-yellow">--</div><small style="font-size:8px;color:#aaa;">ESPERA</small></div></div></div>
        """)
        pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
        time_module.sleep(0.5)
        st.rerun()
    
    es_call = señal['señal'] == 'CALL'
    seg_op = 59 - segundo_actual
    
    c_box = "signal-call" if es_call else "signal-put"
    c_radar = "radar-green" if es_call else "radar-red"
    c_val = "val-green" if es_call else "val-red"
    t_sig = "🟢 EN CURSO (CALL)" if es_call else "🔴 EN CURSO (PUT)"
    
    html_dinamico += textwrap.dedent(f"""
    <div class="box-signal {c_box}">{t_sig}</div>
    """)
    if st.session_state.precio_real_entrada:
        html_dinamico += textwrap.dedent(f"""
        <div class="panel-meta">Precio Entrada: <strong>{round(st.session_state.precio_real_entrada, 5)}</strong></div>
        """)
    html_dinamico += textwrap.dedent(f"""
    <div class="radar-wrap"><div class="radar-circle {c_radar}"><div class="radar-text"><div class="radar-val {c_val}">{seg_op}</div><small style="font-size:8px;color:#aaa;">EXPIRACIÓN</small></div></div></div>
    """)
    
    pantalla_dinamica.markdown(html_dinamico, unsafe_allow_html=True)
    
    if segundo_actual >= 58 and not st.session_state.resultado_guardado:
        if st.session_state.precio_real_entrada and st.session_state.precio_real_entrada > 0:
            precio_actual = obtener_precio_actual(codigo, tipo)
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
