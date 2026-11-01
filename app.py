import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import ta
import time as time_module
import json
import os
from datetime import datetime

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
    body { background: #000; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 0; }
    .login-container { max-width: 320px; margin: 40px auto; padding: 20px; background: linear-gradient(135deg, #0a0a1a, #1a0a2e); border-radius: 12px; border: 2px solid #ffd700; box-shadow: 0 0 30px rgba(255, 215, 0, 0.4); }
    .login-title { color: #ffd700; text-align: center; font-size: 22px; font-weight: bold; margin-bottom: 5px; letter-spacing: 2px; }
    .login-subtitle { color: #aaa; text-align: center; font-size: 11px; margin-bottom: 20px; }
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
                st.warning("Completa todos los campos")
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
    except:
        return None

def obtener_datos_yahoo(symbol, period="5d", interval="1m"):
    try:
        datos = yf.download(symbol, period=period, interval=interval, progress=False)
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
        df = obtener_datos_binance(codigo_par, interval='1m', limit=100)
    else:
        df = obtener_datos_yahoo(codigo_par, period="2d", interval="1m")
    
    if df is None or len(df) < 30:
        return {'señal': 'CALL', 'confianza': 50, 'razon': "Datos insuficientes", 'indicadores': {}}
    
    # Normalizar nombres de columnas según la fuente
    c_close = 'close' if tipo == 'crypto' else 'Close'
    c_high = 'high' if tipo == 'crypto' else 'High'
    c_low = 'low' if tipo == 'crypto' else 'Low'
    
    try:
        df['RSI'] = ta.momentum.RSIIndicator(df[c_close], window=14).rsi()
        bb = ta.volatility.BollingerBands(df[c_close], window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['EMA_20'] = ta.trend.EMAIndicator(df[c_close], window=20).ema_indicator()
        
        u1 = df.iloc[-1]
        precio = float(u1[c_close])
        rsi = float(u1['RSI'])
        bb_high = float(u1['BB_High'])
        bb_low = float(u1['BB_Low'])
        ema_20 = float(u1['EMA_20'])
        
        # Filtros estrictos de calidad para mejorar el Winrate
        razones = []
        confianza = 50
        
        if precio <= bb_low and rsi < 35:
            dirección = 'CALL'
            confianza = 85
            razones.append("Toque Banda Inferior + RSI Sobrevendido")
        elif precio >= bb_high and rsi > 65:
            dirección = 'PUT'
            confianza = 85
            razones.append("Toque Banda Superior + RSI Sobrecomprado")
        elif precio > ema_20 and rsi < 48:
            dirección = 'CALL'
            confianza = 70
            razones.append("Tendencia Alcista (EMA) + Pullback RSI")
        elif precio < ema_20 and rsi > 52:
            dirección = 'PUT'
            confianza = 70
            razones.append("Tendencia Bajista (EMA) + Pullback RSI")
        else:
            # Fallback seguro basado en tendencia general
            dirección = 'CALL' if precio > ema_20 else 'PUT'
            confianza = 60
            razones.append("Seguimiento de Tendencia EMA")

        indicadores = {'rsi': rsi, 'precio': precio, 'direccion': dirección}
        return {'señal': dirección, 'confianza': confianza, 'razon': " | ".join(razones), 'indicadores': indicadores}
    except Exception as e:
        return {'señal': 'CALL', 'confianza': 50, 'razon': "Error de cálculo técnico", 'indicadores': {}}

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
    'EUR/USD': {'codigo': 'EURUSD=X', 'tipo': 'forex'},
    'GBP/USD': {'codigo': 'GBPUSD=X', 'tipo': 'forex'},
    'USD/JPY': {'codigo': 'USDJPY=X', 'tipo': 'forex'},
    'AUD/USD': {'codigo': 'AUDUSD=X', 'tipo': 'forex'},
    'USD/CAD': {'codigo': 'USDCAD=X', 'tipo': 'forex'},
    'Bitcoin': {'codigo': 'BTCUSDT', 'tipo': 'crypto'},
    'Ethereum': {'codigo': 'ETHUSDT', 'tipo': 'crypto'},
    'Solana': {'codigo': 'SOLUSDT', 'tipo': 'crypto'}
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

/* Radar Único Corregido (Elimina el doble círculo desfasado) */
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

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown(f'<div class="header"><h1>⚡ ZT TRADERS</h1><p>User: {st.session_state.usuario_actual} | IA OPTIMIZADA</p></div>', unsafe_allow_html=True)

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
    <div class="stat-box"><strong>ML</strong><span style="font-size:10px;">{len(hist)}/20</span></div>
</div>
""", unsafe_allow_html=True)

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

if minuto_es_par:
    if segundo_actual < 55:
        st.markdown('<div class="signal-box signal-wait">📊 ESCANEANDO TENDENCIA...</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-panel">⏱️ Próxima señal en: <strong>{55 - segundo_actual}s</strong></div>', unsafe_allow_html=True)
        st.markdown('<div class="radar-container"><div class="radar radar-yellow"><div class="radar-center"><div class="radar-number radar-number-yellow">IA</div><div class="radar-label">ESPERA</div></div></div></div>', unsafe_allow_html=True)
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
        st.markdown(f'<div class="info-panel">Confianza: <strong>{señal["confianza"]:.0f}%</strong> | <strong>{señal["razon"]}</strong></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="radar-container"><div class="radar {clase_radar}"><div class="radar-center"><div class="radar-number {clase_numero}">{segundos_restantes}</div><div class="radar-label">SEGUNDOS</div></div></div></div>', unsafe_allow_html=True)
        time_module.sleep(0.3)
        st.rerun()
else:
    if st.session_state.estado != 'operando':
        st.session_state.estado = 'operando'
        st.session_state.precio_real_entrada = obtener_precio_actual(codigo, tipo)
    
    señal = st.session_state.senal
    if señal is None:
        señal = {'señal': 'CALL', 'confianza': 50, 'razon': "Esperando"}
    
    es_call = señal['señal'] == 'CALL'
    segundos_restantes_op = 59 - segundo_actual
    
    clase_box = "signal-call" if es_call else "signal-put"
    clase_radar = "radar-green" if es_call else "radar-red"
    clase_numero = "radar-number-green" if es_call else "radar-number-red"
    tipo_signal = "🟢 EN CURSO (CALL)" if es_call else "🔴 EN CURSO (PUT)"
    
    st.markdown(f'<div class="signal-box {clase_box}">{tipo_signal}</div>', unsafe_allow_html=True)
    if st.session_state.precio_real_entrada:
        st.markdown(f'<div class="info-panel">Entrada: <strong>{round(st.session_state.precio_real_entrada, 5)}</strong></div>', unsafe_allow_html=True)
    
    # Único cronómetro activo y limpio para evitar duplicidad visual
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
            aprender_en_segundo_plano(señal['señal'], resultado_interno, {}, st.session_state.precio_real_entrada)
            st.session_state.pesos = cargar_patrones()
        st.session_state.resultado_guardado = True
    time_module.sleep(0.5)
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
