import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import time
import json
import os
from datetime import datetime

st.set_page_config(page_title="ZT TRADERS", layout="centered", page_icon="🎯")

ARCHIVO_HISTORIAL = "historial.json"
ARCHIVO_PATRONES = "patrones.json"

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

def aprender(señal, resultado):
    hist = cargar_historial()
    pesos = cargar_patrones()
    registro = {'fecha': datetime.now().strftime('%H:%M:%S'), 'señal': señal, 'resultado': resultado}
    hist.append(registro)
    pesos['total_ops'] += 1
    if resultado == 'GANADA':
        pesos['total_ganadas'] += 1
    pesos['winrate'] = (pesos['total_ganadas'] / pesos['total_ops'] * 100) if pesos['total_ops'] > 0 else 0.0
    guardar_historial(hist)
    guardar_patrones(pesos)
    return pesos

def analizar_par(codigo_par):
    try:
        datos = yf.download(codigo_par, period="2d", interval="1m", progress=False)
        if isinstance(datos.columns, pd.MultiIndex):
            datos.columns = datos.columns.get_level_values(0)
        datos = datos[['Open', 'High', 'Low', 'Close']].copy()
        if len(datos) < 50:
            return None
        
        # Indicadores Técnicos Avanzados
        datos['RSI'] = ta.momentum.RSIIndicator(datos['Close'], window=14).rsi()
        bb = ta.volatility.BollingerBands(datos['Close'], window=20, window_dev=2)
        datos['BB_High'] = bb.bollinger_hband()
        datos['BB_Low'] = bb.bollinger_lband()
        datos['SMA_10'] = datos['Close'].rolling(window=10).mean()
        datos['EMA_50'] = ta.trend.EMAIndicator(datos['Close'], window=50).ema_indicator()
        
        ultimo = datos.iloc[-1]
        anterior = datos.iloc[-2]
        anterior2 = datos.iloc[-3]
        
        rsi = 50.0 if pd.isna(ultimo['RSI']) else float(ultimo['RSI'])
        precio = float(ultimo['Close'])
        bb_high = float(ultimo['BB_High'])
        bb_low = float(ultimo['BB_Low'])
        sma = float(ultimo['SMA_10'])
        ema_50 = float(ultimo['EMA_50'])
        
        # Filtro base por Tendencia (EMA 50) y RSI
        tendencia_alcista = precio > ema_50
        señal = 'CALL' if (rsi < 45 and tendencia_alcista) else ('PUT' if (rsi > 55 and not tendencia_alcista) else ('CALL' if rsi < 50 else 'PUT'))
        razon = "RSI bajo con tendencia alcista" if rsi < 50 else "RSI alto con tendencia bajista"

        # Validaciones de Bandas de Bollinger
        if precio <= bb_low:
            señal, razon = 'CALL', "Reote en Banda Inferior (Bollinger)"
        elif precio >= bb_high:
            señal, razon = 'PUT', "Rebote en Banda Superior (Bollinger)"

        # Cruces de SMA
        if precio > sma and anterior['Close'] <= anterior['SMA_10'] and tendencia_alcista:
            señal, razon = 'CALL', "Cruce alcista sobre SMA 10"
        elif precio < sma and anterior['Close'] >= anterior['SMA_10'] and not tendencia_alcista:
            señal, razon = 'PUT', "Cruce bajista bajo SMA 10"

        # Momentum de Velas
        if ultimo['Close'] > anterior['Close'] > anterior2['Close'] and tendencia_alcista and señal != 'PUT':
            señal, razon = 'CALL', "Fuerte Momentum Alcista"
        elif ultimo['Close'] < anterior['Close'] < anterior2['Close'] and not tendencia_alcista and señal != 'CALL':
            señal, razon = 'PUT', "Fuerte Momentum Bajista"

        return {'señal': señal, 'precio': precio, 'rsi': rsi, 'razon': razon}
    except Exception as e:
        return None

PARES = {'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X', 'USD/JPY': 'USDJPY=X', 'AUD/USD': 'AUDUSD=X', 'USD/CAD': 'USDCAD=X', 'BTC/USD': 'BTC-USD'}

if 'estado' not in st.session_state:
    st.session_state.estado = 'analizando'
if 'par_sel' not in st.session_state:
    st.session_state.par_sel = 'EUR/USD'
if 'senal' not in st.session_state:
    st.session_state.senal = None
if 'minuto_operacion' not in st.session_state:
    st.session_state.minuto_operacion = -1
if 'pesos' not in st.session_state:
    st.session_state.pesos = cargar_patrones()

CSS = """
<style>
body { background: #000; color: #fff; font-family: Arial, sans-serif; }
.main-header { background: linear-gradient(90deg, #1a1a2e, #16213e); padding: 20px; border-radius: 10px; border: 2px solid #ffd700; text-align: center; margin-bottom: 20px; }
.main-header h1 { color: #ffd700; margin: 0; font-size: 32px; letter-spacing: 3px; }
.main-header p { color: #aaa; margin: 8px 0 0 0; font-size: 13px; }
.signal-box { padding: 35px; border-radius: 15px; text-align: center; font-size: 36px; font-weight: bold; margin: 20px 0; letter-spacing: 2px; }
.signal-call { background: linear-gradient(135deg, #00cc00, #00ff00); border: 4px solid #00ff00; color: white; box-shadow: 0 0 40px rgba(0, 255, 0, 0.6); }
.signal-put { background: linear-gradient(135deg, #cc0000, #ff0000); border: 4px solid #ff0000; color: white; box-shadow: 0 0 40px rgba(255, 0, 0, 0.6); }
.signal-wait { background: #1a1a2e; border: 3px solid #ffd700; color: #ffd700; }
.radar { width: 240px; height: 240px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 25px auto; }
.radar-green { border: 4px solid #00ff00; background: radial-gradient(circle, #0a2a1a, #000); box-shadow: 0 0 40px rgba(0, 255, 0, 0.6); }
.radar-red { border: 4px solid #ff0000; background: radial-gradient(circle, #2a0a0a, #000); box-shadow: 0 0 40px rgba(255, 0, 0, 0.6); }
.radar-yellow { border: 4px solid #ffd700; background: radial-gradient(circle, #2a2a1a, #000); box-shadow: 0 0 40px rgba(255, 215, 0, 0.6); }
.radar-number { font-size: 90px; font-weight: bold; text-shadow: 0 0 25px currentColor; }
.radar-number-green { color: #00ff00; }
.radar-number-red { color: #ff0000; }
.radar-number-yellow { color: #ffd700; font-size: 24px; text-align: center; }
.info-panel { background: #1a1a2e; border: 2px solid #00ff88; border-radius: 10px; padding: 15px; margin: 12px 0; }
.info-panel strong { color: #00ff88; }
.stat-box { background: #1a1a2e; border: 2px solid #ffd700; border-radius: 10px; padding: 12px; text-align: center; }
.footer { text-align: center; color: #666; font-size: 11px; margin-top: 25px; padding: 15px; border-top: 1px solid #333; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>ZT TRADERS</h1><p>SINCRONIZADO CON RELOJ | CICLO AUTOMATICO INTELIGENTE</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    par_nombre = st.selectbox("Par:", list(PARES.keys()), index=list(PARES.keys()).index(st.session_state.par_sel))
    st.session_state.par_sel = par_nombre
with col2:
    hora_actual = datetime.now().strftime('%H:%M:%S')
    st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">HORA</strong><br><span style="color:#00ff88; font-size:22px;">{hora_actual}</span></div>', unsafe_allow_html=True)

codigo = PARES[par_nombre]
c1, c2, c3 = st.columns(3)
with c1: st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">Winrate</strong><br><span style="color:#00ff88; font-size:22px;">{round(st.session_state.pesos["winrate"], 1)}%</span></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">Operaciones</strong><br><span style="color:#00ff88; font-size:22px;">{st.session_state.pesos["total_ops"]}</span></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">Ganadas</strong><br><span style="color:#00ff88; font-size:22px;">{st.session_state.pesos["total_ganadas"]}</span></div>', unsafe_allow_html=True)
st.markdown("---")

ahora = datetime.now()
minuto_actual = ahora.minute
segundo_actual = ahora.second

# ==========================================
# FASE 1: ANALIZANDO (Segundos 0 a 51)
# ==========================================
if st.session_state.estado == 'analizando':
    st.markdown('<div class="signal-box signal-wait">ESCANEANDO MERCADO...</div>', unsafe_allow_html=True)
    st.markdown('<div class="radar radar-yellow"><div class="radar-number radar-number-yellow">ANALIZANDO<br>IA ACTIVA</div></div>', unsafe_allow_html=True)
    
    proximo_minuto = (minuto_actual + 1) % 60
    st.markdown(f'<div class="info-panel"><strong>Par:</strong> {par_nombre}<br><strong>Estado:</strong> Analizando tendencia y velas de 1m<br><strong>Proxima señal a las:</strong> XX:{proximo_minuto:02d}:52</div>', unsafe_allow_html=True)
    
    if segundo_actual >= 52:
        resultado = analizar_par(codigo)
        if resultado:
            st.session_state.senal = resultado
            st.session_state.estado = 'senal'
            st.rerun()
    
    time.sleep(0.5)
    st.rerun()

# ==========================================
# FASE 2: SEÑAL (Segundos 52 a 59 - 8 Segundos)
# ==========================================
elif st.session_state.estado == 'senal':
    señal = st.session_state.senal
    es_call = señal['señal'] == 'CALL'
    
    countdown = 60 - segundo_actual
    
    tipo = "ENTRAR AHORA! (CALL)" if es_call else "ENTRAR AHORA! (PUT)"
    clase_box = "signal-call" if es_call else "signal-put"
    clase_radar = "radar-green" if es_call else "radar-red"
    clase_numero = "radar-number-green" if es_call else "radar-number-red"
    
    st.markdown(f'<div class="signal-box {clase_box}">{tipo}</div>', unsafe_allow_html=True)
    
    hora_entrada = ahora.strftime('%H:%M') + ":00"
    st.markdown(f'<div class="info-panel"><strong>Entrada Exacta:</strong> {hora_entrada}<br><strong>Punto:</strong> {round(señal["precio"], 5)}<br><strong>Estrategia/Razon:</strong> {señal["razon"]}</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="radar {clase_radar}"><div class="radar-number {clase_numero}">{countdown}</div></div><p style="text-align:center; color:#ffd700; font-size:18px;">SEGUNDOS PARA ENTRAR</p>', unsafe_allow_html=True)
    
    if segundo_actual == 0:
        st.session_state.estado = 'operando'
        st.session_state.minuto_operacion = minuto_actual
        st.rerun()
    
    time.sleep(0.2)
    st.rerun()

# ==========================================
# FASE 3: OPERACIÓN (60 Segundos exactos)
# ==========================================
elif st.session_state.estado == 'operando':
    señal = st.session_state.senal
    es_call = señal['señal'] == 'CALL'
    
    countdown = 60 - segundo_actual
    
    tipo = "CALL" if es_call else "PUT"
    clase_box = "signal-call" if es_call else "signal-put"
    clase_radar = "radar-green" if es_call else "radar-red"
    clase_numero = "radar-number-green" if es_call else "radar-number-red"
    
    st.markdown('<div class="info-panel" style="text-align:center;"><strong>OPERACION EN CURSO (EXPIRACION)</strong></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="signal-box {clase_box}">{tipo}</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="info-panel"><strong>Entrada:</strong> {round(señal["precio"], 5)}<br><strong>Expiracion:</strong> 1 Minuto exacto</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="radar {clase_radar}"><div class="radar-number {clase_numero}">{countdown}</div></div><p style="text-align:center; color:#ffaa00; font-size:18px;">SEGUNDOS RESTANTES</p>', unsafe_allow_html=True)
    st.warning("⚠️ Mantén esta pestaña abierta mientras corre el tiempo.")
    
    if minuto_actual != st.session_state.minuto_operacion:
        st.session_state.estado = 'cerrada'
        st.rerun()
    
    time.sleep(0.5)
    st.rerun()

# ==========================================
# FASE 4: CERRADA (Registrar resultado)
# ==========================================
elif st.session_state.estado == 'cerrada':
    st.markdown('<div class="info-panel" style="text-align:center; border-color:#ffaa00;"><strong>OPERACION CERRADA</strong><br><small>Registra el resultado para actualizar estadísticas de la IA</small></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ GANE"):
            st.session_state.pesos = aprender(st.session_state.senal['señal'], 'GANADA')
            st.session_state.estado = 'analizando'
            st.session_state.senal = None
            st.success("¡Registrado!")
            time.sleep(1)
            st.rerun()
    with col2:
        if st.button("❌ PERDI"):
            st.session_state.pesos = aprender(st.session_state.senal['señal'], 'PERDIDA')
            st.session_state.estado = 'analizando'
            st.session_state.senal = None
            st.error("Registrado")
            time.sleep(1)
            st.rerun()
    with col3:
        if st.button("⏭️ NO LA JUGUE"):
            st.session_state.estado = 'analizando'
            st.session_state.senal = None
            st.info("Saltando...")
            time.sleep(1)
            st.rerun()

st.markdown("---")
st.markdown("### 📊 Historial Reciente")
hist = cargar_historial()
if hist:
    for op in reversed(hist[-10:]):
        color_res = "🟢" if op['resultado'] == 'GANADA' else "🔴"
        st.write(f"{color_res} **{op['fecha']}** | Señal: **{op['señal']}** | Resultado: **{op['resultado']}**")
else:
    st.info("Aún no hay operaciones registradas en el historial.")

st.markdown('<div class="footer">ZT TRADERS | Ciclo Automático Sincronizado v2.0</div>', unsafe_allow_html=True)
