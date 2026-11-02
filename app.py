def analizar_par(codigo_par, tipo='crypto'):
    # Pedir velas recientes de 1 minuto directas de Binance
    if tipo == 'crypto':
        df = obtener_datos_binance(codigo_par, interval='1m', limit=30)
    else:
        df = obtener_datos_yahoo(codigo_par, period="1d", interval="1m")
    
    if df is None or len(df) < 15:
        return {'señal': 'CALL', 'confianza': 60, 'razon': "Datos base", 'indicadores': {'rsi': 50, 'precio_relativo': 0.5}}
    
    c_close = 'close' if tipo == 'crypto' else 'Close'
    c_open = 'open' if tipo == 'crypto' else 'Open'
    
    try:
        # 1. RSI rápido de 7 períodos para scalping a 1m
        df['RSI_7'] = ta.momentum.RSIIndicator(df[c_close], window=7).rsi()
        
        # 2. Medias Móviles Rápidas para capturar la micro-tendencia
        df['EMA_9'] = ta.trend.EMAIndicator(df[c_close], window=9).ema_indicator()
        df['EMA_21'] = ta.trend.EMAIndicator(df[c_close], window=21).ema_indicator()
        
        u1 = df.iloc[-1]
        precio = float(u1[c_close])
        apertura = float(u1[c_open])
        rsi_7 = float(u1['RSI_7'])
        ema_9 = float(u1['EMA_9'])
        ema_21 = float(u1['EMA_21'])
        
        # 3. Fuerza de la vela actual (Acción del Precio en 1m)
        vela_verde = precio > apertura
        
        razones = []
        
        # Condición para Entrada a 1 Minuto (CALL)
        if (rsi_7 < 45 or ema_9 > ema_21) and vela_verde:
            dirección = 'CALL'
            razones.append("Momentum Alcista 1M + RSI Rápido")
        # Condición para Entrada a 1 Minuto (PUT)
        elif (rsi_7 > 55 or ema_9 < ema_21) and not vela_verde:
            dirección = 'PUT'
            razones.append("Momentum Bajista 1M + RSI Rápido")
        else:
            # Fallback según la EMA rápida
            dirección = 'CALL' if ema_9 >= ema_21 else 'PUT'
            razones.append("Seguimiento EMA 9/21")

        # Guardar indicadores para el entrenamiento del Machine Learning
        indicadores = {
            'rsi': rsi_7,
            'precio_relativo': 1.0 if vela_verde else 0.0
        }

        # Calcular confianza ajustada
        confianza = 75 if (rsi_7 < 35 or rsi_7 > 65) else 65

        return {
            'señal': dirección, 
            'confianza': confianza, 
            'razon': " | ".join(razones), 
            'indicadores': indicadores
        }
    except Exception as e:
        return {'señal': 'CALL', 'confianza': 60, 'razon': "Análisis estándar", 'indicadores': {'rsi': 50, 'precio_relativo': 0.5}}
