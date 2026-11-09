# -*- coding: utf-8 -*-
"""
razonamiento_ia.py — Capa de razonamiento con un modelo de lenguaje (LLM)
para ZT TRADERS INSTITUTIONAL PRO (v3.4).

Papel de esta capa (acordado con Jose):
  - El cerebro cuantitativo (nucleo_ia.py) decide la direccion (CALL/PUT)
    y su probabilidad. Esta capa NO sustituye eso y NUNCA veta una señal.
  - El LLM actua como un trader senior que revisa la señal: confirma,
    sube o rebaja su NIVEL, y explica el por que en una linea.
  - Tambien resume el "clima" de la sesion y analiza el historial del
    usuario (diario inteligente) cuando el Admin lo pide.

Proveedores GRATUITOS para la fase de prueba (clave propia del usuario):
  - gemini     -> Google AI Studio (aistudio.google.com), nivel gratuito.
  - groq       -> console.groq.com, nivel gratuito, muy rapido.
  - openrouter -> openrouter.ai, modelos con sufijo :free.
Todos hablan el mismo formato (chat/completions estilo OpenAI), por eso
un solo cliente sirve para los tres, y el dia que la app sea comercial
se cambia de proveedor/modelo SOLO desde los Secrets, sin tocar codigo:
  ia_proveedor = "gemini" | "groq" | "openrouter" | "personalizado"
  ia_api_key   = "..."           (obligatoria)
  ia_modelo    = "..."           (opcional: usa el del preset)
  ia_base_url  = "https://..."   (solo para "personalizado")

Si no hay clave, o la API falla, o responde tarde: TODO devuelve
None / "" y la app sigue funcionando exactamente igual sin esta capa.
Nada en este modulo lanza excepciones hacia la app.
"""

import json
import requests

PRESETS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "modelo": "gemini-3.8-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "modelo": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "modelo": "meta-llama/llama-3.3-70b-instruct:free",
    },
}

_CONFIG = {"proveedor": "gemini", "api_key": "", "modelo": "", "base_url": ""}

SISTEMA = (
    "Eres un trader profesional senior especializado en opciones binarias de "
    "1 minuto. Revisas las señales de un sistema cuantitativo. Reglas fijas: "
    "(1) NUNCA propones cambiar la direccion de la señal ni vetarla; solo "
    "ajustas su nivel de confianza (Alta/Media/Baja). "
    "(2) Piensas como los mejores traders: tendencia manda, no se compra "
    "pegado a resistencia ni se vende pegado a soporte, se desconfia de "
    "entradas tardias tras velas extendidas y de operar en rango sin "
    "confirmacion. (3) Respondes breve y directo, en español."
)


def configurar(proveedor=None, api_key=None, modelo=None, base_url=None):
    """Recibe los valores de los Secrets de Streamlit (los pasa la app)."""
    if proveedor:
        _CONFIG["proveedor"] = str(proveedor).strip().lower()
    if api_key is not None:
        _CONFIG["api_key"] = str(api_key).strip()
    if modelo:
        _CONFIG["modelo"] = str(modelo).strip()
    if base_url:
        _CONFIG["base_url"] = str(base_url).strip().rstrip("/")


def disponible():
    return bool(_CONFIG["api_key"])


def _endpoint_y_modelo():
    preset = PRESETS.get(_CONFIG["proveedor"], {})
    base = _CONFIG["base_url"] or preset.get("base_url", "")
    modelo = _CONFIG["modelo"] or preset.get("modelo", "")
    return (base.rstrip("/") + "/chat/completions") if base else "", modelo


def _chat(mensajes, max_tokens=220, timeout=14):
    """Una llamada al LLM; devuelve el texto o None ante cualquier fallo.

    Gemini usa su endpoint NATIVO (generateContent con x-goog-api-key):
    verificado con clave real; su ruta compatible-OpenAI respondia con
    timeouts. Los demas proveedores usan chat/completions estilo OpenAI.
    """
    if not disponible():
        return None
    sistema = ""
    usuario = ""
    for m in mensajes:
        if m.get("role") == "system":
            sistema = m.get("content", "")
        elif m.get("role") == "user":
            usuario = m.get("content", "")
    if _CONFIG["proveedor"] == "gemini" and not _CONFIG["base_url"]:
        modelo = _CONFIG["modelo"] or PRESETS["gemini"]["modelo"]
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent",
                headers={"x-goog-api-key": _CONFIG["api_key"],
                         "Content-Type": "application/json"},
                json={"system_instruction": {"parts": [{"text": sistema}]},
                      "contents": [{"role": "user", "parts": [{"text": usuario}]}],
                      "generationConfig": {"temperature": 0.2,
                                           "maxOutputTokens": max_tokens}},
                timeout=timeout,
            )
            if r.status_code != 200:
                return None
            partes = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in partes)
        except Exception:
            return None
    url, modelo = _endpoint_y_modelo()
    if not url or not modelo:
        return None
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {_CONFIG['api_key']}",
                     "Content-Type": "application/json"},
            json={"model": modelo, "messages": mensajes,
                  "temperature": 0.2, "max_tokens": max_tokens},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def _extraer_json(texto):
    if not texto:
        return None
    ini = texto.find("{")
    fin = texto.rfind("}")
    if ini == -1 or fin == -1 or fin <= ini:
        return None
    try:
        return json.loads(texto[ini:fin + 1])
    except Exception:
        return None


def segunda_opinion(resumen):
    """Revisa UNA señal. `resumen`: dict con par, tipo, direccion, confianza,
    nivel, razon, regimen y (si hay) probabilidades. Devuelve
    {"veredicto": CONFIRMA|SUBE|REBAJA, "motivo": "..."} o None."""
    prompt = (
        "Revisa esta señal del sistema cuantitativo y responde SOLO con un "
        "JSON asi: {\"veredicto\": \"CONFIRMA\"|\"SUBE\"|\"REBAJA\", "
        "\"motivo\": \"una linea, maximo 25 palabras\"}\n"
        "CONFIRMA = el nivel esta bien. SUBE = merece un nivel mas. "
        "REBAJA = merece un nivel menos (entrada tardia, contra nivel, "
        "contexto debil...). Nunca cambies la direccion.\n\n"
        f"Par: {resumen.get('par')} ({resumen.get('tipo')})\n"
        f"Señal: {resumen.get('direccion')} | Confianza del sistema: "
        f"{resumen.get('confianza')}% | Nivel actual: {resumen.get('nivel')}\n"
        f"Régimen de mercado: {resumen.get('regimen')}\n"
        f"Análisis del sistema: {resumen.get('razon')}\n"
        f"Advertencias: {', '.join(resumen.get('advertencias') or []) or 'ninguna'}"
    )
    datos = _extraer_json(_chat([
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": prompt},
    ], max_tokens=120, timeout=12))
    if not datos:
        return None
    veredicto = str(datos.get("veredicto", "")).upper().strip()
    if veredicto not in ("CONFIRMA", "SUBE", "REBAJA"):
        return None
    return {"veredicto": veredicto,
            "motivo": str(datos.get("motivo", "")).strip()[:200]}


def clima_sesion(par, tipo, variacion_60m_pct, regimen):
    """Frase corta con el clima del par para la sesion (o "" si no hay IA)."""
    prompt = (
        f"En UNA frase breve (maximo 20 palabras), describe el clima actual "
        f"para operar {par} ({tipo}) en operaciones de 1 minuto. "
        f"Variación de la última hora: {variacion_60m_pct}%. "
        f"Régimen detectado: {regimen}. Sin prometer resultados."
    )
    texto = _chat([
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": prompt},
    ], max_tokens=80, timeout=12)
    return (texto or "").strip().strip('"')[:220]


def analizar_historial(resumen_texto):
    """Diario inteligente: informe en Markdown a partir de un resumen del
    historial y las metricas del usuario. Devuelve "" si no hay IA."""
    prompt = (
        "Analiza el historial de operaciones de este usuario como su mentor "
        "de trading. Escribe un informe breve en español con estas secciones: "
        "## Qué está funcionando, ## Qué está fallando, ## Ajustes concretos. "
        "Básate SOLO en los datos dados; si un dato no alcanza, dilo. "
        "Recuerda el equilibrio: con payout 86% necesita acertar mas del "
        "53,8% para no perder.\n\nDATOS:\n" + resumen_texto
    )
    texto = _chat([
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": prompt},
    ], max_tokens=700, timeout=45)
    return (texto or "").strip()
