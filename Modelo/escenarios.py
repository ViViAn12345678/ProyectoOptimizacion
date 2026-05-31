"""
escenarios.py
═════════════
Implementa los 3 análisis what-if requeridos por el proyecto:

  1. Alza de combustible  → incrementa costos en rutas del Meta
  2. Cierre de vía        → elimina una arista del grafo
  3. Falla de calidad     → reduce capacidad de salida de un acopio

Cada función recibe el grafo original, lo copia (sin modificar el original),
aplica el escenario y retorna el grafo modificado + un resumen del impacto
al comparar el costo base vs el costo bajo el escenario.
"""

import copy
import pandas as pd
import networkx as nx
from dataclasses import dataclass


# ─── Estructura de resultado ──────────────────────────────────────────────────

@dataclass
class ResultadoEscenario:
    nombre:          str       # nombre del escenario
    costo_base:      float     # costo óptimo sin modificación
    costo_escenario: float     # costo óptimo con la modificación
    diferencia:      float     # costo_escenario - costo_base
    porcentaje:      float     # % de incremento/decremento
    estado_base:     str       # "Optimal" o no
    estado_escenario:str
    df_base:         pd.DataFrame   # rutas del plan base
    df_escenario:    pd.DataFrame   # rutas del plan modificado
    descripcion:     str       # texto explicando qué cambió


# ─── Utilidad: resolver sobre un grafo modificado ────────────────────────────

def _resolver_sobre(G_mod, construir_modelo_fn, resolver_fn):
    """
    Construye y resuelve el modelo PL sobre el grafo G_mod.
    Retorna el dict {estado, costo_total, df_flujos}.
    """
    modelo, x, n = construir_modelo_fn(G_mod)
    return resolver_fn(G_mod, modelo, x, n)


# ─── ESCENARIO 1 — Alza de combustible en rutas del Meta ─────────────────────

def escenario_combustible(G, construir_modelo_fn, resolver_fn,
                           resultado_base: dict,
                           factor: float = 1.15) -> ResultadoEscenario:
    """
    Simula un alza de combustible en las rutas que salen de orígenes del Meta
    (O4 Pto. Gaitán, O5 Pto. Concordia, O6 Granada) y las rutas entre
    acopios de la región (Villavicencio, Paratebueno, Yopal).

    factor = 1.15 → incremento del 15% (valor por defecto del documento)
    """
    NODOS_META = {"O4", "O5", "O6", "A8", "A9", "A10"}

    G_mod = copy.deepcopy(G)
    aristas_afectadas = 0

    for u, v in G_mod.edges():
        if u in NODOS_META:
            G_mod[u][v]["costo_unitario"] *= factor
            G_mod[u][v]["peso"]           *= factor
            aristas_afectadas += 1

    resultado_mod = _resolver_sobre(G_mod, construir_modelo_fn, resolver_fn)

    costo_base = resultado_base["costo_total"]
    costo_mod  = resultado_mod["costo_total"]
    diff       = costo_mod - costo_base
    pct        = (diff / costo_base * 100) if costo_base else 0

    return ResultadoEscenario(
        nombre           = f"Alza de combustible +{round((factor-1)*100)}% en rutas del Meta",
        costo_base       = costo_base,
        costo_escenario  = costo_mod,
        diferencia       = diff,
        porcentaje       = round(pct, 2),
        estado_base      = resultado_base["estado"],
        estado_escenario = resultado_mod["estado"],
        df_base          = resultado_base["df_flujos"],
        df_escenario     = resultado_mod["df_flujos"],
        descripcion      = (
            f"Se incrementó el costo de transporte un {round((factor-1)*100)}% "
            f"en {aristas_afectadas} rutas que salen de orígenes o pasan por "
            f"acopios del Meta (Villavicencio, Paratebueno, Yopal, "
            f"Pto. Gaitán, Pto. Concordia, Granada)."
        )
    )


# ─── ESCENARIO 2 — Cierre de vía ─────────────────────────────────────────────

def escenario_cierre_via(G, construir_modelo_fn, resolver_fn,
                          resultado_base: dict,
                          arista: tuple) -> ResultadoEscenario:
    """
    Simula el cierre de una vía eliminando la arista (u, v) del grafo.
    Si la arista era bidireccional, también elimina (v, u).

    arista: tupla de IDs de nodo, ej. ("A6", "A10") para Bogotá-Villavicencio
    """
    u, v = arista
    G_mod = copy.deepcopy(G)

    eliminadas = []
    if G_mod.has_edge(u, v):
        nom_u = G_mod.nodes[u]["nombre"]
        nom_v = G_mod.nodes[v]["nombre"]
        G_mod.remove_edge(u, v)
        eliminadas.append(f"{nom_u} → {nom_v}")
    if G_mod.has_edge(v, u):
        nom_v2 = G_mod.nodes[v]["nombre"]
        nom_u2 = G_mod.nodes[u]["nombre"]
        G_mod.remove_edge(v, u)
        eliminadas.append(f"{nom_v2} → {nom_u2}")

    if not eliminadas:
        # La arista no existía
        return ResultadoEscenario(
            nombre="Cierre de vía (arista no encontrada)",
            costo_base=resultado_base["costo_total"],
            costo_escenario=resultado_base["costo_total"],
            diferencia=0, porcentaje=0,
            estado_base=resultado_base["estado"],
            estado_escenario="N/A",
            df_base=resultado_base["df_flujos"],
            df_escenario=pd.DataFrame(),
            descripcion=f"La arista {u}→{v} no existe en el grafo."
        )

    resultado_mod = _resolver_sobre(G_mod, construir_modelo_fn, resolver_fn)

    costo_base = resultado_base["costo_total"]
    costo_mod  = resultado_mod["costo_total"]
    diff       = costo_mod - costo_base
    pct        = (diff / costo_base * 100) if costo_base else 0

    estado_mod = resultado_mod["estado"]
    if estado_mod != "Optimal":
        descripcion = (
            f"Se cerró la vía: {' y '.join(eliminadas)}. "
            f"⚠️ La red quedó INFACTIBLE — no es posible satisfacer toda la demanda "
            f"sin esta ruta. Es un cuello de botella crítico."
        )
    else:
        descripcion = (
            f"Se cerró la vía: {' y '.join(eliminadas)}. "
            f"La red encontró rutas alternativas con un incremento "
            f"de ${diff:,.0f} COP ({pct:+.1f}%)."
        )

    return ResultadoEscenario(
        nombre           = f"Cierre de vía: {' / '.join(eliminadas)}",
        costo_base       = costo_base,
        costo_escenario  = costo_mod,
        diferencia       = diff,
        porcentaje       = round(pct, 2),
        estado_base      = resultado_base["estado"],
        estado_escenario = estado_mod,
        df_base          = resultado_base["df_flujos"],
        df_escenario     = resultado_mod["df_flujos"],
        descripcion      = descripcion
    )


# ─── ESCENARIO 3 — Falla de calidad en acopio ────────────────────────────────

def escenario_falla_calidad(G, construir_modelo_fn, resolver_fn,
                              resultado_base: dict,
                              nodo_acopio: str,
                              perdida: float = 0.40) -> ResultadoEscenario:
    """
    Simula una pérdida masiva de calidad en un centro de acopio:
    reduce la capacidad de SALIDA del acopio afectado en un % dado.

    nodo_acopio : ID del acopio, ej. "A6" para Bogotá
    perdida     : fracción de capacidad perdida, 0.40 = 40% por defecto
    """
    G_mod = copy.deepcopy(G)
    nombre_acopio = G_mod.nodes[nodo_acopio]["nombre"]

    aristas_afectadas = 0
    for _, v in G_mod.out_edges(nodo_acopio):
        G_mod[nodo_acopio][v]["capacidad"] *= (1 - perdida)
        aristas_afectadas += 1

    # También reducimos la capacidad del nodo mismo
    G_mod.nodes[nodo_acopio]["capacidad"] *= (1 - perdida)

    resultado_mod = _resolver_sobre(G_mod, construir_modelo_fn, resolver_fn)

    costo_base = resultado_base["costo_total"]
    costo_mod  = resultado_mod["costo_total"]
    diff       = costo_mod - costo_base
    pct        = (diff / costo_base * 100) if costo_base else 0

    estado_mod = resultado_mod["estado"]
    if estado_mod != "Optimal":
        descripcion = (
            f"Falla de calidad del {round(perdida*100)}% en {nombre_acopio}. "
            f"⚠️ La red quedó INFACTIBLE — la pérdida de capacidad es tan grande "
            f"que no puede satisfacerse toda la demanda. "
            f"Se recomienda activar rutas de contingencia."
        )
    else:
        descripcion = (
            f"Falla de calidad del {round(perdida*100)}% en {nombre_acopio}. "
            f"Se redujo la capacidad de {aristas_afectadas} rutas de salida. "
            f"El modelo redirigió flujos con un impacto de "
            f"${diff:,.0f} COP ({pct:+.1f}%)."
        )

    return ResultadoEscenario(
        nombre           = f"Falla de calidad {round(perdida*100)}% en {nombre_acopio}",
        costo_base       = costo_base,
        costo_escenario  = costo_mod,
        diferencia       = diff,
        porcentaje       = round(pct, 2),
        estado_base      = resultado_base["estado"],
        estado_escenario = estado_mod,
        df_base          = resultado_base["df_flujos"],
        df_escenario     = resultado_mod["df_flujos"],
        descripcion      = descripcion
    )