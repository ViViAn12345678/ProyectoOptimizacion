"""
escenarios.py
═════════════
3 análisis what-if requeridos por el proyecto:

  1. Alza de combustible  → incrementa costos en TODAS las rutas de la red
  2. Cierre de vía        → elimina una arista del grafo
  3. Falla de calidad     → reduce capacidad de un acopio
"""

import copy
import pandas as pd
import networkx as nx
from dataclasses import dataclass


@dataclass
class ResultadoEscenario:
    nombre:           str
    costo_base:       float
    costo_escenario:  float
    diferencia:       float
    porcentaje:       float
    estado_base:      str
    estado_escenario: str
    df_base:          pd.DataFrame
    df_escenario:     pd.DataFrame
    descripcion:      str


def _resolver_sobre(G_mod, construir_modelo_fn, resolver_fn):
    modelo, x, n = construir_modelo_fn(G_mod)
    return resolver_fn(G_mod, modelo, x, n)


# ─── ESCENARIO 1 — Alza de combustible ───────────────────────────────────────
def escenario_combustible(
        G, 
        construir_modelo_fn, 
        resolver_fn,
        resultado_base,
        aristas_afectadas,
        factor: float = 1.15) -> ResultadoEscenario:
    """
    Incrementa el costo de transporte en TODAS las rutas de la red.
    Esto representa un alza nacional del combustible, lo cual es más
    realista y produce un impacto visible en el costo total.
    Las rutas del Meta (distancias largas) se ven más afectadas porque
    su costo variable es mayor al tener más km recorridos.
    """
    G_mod = copy.deepcopy(G)
    afectadas = 0
    for u,v in aristas_afectadas:
        if G_mod.has_edge(u,v):
            G_mod[u][v]["costo_unitario"] *= factor
            G_mod[u][v]["peso"] *= factor

    resultado_mod = _resolver_sobre(G_mod, construir_modelo_fn, resolver_fn)
    costo_base = resultado_base["costo_total"]
    costo_mod  = resultado_mod["costo_total"]
    diff = costo_mod - costo_base
    pct  = (diff / costo_base * 100) if costo_base else 0

    return ResultadoEscenario(
        nombre           = f"Alza de combustible +{round((factor-1)*100)}% en toda la red",
        costo_base       = costo_base,
        costo_escenario  = costo_mod,
        diferencia       = diff,
        porcentaje       = round(pct, 2),
        estado_base      = resultado_base["estado"],
        estado_escenario = resultado_mod["estado"],
        df_base          = resultado_base["df_flujos"],
        df_escenario     = resultado_mod["df_flujos"],
        descripcion      = (
            f"Alza de combustible del {round((factor-1)*100)}% aplicada a las "
            f"{afectadas} rutas de la red. Las rutas largas (Bogotá-Medellín 420km, "
            f"Bogotá-Cali 460km, Quibdó-Bogotá 630km) absorben el mayor impacto "
            f"por su mayor costo variable. El solver puede cambiar el mix de camiones "
            f"o rutas para compensar."
        )
    )


# ─── ESCENARIO 2 — Cierre de vía ─────────────────────────────────────────────
def escenario_cierre_via(G, construir_modelo_fn, resolver_fn,
                          resultado_base: dict,
                          arista: tuple) -> ResultadoEscenario:
    """
    Elimina una arista (y su inversa si es bidireccional).
    Si la red queda infactible, lo reporta claramente.
    """
    u, v = arista
    G_mod = copy.deepcopy(G)

    eliminadas = []
    if G_mod.has_edge(u, v):
        eliminadas.append(f"{G_mod.nodes[u]['nombre']} → {G_mod.nodes[v]['nombre']}")
        G_mod.remove_edge(u, v)
    if G_mod.has_edge(v, u):
        eliminadas.append(f"{G_mod.nodes[v]['nombre']} → {G_mod.nodes[u]['nombre']}")
        G_mod.remove_edge(v, u)

    if not eliminadas:
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
    diff = costo_mod - costo_base
    pct  = (diff / costo_base * 100) if costo_base else 0
    estado_mod = resultado_mod["estado"]

    if estado_mod != "Optimal":
        descripcion = (
            f"Se cerró: {' y '.join(eliminadas)}. "
            f"⚠️ La red quedó INFACTIBLE — esta ruta es un cuello de botella crítico. "
            f"Sin ella no es posible satisfacer toda la demanda."
        )
    else:
        descripcion = (
            f"Se cerró: {' y '.join(eliminadas)}. "
            f"El modelo encontró rutas alternativas con un incremento "
            f"de ${diff:,.0f} COP ({pct:+.1f}%)."
        )

    return ResultadoEscenario(
        nombre           = f"Cierre: {' / '.join(eliminadas)}",
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


# ─── ESCENARIO 3 — Falla de calidad ──────────────────────────────────────────
def escenario_falla_calidad(G, construir_modelo_fn, resolver_fn,
                              resultado_base: dict,
                              nodo_acopio: str,
                              perdida: float = 0.40) -> ResultadoEscenario:
    """
    Reduce la capacidad del acopio y de todas sus aristas de salida.
    Simula una inspección sanitaria que obliga a reducir operaciones.
    """
    G_mod = copy.deepcopy(G)

    if perdida >= 0.8:
        G_mod.nodes[nodo_acopio]["calidad"] = 0
    
    nombre_acopio = G_mod.nodes[nodo_acopio]["nombre"]
    
    afectadas = 0

    for _, v in list(G_mod.out_edges(nodo_acopio)):
        G_mod[nodo_acopio][v]["capacidad"] *= (1 - perdida)
        afectadas += 1

    G_mod.nodes[nodo_acopio]["capacidad"] *= (1 - perdida)

    resultado_mod = _resolver_sobre(G_mod, construir_modelo_fn, resolver_fn)
    costo_base = resultado_base["costo_total"]
    costo_mod  = resultado_mod["costo_total"]
    diff = costo_mod - costo_base
    pct  = (diff / costo_base * 100) if costo_base else 0
    estado_mod = resultado_mod["estado"]

    if estado_mod != "Optimal":
        descripcion = (
            f"Falla de calidad del {round(perdida*100)}% en {nombre_acopio}. "
            f"⚠️ La red quedó INFACTIBLE — la pérdida de capacidad es tan grande "
            f"que no puede satisfacerse toda la demanda. "
            f"Se deben activar rutas de contingencia o reducir demanda."
        )
    else:
        descripcion = (
            f"Falla de calidad del {round(perdida*100)}% en {nombre_acopio}. "
            f"Capacidad reducida en {afectadas} rutas de salida y en el nodo mismo. "
            f"El modelo redistribuyó flujos con un impacto de "
            f"${diff:,.0f} COP ({pct:+.1f}%). "
            f"Si la pérdida supera la capacidad necesaria, la red se vuelve infactible."
        )

    return ResultadoEscenario(
        nombre           = f"Falla calidad {round(perdida*100)}% — {nombre_acopio}",
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
