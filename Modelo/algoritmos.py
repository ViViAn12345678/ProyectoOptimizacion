"""
algoritmos.py
═════════════
Implementa los tres algoritmos de Teoría de Grafos requeridos por el proyecto:

  1. Dijkstra        → ruta de menor costo entre cualquier par de nodos
  2. Flujo Máximo    → cuántas toneladas pueden moverse de una fuente a un sumidero
  3. Corte Mínimo    → cuellos de botella (aristas críticas que limitan el flujo)

Todos operan sobre el mismo DiGraph de NetworkX que construye data_loader.py.
"""

import networkx as nx
from dataclasses import dataclass


# ─── Estructuras de resultado ─────────────────────────────────────────────────

@dataclass
class ResultadoDijkstra:
    origen:       str
    destino:      str
    ruta:         list
    nombres:      list
    costo_total:  float
    distancia_km: float
    num_saltos:   int
    alcanzable:   bool

@dataclass
class ResultadoFlujoMaximo:
    fuente:          str
    sumidero:        str
    flujo_maximo:    float
    distribucion:    dict
    aristas_activas: list

@dataclass
class ResultadoCuellos:
    fuente:         str
    sumidero:       str
    valor_corte:    float
    aristas_corte:  list
    nodos_S:        set
    nodos_T:        set
    descripcion:    list


# ─── 1. DIJKSTRA ──────────────────────────────────────────────────────────────

def dijkstra(G: nx.DiGraph, origen: str, destino: str) -> ResultadoDijkstra:
    """
    Ruta de menor costo (peso = costo_ton_km × distancia_km) entre dos nodos.
    Usa el algoritmo de Dijkstra de NetworkX (O((V+E) log V)).
    """
    try:
        ruta   = nx.dijkstra_path(G, origen, destino, weight="peso")
        costo  = nx.dijkstra_path_length(G, origen, destino, weight="peso")
        dist   = sum(G[ruta[i]][ruta[i+1]]["distancia"] for i in range(len(ruta)-1))
        nombres = [G.nodes[n]["nombre"] for n in ruta]
        return ResultadoDijkstra(
            origen=origen, destino=destino, ruta=ruta, nombres=nombres,
            costo_total=round(costo, 2), distancia_km=round(dist, 1),
            num_saltos=len(ruta)-1, alcanzable=True
        )
    except nx.NetworkXNoPath:
        return ResultadoDijkstra(
            origen=origen, destino=destino, ruta=[], nombres=[],
            costo_total=float("inf"), distancia_km=float("inf"),
            num_saltos=0, alcanzable=False
        )

def tabla_rutas_optimas(G: nx.DiGraph, origenes: list, destinos: list) -> list:
    """Genera una tabla con la ruta óptima entre cada par (origen, destino)."""
    filas = []
    for o in origenes:
        for d in destinos:
            r = dijkstra(G, o, d)
            filas.append({
                "origen_id":    o,
                "origen":       G.nodes[o]["nombre"],
                "destino_id":   d,
                "destino":      G.nodes[d]["nombre"],
                "ruta":         " → ".join(r.nombres) if r.alcanzable else "Sin camino",
                "costo_$/ton":  r.costo_total,
                "distancia_km": r.distancia_km,
                "saltos":       r.num_saltos,
                "alcanzable":   r.alcanzable
            })
    return filas


# ─── 2. FLUJO MÁXIMO ──────────────────────────────────────────────────────────

def _grafo_capacidad(G: nx.DiGraph) -> nx.DiGraph:
    """Construye un grafo con atributo 'capacity' requerido por NetworkX."""
    G_cap = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        G_cap.add_edge(u, v, capacity=data.get("capacidad", 15))
    return G_cap

def flujo_maximo(G: nx.DiGraph, fuente: str, sumidero: str) -> ResultadoFlujoMaximo:
    """
    Flujo máximo entre fuente y sumidero usando Edmonds-Karp (Ford-Fulkerson + BFS).
    La capacidad de cada arista es el límite del camión grande (15 ton por defecto).
    """
    G_cap = _grafo_capacidad(G)
    valor, flujo_dict = nx.maximum_flow(
        G_cap, fuente, sumidero,
        flow_func=nx.algorithms.flow.edmonds_karp
    )
    aristas_activas = [
        (u, v, flujo_dict[u][v])
        for u in flujo_dict
        for v in flujo_dict[u]
        if flujo_dict[u][v] > 0
    ]
    return ResultadoFlujoMaximo(
        fuente=fuente, sumidero=sumidero,
        flujo_maximo=round(valor, 2),
        distribucion=flujo_dict,
        aristas_activas=aristas_activas
    )


# ─── 3. CUELLOS DE BOTELLA (Corte Mínimo) ────────────────────────────────────

def cuellos_de_botella(G: nx.DiGraph, fuente: str, sumidero: str) -> ResultadoCuellos:
    """
    Identifica las aristas críticas por el Teorema Max-Flow Min-Cut:
    el corte mínimo es el conjunto de aristas que, al eliminarse, desconecta
    fuente de sumidero con el menor costo de capacidad total.

    Estas aristas son los CUELLOS DE BOTELLA — si se bloquean, el flujo se detiene.
    """
    G_cap = _grafo_capacidad(G)
    valor_corte, (S, T) = nx.minimum_cut(G_cap, fuente, sumidero)

    aristas_corte = [(u, v) for u in S for v in G_cap.successors(u) if v in T]

    descripcion = []
    for u, v in aristas_corte:
        nom_u = G.nodes[u]["nombre"] if u in G.nodes else u
        nom_v = G.nodes[v]["nombre"] if v in G.nodes else v
        cap   = G[u][v].get("capacidad", "?") if G.has_edge(u, v) else "?"
        dist  = G[u][v].get("distancia", "?") if G.has_edge(u, v) else "?"
        descripcion.append(f"{nom_u} → {nom_v}  |  cap: {cap} ton  |  dist: {dist} km")

    return ResultadoCuellos(
        fuente=fuente, sumidero=sumidero,
        valor_corte=round(valor_corte, 2),
        aristas_corte=aristas_corte,
        nodos_S=S, nodos_T=T,
        descripcion=descripcion
    )

def analisis_cuellos_red(G: nx.DiGraph) -> list:
    """
    Analiza cuellos de botella entre todos los pares origen → acopio
    y los ordena de más a menos crítico (menor flujo máximo = más restringido).
    """
    origenes = [n for n, d in G.nodes(data=True) if d["tipo"] == "origen"]
    acopios  = [n for n, d in G.nodes(data=True) if d["tipo"] == "acopio"]
    filas = []
    for o in origenes:
        for a in acopios:
            try:
                r = cuellos_de_botella(G, o, a)
                filas.append({
                    "origen":           G.nodes[o]["nombre"],
                    "acopio":           G.nodes[a]["nombre"],
                    "flujo_max_ton":    r.valor_corte,
                    "num_cuellos":      len(r.aristas_corte),
                    "aristas_criticas": " | ".join(r.descripcion) if r.descripcion else "—"
                })
            except Exception:
                pass
    return sorted(filas, key=lambda x: x["flujo_max_ton"])


# ─── DEMO ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from Modelo.data_loader import cargar_red
    import pandas as pd

    G = cargar_red()
    print(f"Red: {G.number_of_nodes()} nodos | {G.number_of_edges()} aristas\n")

    # 1. Dijkstra
    print("═"*60)
    print("  DIJKSTRA — Granada → Bogotá")
    print("═"*60)
    r = dijkstra(G, "O6", "A6")
    print(f"  Ruta     : {' → '.join(r.nombres)}")
    print(f"  Costo    : ${r.costo_total:,.2f} /ton")
    print(f"  Distancia: {r.distancia_km} km  |  {r.num_saltos} salto(s)")

    print()
    r2 = dijkstra(G, "O2", "A7")
    print(f"  Pto. Salgar → Bucaramanga: {' → '.join(r2.nombres)}")
    print(f"  Costo: ${r2.costo_total:,.2f}/ton  |  {r2.distancia_km} km")

    # 2. Tabla de rutas
    print()
    print("═"*60)
    print("  TABLA — Orígenes → Bogotá y Villavicencio")
    print("═"*60)
    origenes = [n for n, d in G.nodes(data=True) if d["tipo"] == "origen"]
    tabla = tabla_rutas_optimas(G, origenes, ["A6", "A10"])
    df = pd.DataFrame(tabla)
    print(df[["origen","destino","costo_$/ton","distancia_km","ruta"]].to_string(index=False))

    # 3. Flujo máximo
    print()
    print("═"*60)
    print("  FLUJO MÁXIMO — Granada → Villavicencio")
    print("═"*60)
    fm = flujo_maximo(G, "O6", "A10")
    print(f"  Flujo máximo posible: {fm.flujo_maximo} ton")
    for u, v, f in fm.aristas_activas:
        nu = G.nodes[u]["nombre"] if u in G.nodes else u
        nv = G.nodes[v]["nombre"] if v in G.nodes else v
        print(f"    {nu:22s} → {nv:22s}  {f:.1f} ton")

    # 4. Cuellos de botella
    print()
    print("═"*60)
    print("  CUELLOS DE BOTELLA — Chocontá → Bogotá")
    print("═"*60)
    cb = cuellos_de_botella(G, "O3", "A6")
    print(f"  Valor del corte mínimo: {cb.valor_corte} ton")
    for desc in cb.descripcion:
        print(f"  ⚠ {desc}")

    # 5. Ranking cuellos red completa
    print()
    print("═"*60)
    print("  RANKING CUELLOS — top 5 pares más restringidos")
    print("═"*60)
    ranking = analisis_cuellos_red(G)
    df_r = pd.DataFrame(ranking[:5])
    print(df_r[["origen","acopio","flujo_max_ton","num_cuellos"]].to_string(index=False))
    if ranking:
        print("\n  Aristas críticas del cuello más severo:")
        for linea in ranking[0]["aristas_criticas"].split(" | "):
            print(f"  ⚠ {linea}")
