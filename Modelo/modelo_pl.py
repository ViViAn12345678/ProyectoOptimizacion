"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     MODELO MATEMÁTICO — RED LOGÍSTICA ACUÍCOLA REAL DEL META               ║
║     Programación Lineal con dos tipos de camión                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

ÍNDICES Y CONJUNTOS
  O  = {O1..O6}   → Estaciones de origen (pisciculturas)
  A  = {A1..A10}  → Centros de acopio (ciudades intermedias)
  D  = {D1..D25}  → Supermercados destino
  N  = O ∪ A ∪ D  → Todos los nodos del grafo
  E  ⊆ N × N      → Aristas dirigidas (rutas; las bidireccionales se duplican)
  K  = {P, G}     → Tipos de camión: P=Pequeño(5t), G=Grande(15t)

PARÁMETROS
  d_ij    → Distancia en km entre nodo i y j              [km]
  c_ij    → Costo unitario de transporte en arista (i,j)  [$/ton·km]
  cap_k   → Capacidad máxima del camión tipo k            [ton]
            cap_P = 5 ton  |  cap_G = 15 ton
  Of_i    → Oferta del origen i                           [ton]
  De_j    → Demanda del supermercado j                    [ton]
  Cap_a   → Capacidad de almacenamiento en acopio a       [ton]
  α       → Factor de merma en acopios = 0.98
  CF_k    → Costo fijo por viaje: CF_P=$80.000  CF_G=$180.000

VARIABLES DE DECISIÓN
  x_ij_k ≥ 0  → Toneladas en arista (i,j) con camión tipo k
  n_ij_k ∈ ℤ+ → Número de viajes del camión k en arista (i,j)
  Relación:   x_ij_k ≤ cap_k · n_ij_k

FUNCIÓN OBJETIVO
  min Z = Σ_(i,j)∈E Σ_k [ c_ij·d_ij·x_ij_k  +  CF_k·n_ij_k ]
           ↑ costo variable (por ton transportada)   ↑ costo fijo (por viaje)

RESTRICCIONES
  R1  Σ_j x_ij_k  ≤  Of_i                    ∀ i ∈ O   (oferta)
  R2  Σ_j x_aj_k  =  α · Σ_i x_ia_k          ∀ a ∈ A   (equilibrio + merma)
  R3  Σ_i x_ia_k  ≤  Cap_a                   ∀ a ∈ A   (capacidad acopio)
  R4  Σ_i x_ij_k  =  De_j                    ∀ j ∈ D   (demanda exacta)
  R5  x_ij_k      ≤  cap_k · n_ij_k          ∀(i,j),k  (límite camión)
  R6  x_ij_k ≥ 0,  n_ij_k ∈ ℤ+
"""

import pulp
import pandas as pd

# ─── PARÁMETROS DE CAMIONES ───────────────────────────────────────────────────
CAMIONES = {
    "pequeño": {"capacidad": 5,  "costo_fijo": 80_000},
    "grande":  {"capacidad": 15, "costo_fijo": 180_000},
}
MERMA = 0.98


# ─── MODELO PL ────────────────────────────────────────────────────────────────
def construir_modelo(G):
    modelo  = pulp.LpProblem("Acuicola_Real_Meta", pulp.LpMinimize)
    aristas = list(G.edges())
    tipos   = list(CAMIONES.keys())

    x = pulp.LpVariable.dicts("flujo",
        [(i, j, k) for (i,j) in aristas for k in tipos],
        lowBound=0, cat="Continuous")

    n = pulp.LpVariable.dicts("viajes",
        [(i, j, k) for (i,j) in aristas for k in tipos],
        lowBound=0, cat="Integer")

    # Función objetivo
    modelo += (
        pulp.lpSum(G[i][j]["peso"] * x[(i,j,k)]
                   for (i,j) in aristas for k in tipos)
        + pulp.lpSum(CAMIONES[k]["costo_fijo"] * n[(i,j,k)]
                     for (i,j) in aristas for k in tipos)
    ), "Costo_Total"

    # Restricciones
    for nodo, datos in G.nodes(data=True):
        salida  = pulp.lpSum(x[(nodo,j,k)] for j in G.successors(nodo)   for k in tipos)
        entrada = pulp.lpSum(x[(i,nodo,k)] for i in G.predecessors(nodo) for k in tipos)

        if datos["tipo"] == "origen":
            modelo += salida <= datos["capacidad"],          f"R1_oferta_{nodo}"
        elif datos["tipo"] == "acopio":
            modelo += salida == MERMA * entrada,             f"R2_equilibrio_{nodo}"
            modelo += entrada <= datos["capacidad"],         f"R3_capacidad_{nodo}"
        elif datos["tipo"] == "destino":
            modelo += entrada == datos["demanda"],           f"R4_demanda_{nodo}"

    for (i,j) in aristas:
        for k in tipos:
            modelo += (x[(i,j,k)] <= CAMIONES[k]["capacidad"] * n[(i,j,k)],
                       f"R5_cap_{i}_{j}_{k}")

    return modelo, x, n


# ─── RESOLVER Y MOSTRAR ───────────────────────────────────────────────────────
def resolver(G, modelo, x, n):
    tipos = list(CAMIONES.keys())
    modelo.solve(pulp.PULP_CBC_CMD(msg=0))

    estado      = pulp.LpStatus[modelo.status]
    costo_total = pulp.value(modelo.objective) or 0

    filas = []
    for (i, j, k), var in x.items():
        flujo  = pulp.value(var)  or 0
        viajes = pulp.value(n[(i,j,k)]) or 0
        if flujo > 0.01:
            peso = G[i][j]["peso"]
            filas.append({
                "origen":       G.nodes[i]["nombre"],
                "destino":      G.nodes[j]["nombre"],
                "camion":       k,
                "toneladas":    round(flujo, 2),
                "viajes":       int(viajes),
                "costo_ruta_COP": round(peso * flujo + CAMIONES[k]["costo_fijo"] * viajes, 0),
            })

    return {
        "estado":      estado,
        "costo_total": costo_total,
        "df_flujos":   pd.DataFrame(filas),
    }
