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
    "pequeño": {"capacidad": 5,  "costo_fijo": 800000},
    "grande":  {"capacidad": 15, "costo_fijo": 2000000},
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
    
    calidad = {}

    for nodo, datos in G.nodes(data=True):
        calidad[nodo] = datos.get("calidad", 1)

    PENALIZACION_CALIDAD = 500000

    falla_calidad = pulp.LpVariable.dicts(
        "falla_calidad",
        G.nodes(),
        cat="Binary"
    )
    
    for nodo in G.nodes():

        if calidad[nodo] == 0:

            modelo += (
                falla_calidad[nodo] == 1,
                f"FallaCalidad_{nodo}"
            )

        else:

            modelo += (
                falla_calidad[nodo] == 0,
                f"SinFallaCalidad_{nodo}"
            )

    # Función objetivo
    modelo += (

        pulp.lpSum(
            G[i][j]["peso"] * x[(i,j,k)]
            for (i,j) in aristas
            for k in tipos
        )
        +pulp.lpSum(
            CAMIONES[k]["costo_fijo"] * n[(i,j,k)]
            for (i,j) in aristas
            for k in tipos
        )
        +pulp.lpSum(
            PENALIZACION_CALIDAD *
            falla_calidad[nodo]
            for nodo in G.nodes()
        )
    )

    # Restricciones
    for nodo, datos in G.nodes(data=True):
        salida  = pulp.lpSum(x[(nodo,j,k)] for j in G.successors(nodo)   for k in tipos)
        entrada = pulp.lpSum(x[(i,nodo,k)] for i in G.predecessors(nodo) for k in tipos)
        
        if calidad[nodo] == 0:

            for j in G.successors(nodo):
                for k in tipos:

                    modelo += (
                        x[(nodo,j,k)] == 0,
                        f"Calidad_{nodo}_{j}_{k}"
                    )
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
 # --- SEGMENTO FINANCIERO ---
    COSTO_PRODUCCION_TON = 9000000
    ingresos         = 0
    costo_produccion = 0
    for nodo, datos in G.nodes(data=True):
        if datos.get("tipo") == "destino":
            demanda     = datos.get("demanda", 0)
            precio      = datos.get("precio_venta", 1200000000)
            ingresos         += demanda * precio
            costo_produccion += demanda * COSTO_PRODUCCION_TON

    ganancia = ingresos - costo_total - costo_produccion

    return {
        "estado":      estado,
        "costo_total": costo_total,
        "ingresos": ingresos,
        "costo_produccion":  costo_produccion,
        "ganancia": ganancia,
        "df_flujos":   pd.DataFrame(filas),
        
    }
