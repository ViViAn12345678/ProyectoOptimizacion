import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pulp
import os

from Modelo.data_loader import cargar_red
from Modelo.modelo_pl    import construir_modelo, resolver, CAMIONES, MERMA
from Modelo.algoritmos  import (dijkstra, flujo_maximo,cuellos_de_botella, tabla_rutas_optimas)
from Modelo.escenarios import (escenario_combustible,escenario_cierre_via,escenario_falla_calidad)
from Modelo.Transporte.esquina_noroeste import esquina_noroeste
from Modelo.Transporte.costo_minimo import costo_minimo
from Modelo.Transporte.vogel import vogel
from Modelo.Transporte.seleccion_inicial import mejor_solucion
from Modelo.Transporte.modi import optimizar_modi
from Modelo.Transporte.matriz_costos import construir_matriz_costos
from Modelo.Transporte.equivalencias import equivalencias

# ─── Configuración ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Acuícola Real del Meta",
    page_icon="🐟",
    layout="wide"
)

# ─── Cache: cargar la red una sola vez ───────────────────────────────────────
@st.cache_resource
def get_red():
    return cargar_red()


def cargar_red_usuario(archivo_nodos, archivo_aristas):
    return cargar_red(
        ruta_nodos=archivo_nodos,
        ruta_aristas=archivo_aristas
    )


archivo_nodos = st.sidebar.file_uploader(
    "Nodos CSV",
    type=["csv"]
)

archivo_aristas = st.sidebar.file_uploader(
    "Aristas CSV",
    type=["csv"]
)

if archivo_nodos is not None and archivo_aristas is not None:

    try:
        G = cargar_red_usuario(
            archivo_nodos,
            archivo_aristas
        )

        st.sidebar.success(
            "Red cargada desde archivos del usuario"
        )

    except Exception as e:

        st.sidebar.error(
            f"Error al cargar archivos: {e}"
        )

        G = get_red()

else:
    G = get_red()

todos_nodos   = {d["nombre"]: n for n, d in G.nodes(data=True)}
nodos_fuente  = sorted(d["nombre"] for _, d in G.nodes(data=True) if d["tipo"] in ("origen", "acopio"))
nombres_lista = sorted(todos_nodos.keys())
def nombres_tipo(tipo):
    return sorted([d["nombre"] for _, d in G.nodes(data=True) if d["tipo"] == tipo])

def id_por_nombre(nombre):
    for nid, d in G.nodes(data=True):
        if d["nombre"] == nombre:
            return nid
    return None
TIPO_COLOR = {"origen": "#1D9E75", "acopio": "#7F77DD", "destino": "#D85A30"}

# ─── Sidebar: navegación ──────────────────────────────────────────────────────
st.sidebar.title("🐟 Acuícola Real del Meta")
pagina = st.sidebar.radio("Navegar", [
    "🏠 Inicio",
    "📐 Función Objetivo",
    "🗺️ Grafo de la Red",
    "⚙️ Optimización PL",
    "🔍 Algoritmos de Grafos",
    "🧪 Escenarios What-If",
    "🗄️ Administración de Datos",
    "🚚 Método de Transporte"
])

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — INICIO
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠 Inicio":
    st.title("Red Logística — Acuícola Real del Meta")
    st.markdown("Sistema de optimización de distribución de pescado a nivel nacional.")

    col1, col2, col3, col4 = st.columns(4)
    origenes  = [n for n,d in G.nodes(data=True) if d["tipo"]=="origen"]
    acopios   = [n for n,d in G.nodes(data=True) if d["tipo"]=="acopio"]
    destinos  = [n for n,d in G.nodes(data=True) if d["tipo"]=="destino"]

    col1.metric("Estaciones de origen",  len(origenes))
    col2.metric("Centros de acopio",     len(acopios))
    col3.metric("Supermercados destino", len(destinos))
    col4.metric("Rutas de transporte",   G.number_of_edges())

    st.divider()
    st.subheader("¿Qué hace este sistema?")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**Optimización (PL)**\n\nEncuentra la combinación de rutas y camiones que minimiza el costo total de transporte cumpliendo toda la demanda.")
    with col_b:
        st.info("**Algoritmos de Grafos**\n\nCalcula la ruta más barata entre dos puntos (Dijkstra) e identifica qué rutas son críticas para la red (flujo máximo).")
    with col_c:
        st.info("**Camiones disponibles**\n\n🚛 Camión pequeño: 5 ton — $80.000/viaje\n\n🚚 Camión grande: 15 ton — $180.000/viaje")

    st.divider()
    st.subheader("Demanda total por región")
    filas = []
    for n, d in G.nodes(data=True):
        if d["tipo"] == "destino" and d["demanda"] > 0:
            filas.append({"Supermercado": d["nombre"],
                          "Departamento": d["departamento"],
                          "Demanda (ton)": d["demanda"]})
    df_dem = pd.DataFrame(filas)
    resumen = df_dem.groupby("Departamento")["Demanda (ton)"].sum().reset_index()
    resumen = resumen.sort_values("Demanda (ton)", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(resumen["Departamento"], resumen["Demanda (ton)"],
            color="#7F77DD", edgecolor="none")
    ax.set_xlabel("Toneladas")
    ax.set_title("Demanda total por departamento")
    ax.spines[["top","right","left"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — FUNCIÓN OBJETIVO (visual y explicada)
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📐 Función Objetivo":
    st.title("📐 Función Objetivo y Restricciones")
    st.markdown("Aquí se explica **qué minimiza el modelo** y **por qué cada restricción existe**.")

    # ── Función objetivo ──────────────────────────────────────────────────────
    st.subheader("Función Objetivo")
    st.markdown("El modelo busca minimizar el **costo total** de operar la red:")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.latex(r"""
        \min Z = \underbrace{\sum_{(i,j)\in E} \sum_{k\in K} c_{ij} \cdot d_{ij} \cdot x_{ij}^k}_{\text{Costo variable (por tonelada)}}
        + \underbrace{\sum_{(i,j)\in E} \sum_{k\in K} CF_k \cdot n_{ij}^k}_{\text{Costo fijo (por viaje)}}
        """)
    with col2:
        st.markdown("""
        | Símbolo | Significado |
        |---|---|
        | $x_{ij}^k$ | Toneladas enviadas por ruta $(i→j)$ con camión $k$ |
        | $n_{ij}^k$ | Número de viajes del camión $k$ en ruta $(i→j)$ |
        | $c_{ij}$ | Costo por tonelada/km en la ruta |
        | $d_{ij}$ | Distancia en km de la ruta |
        | $CF_k$ | Costo fijo por viaje del camión $k$ |
        """)

    # ── Visualización del costo variable vs fijo ──────────────────────────────
    st.divider()
    st.subheader("¿Cómo interactúan el costo variable y el fijo?")
    st.markdown("Mueve el slider para ver cómo cambia el costo total según las toneladas enviadas.")

    toneladas = st.slider("Toneladas a enviar", 1, 30, 10)
    distancia_ej = st.slider("Distancia de la ruta (km)", 50, 700, 200)
    costo_unit   = 2.0

    costo_var = costo_unit * distancia_ej * toneladas

    # Camión pequeño
    viajes_p  = -(-toneladas // 5)   # ceil division
    costo_p   = costo_var + viajes_p * 80_000
    # Camión grande
    viajes_g  = -(-toneladas // 15)
    costo_g   = costo_var + viajes_g * 180_000

    col1, col2, col3 = st.columns(3)
    col1.metric("Costo variable", f"${costo_var:,.0f}")
    col2.metric(f"Total camión pequeño ({viajes_p} viajes)", f"${costo_p:,.0f}")
    col3.metric(f"Total camión grande ({viajes_g} viajes)",  f"${costo_g:,.0f}")

    if costo_p < costo_g:
        st.success(f"✅ Conviene el **camión pequeño** para esta ruta (ahorra ${costo_g-costo_p:,.0f})")
    else:
        st.success(f"✅ Conviene el **camión grande** para esta ruta (ahorra ${costo_p-costo_g:,.0f})")

    # Gráfica comparativa
    tons_range = list(range(1, 31))
    costos_p, costos_g = [], []
    for t in tons_range:
        cv = costo_unit * distancia_ej * t
        costos_p.append(cv + (-(-t//5))  * 80_000)
        costos_g.append(cv + (-(-t//15)) * 180_000)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(tons_range, costos_p, color="#1D9E75", label="Camión pequeño (5t)", linewidth=2)
    ax.plot(tons_range, costos_g, color="#7F77DD", label="Camión grande (15t)", linewidth=2)
    ax.axvline(toneladas, color="#D85A30", linestyle="--", alpha=0.7, label=f"Selección: {toneladas} ton")
    ax.set_xlabel("Toneladas enviadas")
    ax.set_ylabel("Costo total ($)")
    ax.set_title("Costo total según tipo de camión y cantidad")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}k"))
    st.pyplot(fig)
    plt.close()

    # ── Restricciones ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Las 6 Restricciones del Modelo")

    restricciones = [
        ("R1 — Oferta",
         r"\sum_j x_{ij}^k \leq Of_i \quad \forall i \in O",
         "Lo que sale de cada piscicultura no puede superar su producción disponible.",
         "origen"),
        ("R2 — Equilibrio de flujo",
         r"\sum_j x_{aj}^k = \alpha \cdot \sum_i x_{ia}^k \quad \forall a \in A",
         "Todo lo que entra a un acopio debe salir (menos el 2% de merma por refrigeración). α = 0.98",
         "acopio"),
        ("R3 — Capacidad de acopio",
         r"\sum_i x_{ia}^k \leq Cap_a \quad \forall a \in A",
         "Un acopio no puede recibir más toneladas de las que tiene capacidad de almacenar.",
         "acopio"),
        ("R4 — Demanda exacta",
         r"\sum_i x_{ij}^k = De_j \quad \forall j \in D",
         "Cada supermercado debe recibir exactamente la cantidad que pidió, ni más ni menos.",
         "destino"),
        ("R5 — Capacidad del camión",
         r"x_{ij}^k \leq cap_k \cdot n_{ij}^k \quad \forall (i,j), k",
         "Las toneladas enviadas no pueden superar la capacidad física del camión multiplicada por el número de viajes.",
         "camion"),
        ("R6 — No negatividad",
         r"x_{ij}^k \geq 0, \quad n_{ij}^k \in \mathbb{Z}^+",
         "No se pueden enviar toneladas negativas y los viajes deben ser números enteros positivos.",
         "logica"),
    ]

    colores = {"origen":"#1D9E75","acopio":"#7F77DD","destino":"#D85A30",
               "camion":"#BA7517","logica":"#888780"}

    for nombre, formula, explicacion, tipo in restricciones:
        color = colores[tipo]
        with st.container(border=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**{nombre}**")
                st.latex(formula)
            with col2:
                st.markdown(f"<div style='padding:12px;border-left:4px solid {color};background:var(--background-color)'>{explicacion}</div>",
                            unsafe_allow_html=True)

    # ── Tabla de parámetros ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Parámetros del modelo")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Camiones**")
        st.dataframe(pd.DataFrame([
            {"Tipo": "Pequeño", "Capacidad (ton)": 5,  "Costo fijo/viaje": "$80.000"},
            {"Tipo": "Grande",  "Capacidad (ton)": 15, "Costo fijo/viaje": "$180.000"},
        ]), hide_index=True)
    with col2:
        st.markdown("**Otros parámetros**")
        st.dataframe(pd.DataFrame([
            {"Parámetro": "Factor de merma (α)", "Valor": "0.98 (2% pérdida)"},
            {"Parámetro": "Costo base ($/ton·km)", "Valor": "0.8 – 2.9"},
            {"Parámetro": "Distancias", "Valor": "65 – 630 km"},
        ]), hide_index=True)

    st.divider()
    st.subheader("Cálculo de la Matriz de Costos")

    st.markdown("""
    Antes de aplicar el método de transporte, el programa calcula el costo mínimo
    de transportar una tonelada desde cada origen hasta cada destino.
    """)

    st.latex(r"""
    c_{ij} = \sum_{(u,v)\in Ruta_{ij}}
    (distancia_{uv}\times costo_{uv})
    """)

    st.info("""
    Para cada combinación origen-destino se ejecuta el algoritmo de Dijkstra,
    obteniendo la ruta de menor costo dentro de la red logística.
    """)

    st.subheader("Matriz de Transporte")

    st.latex(r"""
    C =
    \begin{bmatrix}
    c_{11} & c_{12} & \cdots & c_{1n}\\
    c_{21} & c_{22} & \cdots & c_{2n}\\
    \vdots & \vdots & \ddots & \vdots\\
    c_{m1} & c_{m2} & \cdots & c_{mn}
    \end{bmatrix}
    """)

    st.markdown("""
    Cada elemento de la matriz representa el costo de transportar una unidad
    desde un origen hasta un destino.
    """)

    st.divider()
    st.subheader("Generación de la Solución Inicial")

    with st.expander("Método Esquina Noroeste"):

        st.markdown("""
        Comienza en la esquina superior izquierda de la matriz y asigna la mayor
        cantidad posible sin violar restricciones de oferta o demanda.
        """)

        st.latex(r"""
        x_{ij}= \min(Oferta_i,Demanda_j)
        """)

    with st.expander("Método de Costo Mínimo"):

        st.markdown("""
        Selecciona primero la celda con menor costo unitario y asigna la mayor
        cantidad posible.
        """)

        st.latex(r"""
        \min(c_{ij})
        """)

    with st.expander("Método de Aproximación de Vogel"):

        st.markdown("""
        Calcula penalizaciones entre los dos costos más bajos de cada fila y
        columna para decidir dónde asignar primero.
        """)

        st.latex(r"""
        Penalización =
        c_{segundo\ menor}
        -
        c_{menor}
        """)

    st.divider()
    st.subheader("Optimización MODI")
    st.markdown("""
    Una vez obtenida la solución inicial, se aplica el método MODI
    (Modified Distribution Method) para verificar si existe una
    solución de menor costo.
    """)

    st.latex(r"""
    u_i + v_j = c_{ij}
    """)
    st.markdown("""
    Los potenciales \(u_i\) y \(v_j\) se calculan usando las celdas básicas
    de la solución actual.
    """)

    st.latex(r"""
    \Delta_{ij}\ge 0
    \qquad
    \forall(i,j)
    """)

    st.divider()
    st.subheader("Costo Total Optimizado")

    st.latex(r"""
    Z =
    \sum_{i=1}^{m}
    \sum_{j=1}^{n}
    c_{ij}x_{ij}
    """)

    st.markdown("""
    El costo total corresponde a la suma del costo unitario de transporte
    multiplicado por la cantidad enviada en cada asignación.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — GRAFO DE LA RED
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🗺️ Grafo de la Red":
    st.title(" Grafo de la Red Logística")

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"🟢 **Orígenes** — pisciculturas")
    col2.markdown(f"🟣 **Acopios** — ciudades intermedias")
    col3.markdown(f"🔴 **Destinos** — supermercados")

    mostrar = st.multiselect("Mostrar tipos de nodo",
        ["origen","acopio","destino"], default=["origen","acopio"])

    nodos_vis = [n for n,d in G.nodes(data=True) if d["tipo"] in mostrar]
    G_vis = G.subgraph(nodos_vis)

    fig, ax = plt.subplots(figsize=(14, 8))
    pos = nx.spring_layout(G_vis, seed=42, k=2)
    colores_nodos = [TIPO_COLOR[G.nodes[n]["tipo"]] for n in G_vis.nodes()]
    labels = {n: G.nodes[n]["nombre"].replace(" ","\\n") for n in G_vis.nodes()}

    nx.draw_networkx_nodes(G_vis, pos, node_color=colores_nodos,
                           node_size=800, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G_vis, pos, edge_color="#cccccc",
                           arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1", ax=ax)
    nx.draw_networkx_labels(G_vis, pos,
                            labels={n: G.nodes[n]["nombre"] for n in G_vis.nodes()},
                            font_size=7, ax=ax)

    parches = [mpatches.Patch(color=c, label=t)
               for t, c in TIPO_COLOR.items() if t in mostrar]
    ax.legend(handles=parches, loc="upper left")
    ax.axis("off")
    ax.set_title("Red Logística Acuícola Real del Meta")
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("Tabla de aristas")
    filas = []
    for u, v, d in G.edges(data=True):
        filas.append({
            "Origen": G.nodes[u]["nombre"],
            "Destino": G.nodes[v]["nombre"],
            "Distancia (km)": d["distancia"],
            "Costo ($/ton·km)": d["costo_unitario"],
            "Peso ($/ton)": round(d["peso"], 2),
        })
    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — OPTIMIZACIÓN PL
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "⚙️ Optimización PL":
    st.title("Optimización — Programación Lineal")

    st.info("El solver encuentra la combinación de rutas y camiones que **minimiza el costo total** cumpliendo oferta, demanda, capacidades y merma.")

    if st.button("▶ Resolver modelo", type="primary"):
        with st.spinner("Resolviendo..."):
            modelo, x, n_var = construir_modelo(G)
            resultado = resolver(G, modelo, x, n_var)
            df_res = resultado["df_flujos"]

            st.metric(
                "Ingresos",
                f"${resultado['ingresos']:,.0f}"
            )

            st.metric(
                "Ganancia",
                f"${resultado['ganancia']:,.0f}"
            )

        estado = resultado["estado"]
        costo  = resultado["costo_total"]

        if estado == "Optimal":
            st.success(f"Solución óptima encontrada!")
        else:
            st.error(f"Estado: {estado}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Costo total mínimo", f"${costo:,.0f} COP")
        col2.metric("Rutas activas", len(df_res))
        col3.metric("Estado", estado)

        st.divider()
        st.subheader("Desglose de rutas activas")
        st.dataframe(df_res, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Costo por tipo de camión")
        if not df_res.empty:
            resumen = df_res.groupby("camion")["costo_ruta_COP"].sum().reset_index()
            resumen.columns = ["Camión", "Costo total ($)"]
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(resumen["Camión"], resumen["Costo total ($)"],
                   color=["#1D9E75","#7F77DD"])
            ax.set_ylabel("Costo ($)")
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x,_: f"${x/1_000_000:.1f}M"))
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig)
            plt.close()

        # Guardar en data/
        out = Path(__file__).parent / "Data" / "solucion_optima.csv"
        df_res.to_csv(out, index=False)
        st.caption(f"Resultados guardados en data/solucion_optima.csv")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — ALGORITMOS DE GRAFOS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔍 Algoritmos de Grafos":
    st.title(" Algoritmos de Grafos")

    tab1, tab2, tab3 = st.tabs(["Dijkstra", "Flujo Máximo", "Cuellos de Botella"])


    # ── Dijkstra ──────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Ruta de menor costo — Dijkstra")
        st.markdown("Encuentra el camino más barato (en $/ton) entre dos puntos de la red.")

        col1, col2 = st.columns(2)
        origen_n  = col1.selectbox("Origen", nodos_fuente,  key="dij_o")
        destino_n = col2.selectbox("Destino", nombres_lista, key="dij_d")

        if st.button("Calcular ruta óptima", key="btn_dij"):
            origen_id  = todos_nodos[origen_n]
            destino_id = todos_nodos[destino_n]
            r = dijkstra(G, origen_id, destino_id)

            if r.alcanzable:
                st.success("Ruta encontrada")
                col1, col2, col3 = st.columns(3)
                col1.metric("Costo total", f"${r.costo_total:,.2f} /ton")
                col2.metric("Distancia",   f"{r.distancia_km} km")
                col3.metric("Saltos",      r.num_saltos)

                st.markdown("**Ruta:**  " + "  →  ".join(r.nombres))

                # Resaltar ruta en el grafo
                fig, ax = plt.subplots(figsize=(12, 7))
                nodos_vis = [n for n,d in G.nodes(data=True)
                             if d["tipo"] in ["origen","acopio"]]
                G_vis = G.subgraph(nodos_vis)
                pos = nx.spring_layout(G_vis, seed=42, k=2)

                colores = [TIPO_COLOR[G.nodes[n]["tipo"]] for n in G_vis.nodes()]
                nx.draw_networkx_nodes(G_vis, pos, node_color=colores,
                                       node_size=600, ax=ax, alpha=0.5)
                nx.draw_networkx_edges(G_vis, pos, edge_color="#dddddd",
                                       arrows=True, ax=ax,
                                       connectionstyle="arc3,rad=0.1")
                nx.draw_networkx_labels(G_vis, pos,
                    labels={n: G.nodes[n]["nombre"] for n in G_vis.nodes()},
                    font_size=7, ax=ax)

                # Aristas de la ruta en rojo
                aristas_ruta = list(zip(r.ruta[:-1], r.ruta[1:]))
                aristas_ruta_vis = [(u,v) for u,v in aristas_ruta if u in G_vis and v in G_vis]
                if aristas_ruta_vis:
                    nx.draw_networkx_edges(G_vis, pos,
                        edgelist=aristas_ruta_vis,
                        edge_color="#D85A30", width=3,
                        arrows=True, ax=ax,
                        connectionstyle="arc3,rad=0.1")
                ax.axis("off")
                st.pyplot(fig)
                plt.close()
            else:
                st.error("No existe ruta entre esos dos nodos.")

    # ── Flujo Máximo ──────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Flujo Máximo — Edmonds-Karp")
        st.markdown("Calcula cuántas toneladas pueden circular **como máximo** entre dos nodos considerando las capacidades de cada ruta.")

        col1, col2 = st.columns(2)
        fuente_n   = col1.selectbox("Fuente",   nodos_fuente,  key="fm_f")
        sumidero_n = col2.selectbox("Sumidero", nombres_lista, key="fm_s")

        if st.button("Calcular flujo máximo", key="btn_fm"):
            fuente_id   = todos_nodos[fuente_n]
            sumidero_id = todos_nodos[sumidero_n]
            try:
                fm = flujo_maximo(G, fuente_id, sumidero_id)
                st.success(f"Flujo máximo: **{fm.flujo_maximo} toneladas**")

                filas = []
                for u, v, f in fm.aristas_activas:
                    filas.append({
                        "Desde": G.nodes[u]["nombre"] if u in G.nodes else u,
                        "Hacia": G.nodes[v]["nombre"] if v in G.nodes else v,
                        "Flujo (ton)": f,
                        "Capacidad (ton)": G[u][v].get("capacidad",15) if G.has_edge(u,v) else "—"
                    })
                if filas:
                    st.dataframe(pd.DataFrame(filas), hide_index=True,
                                 use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Cuellos de Botella ────────────────────────────────────────────────────
    with tab3:
        st.subheader("Cuellos de Botella — Corte Mínimo")
        st.markdown("""
        Identifica las aristas **críticas** de la red: si alguna de estas rutas
        se bloquea (cierre de vía, accidente), el flujo entre fuente y sumidero
        se interrumpe o se reduce drásticamente.
        """)

        col1, col2 = st.columns(2)
        fuente_cb_n   = col1.selectbox("Fuente",   nodos_fuente,  key="cb_f")
        sumidero_cb_n = col2.selectbox("Sumidero", nombres_lista, key="cb_s")

        if st.button("Identificar cuellos de botella", key="btn_cb"):
            fuente_cb_id   = todos_nodos[fuente_cb_n]
            sumidero_cb_id = todos_nodos[sumidero_cb_n]
            try:
                cb = cuellos_de_botella(G, fuente_cb_id, sumidero_cb_id)

                col1, col2 = st.columns(2)
                col1.metric("Valor del corte mínimo", f"{cb.valor_corte} ton")
                col2.metric("Aristas críticas", len(cb.aristas_corte))

                if cb.descripcion:
                    st.warning("⚠️ Rutas críticas — si se bloquean, afectan el flujo:")
                    for desc in cb.descripcion:
                        st.markdown(f"- `{desc}`")
                else:
                    st.info("No se encontraron cuellos de botella entre estos nodos.")
            except Exception as e:
                st.error(f"Error: {e}")
# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 6 — ESCENARIOS WHAT-IF
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🧪 Escenarios What-If":
    st.title("🧪 Análisis de Escenarios — What-If")
    st.markdown("""
    Modifica variables clave de la red y observa cómo cambia el costo óptimo.
    El modelo se resuelve dos veces: **base** y **con el escenario**, y se comparan los resultados.
    """)
    # Diccionario para convertir nombre visible -> IDs reales
    mapa_rutas = {
        f"{G.nodes[u]['nombre']} → {G.nodes[v]['nombre']}": (u, v)
        for u, v in G.edges()
    }

    # Lista visible al usuario
    lista_aristas = list(mapa_rutas.keys())

    # Selector
    aristas_seleccionadas = st.multiselect(
        "Rutas afectadas",
        lista_aristas
    )

    # Convertir nombres nuevamente a IDs internos
    aristas_afectadas = [
        mapa_rutas[a]
        for a in aristas_seleccionadas
    ]
        
    # ── Resolver base primero ─────────────────────────────────────────────────
    st.info("⚙️ Primero se resuelve el modelo base para tener un punto de comparación.")
    if st.button("▶ Resolver modelo base", type="primary", key="btn_base_esc"):
        with st.spinner("Resolviendo modelo base..."):
            modelo_b, x_b, n_b = construir_modelo(G)
            st.session_state["resultado_base"] = resolver(G, modelo_b, x_b, n_b)
        st.success(f"Base resuelta — Costo: ${st.session_state['resultado_base']['costo_total']:,.0f} COP")
 
    if "resultado_base" not in st.session_state:
        st.warning("Resuelve el modelo base primero para poder comparar escenarios.")
        st.stop()
 
    resultado_base = st.session_state["resultado_base"]
    st.metric("Costo base actual", f"${resultado_base['costo_total']:,.0f} COP")
 
    st.divider()
    tab1, tab2, tab3 = st.tabs([
        "🔥 Alza de combustible",
        "🚧 Cierre de vía",
        "🐟 Falla de calidad",
    ])
 
    # ── Tab 1: Combustible ────────────────────────────────────────────────────
    with tab1:
        st.subheader("Escenario 1 — Alza de combustible en rutas del Meta")
        st.markdown("""
        **¿Qué simula?** Un incremento en el precio del combustible afecta
        directamente el costo de transporte en las rutas que salen de las
        pisciculturas del Meta y circulan por sus acopios
        (Villavicencio, Paratebueno, Yopal).
        """)
 
        pct_combustible = st.slider(
            "Incremento en el precio del combustible (%)",
            min_value=5, max_value=50, value=15, step=5,
            key="sl_comb"
        )
        factor = 1 + pct_combustible / 100
 
        if st.button("Simular alza de combustible", key="btn_comb"):
            with st.spinner("Resolviendo escenario..."):
                from Modelo.escenarios import escenario_combustible
                r = escenario_combustible(G, construir_modelo, resolver,
                                          resultado_base, aristas_afectadas ,factor)
 
            c1, c2, c3 = st.columns(3)
            c1.metric("Costo base",       f"${r.costo_base:,.0f} COP")
            c2.metric("Costo escenario",  f"${r.costo_escenario:,.0f} COP",
                      delta=f"+${r.diferencia:,.0f}" if r.diferencia >= 0
                            else f"-${abs(r.diferencia):,.0f}")
            c3.metric("Incremento",       f"{r.porcentaje:+.2f}%")
 
            st.info(r.descripcion)
 
            if r.estado_escenario == "Optimal":
                st.subheader("Comparación de rutas activas")
                col_b, col_e = st.columns(2)
                with col_b:
                    st.markdown("**Plan base**")
                    st.dataframe(r.df_base[["origen","destino","camion","toneladas","costo_ruta_COP"]],
                                 hide_index=True, use_container_width=True)
                with col_e:
                    st.markdown("**Plan con alza de combustible**")
                    st.dataframe(r.df_escenario[["origen","destino","camion","toneladas","costo_ruta_COP"]],
                                 hide_index=True, use_container_width=True)
 
                # Gráfica comparativa
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(["Base", f"+{pct_combustible}% combustible"],
                       [r.costo_base, r.costo_escenario],
                       color=["#1D9E75", "#D85A30"])
                ax.set_ylabel("Costo total ($)")
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda v,_: f"${v/1_000_000:.2f}M"))
                ax.spines[["top","right"]].set_visible(False)
                st.pyplot(fig)
                plt.close()
            else:
                st.error(f"⚠️ Estado del solver: {r.estado_escenario}")
 
    # ── Tab 2: Cierre de vía ──────────────────────────────────────────────────
    with tab2:
        st.subheader("Escenario 2 — Cierre de una vía principal")
        st.markdown("""
        **¿Qué simula?** El cierre de una ruta por derrumbe, inundación u otro
        evento elimina esa arista del grafo. El modelo busca rutas alternativas
        y muestra cuánto más caro resulta o si la red queda infactible.
        """)
 
        # Selector de arista: solo mostrar rutas entre acopios (las más críticas)
        aristas_acopio = [
            (u, v) for u, v in G.edges()
            if G.nodes[u]["tipo"] in ["origen","acopio"]
            and G.nodes[v]["tipo"] in ["acopio"]
        ]
        opciones_aristas = {
            f"{G.nodes[u]['nombre']} → {G.nodes[v]['nombre']}": (u, v)
            for u, v in aristas_acopio
        }
        arista_sel = st.selectbox(
            "Selecciona la vía a cerrar",
            list(opciones_aristas.keys()),
            key="sel_via"
        )
 
        if st.button("Simular cierre de vía", key="btn_via"):
            arista_ids = opciones_aristas[arista_sel]
            with st.spinner("Resolviendo escenario..."):
                from Modelo.escenarios import escenario_cierre_via
                r = escenario_cierre_via(G, construir_modelo, resolver,
                                         resultado_base, arista_ids)
 
            c1, c2, c3 = st.columns(3)
            c1.metric("Costo base",      f"${r.costo_base:,.0f} COP")
            if r.estado_escenario == "Optimal":
                c2.metric("Costo escenario", f"${r.costo_escenario:,.0f} COP",
                          delta=f"+${r.diferencia:,.0f}")
                c3.metric("Incremento",      f"{r.porcentaje:+.2f}%")
                st.info(r.descripcion)
 
                col_b, col_e = st.columns(2)
                with col_b:
                    st.markdown("**Plan base**")
                    st.dataframe(r.df_base[["origen","destino","camion","toneladas"]],
                                 hide_index=True, use_container_width=True)
                with col_e:
                    st.markdown("**Plan con vía cerrada**")
                    st.dataframe(r.df_escenario[["origen","destino","camion","toneladas"]],
                                 hide_index=True, use_container_width=True)
            else:
                c2.metric("Estado", r.estado_escenario)
                c3.metric("Impacto", "RED INFACTIBLE")
                st.error(r.descripcion)
 
    # ── Tab 3: Falla de calidad ───────────────────────────────────────────────
    with tab3:
        st.subheader("Escenario 3 — Falla masiva de calidad en un acopio")
        st.markdown("""
        **¿Qué simula?** Una inspección sanitaria detecta problemas en un centro
        de acopio y obliga a reducir su capacidad operativa. El modelo redistribuye
        el flujo hacia otros acopios disponibles.
        """)
 
        nom_acopios_lista = nombres_tipo("acopio")
        acopio_sel = st.selectbox("Centro de acopio afectado",
                                   nom_acopios_lista, key="sel_acopio")
        perdida_pct = st.slider("Pérdida de capacidad (%)",
                                 min_value=10, max_value=80,
                                 value=40, step=10, key="sl_calidad")
 
        if st.button("Simular falla de calidad", key="btn_calidad"):
            acopio_id = id_por_nombre(acopio_sel)
            with st.spinner("Resolviendo escenario..."):
                from Modelo.escenarios import escenario_falla_calidad
                r = escenario_falla_calidad(G, construir_modelo, resolver,
                                             resultado_base,
                                             acopio_id, aristas_afectadas, perdida_pct / 100)
 
            c1, c2, c3 = st.columns(3)
            c1.metric("Costo base",      f"${r.costo_base:,.0f} COP")
            if r.estado_escenario == "Optimal":
                c2.metric("Costo escenario", f"${r.costo_escenario:,.0f} COP",
                          delta=f"+${r.diferencia:,.0f}")
                c3.metric(
                    "Ganancia perdida",
                    f"${r.perdida_ganancia:,.0f} COP"
                )
 
                col_b, col_e = st.columns(2)
                with col_b:
                    st.markdown("**Plan base**")
                    st.dataframe(r.df_base[["origen","destino","camion","toneladas"]],
                                 hide_index=True, use_container_width=True)
                with col_e:
                    st.markdown(f"**Plan con falla en {acopio_sel}**")
                    st.dataframe(r.df_escenario[["origen","destino","camion","toneladas"]],
                                 hide_index=True, use_container_width=True)
 
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(["Base", f"Falla {perdida_pct}% en {acopio_sel}"],
                       [r.costo_base, r.costo_escenario],
                       color=["#1D9E75", "#D85A30"])
                ax.set_ylabel("Costo total ($)")
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda v,_: f"${v/1_000_000:.2f}M"))
                ax.spines[["top","right"]].set_visible(False)
                st.pyplot(fig)
                plt.close()
            else:
                c2.metric("Estado", r.estado_escenario)
                c3.metric("Impacto", "RED INFACTIBLE")
                st.error(r.descripcion)
# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 7 — ADMINISTRACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🗄️ Administración de Datos":

    st.title("Administración de Nodos y Aristas")

    nodos_df = pd.read_csv("Data/nodos.csv")
    aristas_df = pd.read_csv("Data/aristas.csv")

    st.subheader("Nodos actuales")
    st.dataframe(nodos_df)

    st.subheader("Aristas")
    st.dataframe(aristas_df)

    st.subheader("Agregar Nodo")

    with st.form("nuevo_nodo"):

        id_nodo = st.text_input("ID")
        nombre = st.text_input("Nombre")

        tipo = st.selectbox(
            "Tipo",
            ["origen", "acopio", "destino"]
        )

        capacidad = st.number_input(
            "Capacidad",
            min_value=0.0
        )

        demanda = st.number_input(
            "Demanda",
            min_value=0.0
        )

        departamento = st.text_input(
            "Departamento"
        )

        enviar = st.form_submit_button(
            "Guardar"
        )

        if enviar:

            nuevo = pd.DataFrame([
                {
                    "id": id_nodo,
                    "nombre": nombre,
                    "tipo": tipo,
                    "capacidad_ton": capacidad,
                    "demanda_ton": demanda,
                    "departamento": departamento
                }
            ])

            nodos_df = pd.concat(
                [nodos_df, nuevo],
                ignore_index=True
            )

            nodos_df.to_csv(
                "Data/nodos.csv",
                index=False
            )

            st.success("Nodo guardado")

    st.subheader("Editar Nodo")

    id_editar = st.selectbox(
        "Seleccione Nodo",
        nodos_df["id"],
        key="editar_nodo"
    )

    fila = nodos_df[
        nodos_df["id"] == id_editar
    ].iloc[0]

    nombre_edit = st.text_input(
        "Nombre",
        value=fila["nombre"]
    )

    tipo_edit = st.selectbox(
        "Tipo",
        ["origen","acopio","destino"],
        index=[
            "origen",
            "acopio",
            "destino"
        ].index(fila["tipo"])
    )

    capacidad_edit = st.number_input(
        "Capacidad",
        value=float(fila["capacidad_ton"])
    )

    demanda_edit = st.number_input(
        "Demanda",
        value=float(fila["demanda_ton"])
    )

    departamento_edit = st.text_input(
        "Departamento",
        value=fila["departamento"]
    )

    if st.button("Actualizar Nodo"):

        idx = nodos_df[
            nodos_df["id"] == id_editar
        ].index[0]

        nodos_df.loc[idx,"nombre"] = nombre_edit
        nodos_df.loc[idx,"tipo"] = tipo_edit
        nodos_df.loc[idx,"capacidad_ton"] = capacidad_edit
        nodos_df.loc[idx,"demanda_ton"] = demanda_edit
        nodos_df.loc[idx,"departamento"] = departamento_edit

        nodos_df.to_csv(
            "Data/nodos.csv",
            index=False
        )

        st.success("Nodo actualizado")

    st.subheader("Eliminar Nodo")

    id_borrar = st.selectbox(
        "Nodo a eliminar",
        nodos_df["id"],
        key="borrar_nodo"
    )

    if st.button("Eliminar Nodo"):

        nodos_df = nodos_df[
            nodos_df["id"] != id_borrar
        ]

        aristas_df = aristas_df[
            (aristas_df["origen"] != id_borrar)
            &
            (aristas_df["destino"] != id_borrar)
        ]

        nodos_df.to_csv(
            "Data/nodos.csv",
            index=False
        )

        aristas_df.to_csv(
            "Data/aristas.csv",
            index=False
        )

        st.success(
            "Nodo y aristas asociadas eliminados"
        )
    st.subheader("Agregar Arista")

    with st.form("nueva_arista"):

        origen = st.selectbox(
            "Origen",
            nodos_df["id"]
        )

        destino = st.selectbox(
            "Destino",
            nodos_df["id"]
        )

        distancia = st.number_input(
            "Distancia km",
            min_value=0.0
        )

        costo = st.number_input(
            "Costo ton/km",
            min_value=0.0
        )

        guardar = st.form_submit_button(
            "Guardar Arista"
        )

        if guardar:

            nueva = pd.DataFrame([
                {
                    "origen": origen,
                    "destino": destino,
                    "distancia_km": distancia,
                    "costo_ton_km": costo,
                    "capacidad_pequeño_ton": 5,
                    "capacidad_grande_ton": 15,
                    "bidireccional": "no"
                }
            ])

            aristas_df = pd.concat(
                [aristas_df, nueva],
                ignore_index=True
            )

            aristas_df.to_csv(
                "Data/aristas.csv",
                index=False
            )

            st.success("Arista agregada")

    st.subheader("Editar Arista")

    aristas_df["ruta"] = (
        aristas_df["origen"]
        + " → "
        + aristas_df["destino"]
    )

    ruta = st.selectbox(
        "Seleccione Ruta",
        aristas_df["ruta"]
    )

    fila = aristas_df[
        aristas_df["ruta"] == ruta
    ].iloc[0]

    distancia_edit = st.number_input(
        "Distancia",
        value=float(fila["distancia_km"]),
        key="dist_edit"
    )

    costo_edit = st.number_input(
        "Costo",
        value=float(fila["costo_ton_km"]),
        key="cost_edit"
    )

    cap_peq_edit = st.number_input(
        "Capacidad Camión Pequeño",
        value=float(
            fila["capacidad_pequeño_ton"]
        )
    )

    cap_grande_edit = st.number_input(
        "Capacidad Camión Grande",
        value=float(
            fila["capacidad_grande_ton"]
        )
    )

    if st.button("Actualizar Arista"):

        idx = fila.name

        aristas_df.loc[
            idx,
            "distancia_km"
        ] = distancia_edit

        aristas_df.loc[
            idx,
            "costo_ton_km"
        ] = costo_edit

        aristas_df.loc[
            idx,
            "capacidad_pequeño_ton"
        ] = cap_peq_edit

        aristas_df.loc[
            idx,
            "capacidad_grande_ton"
        ] = cap_grande_edit

        aristas_df.to_csv(
            "Data/aristas.csv",
            index=False
        )

        st.success(
            "Arista actualizada"
        )
    st.subheader("Eliminar Arista")

    ruta_borrar = st.selectbox(
        "Ruta",
        aristas_df["ruta"],
        key="borrar_arista"
    )

    if st.button("Eliminar Arista"):

        aristas_df = aristas_df[
            aristas_df["ruta"] != ruta_borrar
        ]

        aristas_df.drop(
            columns=["ruta"],
            errors="ignore",
            inplace=True
        )

        aristas_df.to_csv(
            "Data/aristas.csv",
            index=False
        )

        st.success(
            "Arista eliminada"
        )
# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — ALGORITMOS DE GRAFOS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🚚 Método de Transporte":

    st.title("Método de Transporte")

    if "pedidos" not in st.session_state:
        st.session_state.pedidos = {}

    nodos_df = pd.read_csv("Data/nodos.csv")

    archivo_od = "Data/oferta_demanda.csv"

    if os.path.exists(archivo_od):

        od_df = pd.read_csv(archivo_od)

    else:

        od_df = nodos_df[
            ["id","nombre","tipo"]
        ].copy()

        od_df["oferta_ton"] = 0.0
        od_df["demanda_ton"] = 0.0

    # Actualizar nombre y tipo según nodos.csv
    for _, nodo in nodos_df.iterrows():

        id_transporte = equivalencias.get(
            nodo["id"],
            nodo["id"]
        )

        if id_transporte in od_df["id_nodo"].values:

            od_df.loc[
                od_df["id_nodo"] == id_transporte,
                "tipo"
            ] = nodo["tipo"]

            od_df.loc[
                od_df["id_nodo"] == id_transporte,
                "nombre"
            ] = nodo["nombre"]
        else:

            nueva_fila = pd.DataFrame([{
                "id_nodo": id_transporte,
                "nombre": nodo["nombre"],
                "tipo": nodo["tipo"],
                "oferta": 0,
                "demanda": 0
            }])

            od_df = pd.concat(
                [od_df, nueva_fila],
                ignore_index=True
            )

    # Eliminar registros de nodos borrados
    ids_validos = [
        equivalencias.get(i, i)
        for i in nodos_df["id"]
    ]

    od_df = od_df[
        od_df["id_nodo"].isin(ids_validos)
    ]

    # Forzar que los acopios nunca tengan oferta ni demanda
    od_df.loc[
        od_df["tipo"] == "acopio",
        "oferta"
    ] = 0

    od_df.loc[
        od_df["tipo"] == "acopio",
        "demanda"
    ] = 0

    # Guardar cambios
    od_df.to_csv(
        archivo_od,
        index=False
    )

    st.subheader("Oferta y Demanda Registrada")

    st.dataframe(
        od_df[
            ["id_nodo","nombre","tipo","oferta","demanda"]
        ]
    )

    st.subheader("Configurar Oferta y Demanda")

    id_nodo = st.selectbox(
        "Seleccione Nodo",
        od_df["id_nodo"]
    )

    filtro = od_df[
        od_df["id_nodo"] == id_nodo
    ]

    if filtro.empty:

        st.error(
            f"No se encontró el nodo {id_nodo}"
        )

        st.stop()

    fila = filtro.iloc[0]

    st.write(
        f"Tipo: {fila['tipo']}"
    )

    if fila["tipo"] == "destino":

        cantidad = st.number_input(
            "Cantidad solicitada",
            min_value=0.0,
            value=0.0
        )

        if st.button("Agregar Pedido"):

            st.session_state.pedidos[
                fila["id_nodo"]
            ] = cantidad

            st.success(
                f"Pedido agregado para {fila['nombre']}"
            )

    elif fila["tipo"] == "origen":

        st.info(
            f"Oferta disponible: {fila['oferta']}"
        )

    else:

        st.info(
            "Los centros de acopio no reciben pedidos."
        )

    st.subheader("Pedidos Registrados")

    if st.session_state.pedidos:

        pedidos_df = pd.DataFrame([
            {
                "Destino": destino,
                "Cantidad": cantidad
            }
            for destino, cantidad
            in st.session_state.pedidos.items()
        ])

        st.dataframe(pedidos_df)

    st.subheader("Validación del Problema")

    total_oferta = od_df[
        od_df["tipo"] == "origen"
    ]["oferta"].sum()

    total_demanda = sum(st.session_state.pedidos.values())

    col1,col2 = st.columns(2)

    with col1:

        st.metric(
            "Oferta Total",
            total_oferta
        )

    with col2:

        st.metric(
            "Demanda Total",
            total_demanda
        )
    st.subheader("Método Inicial")

    metodo = st.selectbox(
        "Seleccione método",
        [
            "Esquina Noroeste",
            "Costo Mínimo",
            "Aproximación de Vogel",
            "Mejor Automático"
        ]
    )

    if st.button(
        "Calcular Transporte"
    ):
        demanda = {}
        for destino, cantidad in st.session_state.pedidos.items():
            if cantidad > 0:
                demanda[destino] = cantidad

        if total_oferta == total_demanda:

            st.success(
                "Problema balanceado."
            )

        elif total_oferta > total_demanda:

            st.warning(
                "La oferta es mayor que la demanda. "
                "Se agregará un destino ficticio."
            )

        oferta = {
            fila["id_nodo"]: fila["oferta"]
            for _, fila in od_df.iterrows()
            if fila["tipo"] == "origen"
        }

        costos, rutas = construir_matriz_costos(
            G,
            oferta.keys(),
            demanda.keys(),
        )
        st.info(
            "Construyendo matriz de costos..."
        )

        st.info(
            "Calculando rutas mínimas con Dijkstra..."
        )

        st.info(
            "Generando solución inicial..."
        )
        if metodo == "Esquina Noroeste":

            solucion_inicial = esquina_noroeste(
                costos,
                oferta,
                demanda
            )

        elif metodo == "Costo Mínimo":

            solucion_inicial = costo_minimo(
                costos,
                oferta,
                demanda
            )

        elif metodo == "Aproximación de Vogel":

            solucion_inicial = vogel(
                costos,
                oferta,
                demanda
            )

        elif metodo == "Mejor Automático":

            solucion_inicial = mejor_solucion(
                costos,
                oferta,
                demanda
            )

        st.info(
            "Aplicando MODI..."
        )

        solucion_optima = optimizar_modi(
            costos,
            solucion_inicial
        )

        resultado_df = pd.DataFrame(
            solucion_optima["asignaciones"]
        )

        st.subheader(
            "Asignaciones Óptimas"
        )

        st.dataframe(
            resultado_df
        )
        
        st.metric(
            "Costo Total",
            f"${solucion_optima['costo']:,.0f}"
        )