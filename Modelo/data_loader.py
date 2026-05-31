import pandas as pd
import networkx as nx
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "Data"   # ← "Data" con mayúscula (igual que la carpeta real)

def cargar_red(ruta_nodos=None, ruta_aristas=None) -> nx.DiGraph:
    ruta_nodos   = ruta_nodos   or DATA_DIR / "nodos.csv"
    ruta_aristas = ruta_aristas or DATA_DIR / "aristas.csv"

    nodos_df   = pd.read_csv(ruta_nodos)
    aristas_df = pd.read_csv(ruta_aristas)

    G = nx.DiGraph()

    for _, r in nodos_df.iterrows():
        G.add_node(r["id"],
                   nombre=r["nombre"],
                   tipo=r["tipo"],
                   capacidad=r["capacidad_ton"],
                   demanda=r["demanda_ton"],
                   departamento=r["departamento"])

    for _, r in aristas_df.iterrows():
        attrs = dict(
            distancia      = r["distancia_km"],
            costo_unitario = r["costo_ton_km"],
            capacidad      = r["capacidad_grande_ton"],   # capacidad máxima de la arista (para flujo máximo)
            cap_pequeño    = r["capacidad_pequeño_ton"],  # necesario en modelo_pl construir_modelo no, pero útil para algoritmos
            cap_grande     = r["capacidad_grande_ton"],
            peso           = r["costo_ton_km"] * r["distancia_km"]
        )
        G.add_edge(r["origen"], r["destino"], **attrs)
        if str(r.get("bidireccional", "no")).strip().lower() == "si":
            G.add_edge(r["destino"], r["origen"], **attrs)

    return G
