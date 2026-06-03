import networkx as nx

from Modelo.Transporte.equivalencias import equivalencias

def construir_matriz_costos(
    G,
    origenes,
    destinos,
):

    equivalencias_inversas = {
        v: k
        for k, v in equivalencias.items()
    }

    costos = {}
    rutas = {}

    for origen in origenes:

        costos[origen] = {}
        rutas[origen] = {}

        # Convertir P1 -> O1
        origen_grafo = equivalencias_inversas.get(
            origen,
            origen
        )

        for destino in destinos:

             # Convertir S1 -> D1
            destino_grafo = equivalencias_inversas.get(
                destino,
                destino
            )

            try:

                ruta = nx.shortest_path(
                    G,
                    source=origen_grafo,
                    target=destino_grafo,
                    weight="peso"
                )

                costo_ruta = nx.shortest_path_length(
                    G,
                    source=origen_grafo,
                    target=destino_grafo,
                    weight="peso"
                )

                costos[origen][destino] = costo_ruta
                rutas[origen][destino] = ruta

            except (
                nx.NetworkXNoPath,
                nx.NodeNotFound
            ):

                costos[origen][destino] = float("inf")
                rutas[origen][destino] = []

    return costos, rutas