import networkx as nx

def construir_matriz_costos(
    G,
    origenes,
    destinos
):

    costos = {}
    rutas = {}

    for origen in origenes:

        costos[origen] = {}
        rutas[origen] = {}

        for destino in destinos:

            try:

                ruta = nx.shortest_path(
                    G,
                    source=origen,
                    target=destino,
                    weight="peso"
                )

                costo = nx.shortest_path_length(
                    G,
                    source=origen,
                    target=destino,
                    weight="peso"
                )

                costos[origen][destino] = costo
                rutas[origen][destino] = ruta

            except nx.NetworkXNoPath:

                costos[origen][destino] = float("inf")
                rutas[origen][destino] = []

    return costos, rutas