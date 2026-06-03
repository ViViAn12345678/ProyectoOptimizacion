from Modelo.algoritmos import dijkstra

def construir_matriz_costos(
        grafo,
        origenes,
        destinos
):

    costos = {}

    for o in origenes:

        costos[o] = {}

        for d in destinos:

            ruta = dijkstra(
                grafo,
                o,
                d
            )

            costos[o][d] = ruta.costo_total

    return costos