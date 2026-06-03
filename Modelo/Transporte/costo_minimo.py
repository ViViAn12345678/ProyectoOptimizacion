def costo_minimo(
    costos,
    oferta,
    demanda
):

    oferta = oferta.copy()
    demanda = demanda.copy()

    asignaciones = []
    costo_total = 0

    while True:

        mejor_origen = None
        mejor_destino = None
        menor_costo = float("inf")

        for origen in oferta:

            if oferta[origen] <= 0:
                continue

            for destino in demanda:

                if demanda[destino] <= 0:
                    continue

                costo = costos[origen][destino]

                if costo < menor_costo:

                    menor_costo = costo
                    mejor_origen = origen
                    mejor_destino = destino

        if mejor_origen is None:
            break

        cantidad = min(
            oferta[mejor_origen],
            demanda[mejor_destino]
        )

        asignaciones.append({
            "origen": mejor_origen,
            "destino": mejor_destino,
            "cantidad": cantidad,
            "costo_unitario": menor_costo
        })

        costo_total += cantidad * menor_costo

        oferta[mejor_origen] -= cantidad
        demanda[mejor_destino] -= cantidad

    return {
        "costo": costo_total,
        "asignaciones": asignaciones
    }