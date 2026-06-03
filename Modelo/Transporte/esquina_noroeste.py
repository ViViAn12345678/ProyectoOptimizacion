def esquina_noroeste(
    costos,
    oferta,
    demanda
):

    oferta = oferta.copy()
    demanda = demanda.copy()

    origenes = list(oferta.keys())
    destinos = list(demanda.keys())

    i = 0
    j = 0

    asignaciones = []
    costo_total = 0

    while i < len(origenes) and j < len(destinos):

        origen = origenes[i]
        destino = destinos[j]

        cantidad = min(
            oferta[origen],
            demanda[destino]
        )

        if cantidad > 0:

            asignaciones.append({
                "origen": origen,
                "destino": destino,
                "cantidad": cantidad,
                "costo_unitario": costos[origen][destino]
            })

            costo_total += (
                cantidad *
                costos[origen][destino]
            )

        oferta[origen] -= cantidad
        demanda[destino] -= cantidad

        if oferta[origen] == 0:
            i += 1

        if demanda[destino] == 0:
            j += 1

    return {
        "costo": costo_total,
        "asignaciones": asignaciones
    }