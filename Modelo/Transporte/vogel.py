def vogel(
    costos,
    oferta,
    demanda
):

    oferta = oferta.copy()
    demanda = demanda.copy()

    asignaciones = []
    costo_total = 0

    while True:

        activos_origen = [
            o for o in oferta
            if oferta[o] > 0
        ]

        activos_destino = [
            d for d in demanda
            if demanda[d] > 0
        ]

        if not activos_origen or not activos_destino:
            break

        penalizaciones = []

        for origen in activos_origen:

            costos_fila = sorted([
                costos[origen][d]
                for d in activos_destino
            ])

            if len(costos_fila) >= 2:
                penalizacion = (
                    costos_fila[1]
                    - costos_fila[0]
                )
            else:
                penalizacion = costos_fila[0]

            penalizaciones.append(
                ("fila", origen, penalizacion)
            )

        for destino in activos_destino:

            costos_columna = sorted([
                costos[o][destino]
                for o in activos_origen
            ])

            if len(costos_columna) >= 2:
                penalizacion = (
                    costos_columna[1]
                    - costos_columna[0]
                )
            else:
                penalizacion = costos_columna[0]

            penalizaciones.append(
                ("columna", destino, penalizacion)
            )

        tipo, elemento, _ = max(
            penalizaciones,
            key=lambda x: x[2]
        )

        if tipo == "fila":

            origen = elemento

            destino = min(
                activos_destino,
                key=lambda d:
                costos[origen][d]
            )

        else:

            destino = elemento

            origen = min(
                activos_origen,
                key=lambda o:
                costos[o][destino]
            )

        cantidad = min(
            oferta[origen],
            demanda[destino]
        )

        asignaciones.append({
            "origen": origen,
            "destino": destino,
            "cantidad": cantidad,
            "costo_unitario":
            costos[origen][destino]
        })

        costo_total += (
            cantidad *
            costos[origen][destino]
        )

        oferta[origen] -= cantidad
        demanda[destino] -= cantidad

    return {
        "costo": costo_total,
        "asignaciones": asignaciones
    }