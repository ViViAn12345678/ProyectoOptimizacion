from Modelo.Transporte.esquina_noroeste import esquina_noroeste
from Modelo.Transporte.costo_minimo import costo_minimo
from Modelo.Transporte.vogel import vogel


def mejor_solucion(
        costos,
        oferta,
        demanda
):

    nw = esquina_noroeste(
        costos,
        oferta.copy(),
        demanda.copy()
    )

    cm = costo_minimo(
        costos,
        oferta.copy(),
        demanda.copy()
    )

    vg = vogel(
        costos,
        oferta.copy(),
        demanda.copy()
    )

    candidatos = [
        nw,
        cm,
        vg
    ]

    candidatos = [
        c for c in candidatos
        if c is not None and "costo" in c
    ]

    if not candidatos:

        raise ValueError(
            "No se encontró una solución inicial válida."
        )

    mejor = min(
        candidatos,
        key=lambda x: x["costo"]
    )

    return mejor