positivas = [
    "hermosa",
    "excelente",
    "bonita",
    "limpia",
    "agradable",
    "buena",
    "increible"
]

negativas = [
    "mala",
    "horrible",
    "sucia",
    "terrible",
    "feo",
    "pésimo"
]


def analizar_sentimiento(texto):

    texto = texto.lower()

    positivos = 0
    negativos = 0

    for palabra in positivas:

        if palabra in texto:
            positivos += 1

    for palabra in negativas:

        if palabra in texto:
            negativos += 1

    if positivos > negativos:
        return "POS"

    elif negativos > positivos:
        return "NEG"

    else:
        return "NEU"
