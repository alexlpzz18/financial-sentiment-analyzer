import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.scrapers.agregador import obtener_todas_noticias
from src.pipeline.traductor import traducir_noticias
from src.pipeline.preprocesador import preprocesar_noticias
from src.model.finbert_model import predecir_sentimiento, ETIQUETAS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUTA_MODELO = "models/finbert_finetuned"


def cargar_modelo_finetuned() -> tuple:
    logger.info(f"Cargando modelo fine-tuned desde {RUTA_MODELO}")
    tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO)
    modelo = AutoModelForSequenceClassification.from_pretrained(RUTA_MODELO)
    modelo.eval()
    logger.info("Modelo cargado correctamente")
    return modelo, tokenizer


def analizar_empresa(empresa: str, modelo, tokenizer, max_articulos: int = 20) -> dict:
    logger.info(f"Analizando sentimiento para: {empresa}")

    noticias = obtener_todas_noticias(empresa, max_por_fuente=max_articulos)
    if not noticias:
        logger.warning(f"No se encontraron noticias para {empresa}")
        return {"empresa": empresa, "noticias": [], "resumen": None}

    noticias = traducir_noticias(noticias)
    noticias = preprocesar_noticias(noticias)

    resultados = []
    for noticia in noticias:
        texto = noticia.get("texto_preprocesado", noticia.get("texto_completo", ""))
        if not texto:
            continue

        prediccion = predecir_sentimiento(texto, modelo, tokenizer)

        resultados.append({
            "titulo": noticia["titulo"],
            "fuente": noticia["fuente"],
            "fecha": noticia["fecha"],
            "url": noticia.get("url", ""),
            "idioma_original": noticia.get("idioma_original", "en"),
            "traducido": noticia.get("traducido", False),
            "sentimiento": prediccion["sentimiento"],
            "confianza": prediccion["confianza"],
            "probabilidades": prediccion["probabilidades"],
            "scraper": noticia["scraper"]
        })

    resumen = calcular_resumen(resultados)

    logger.info(f"Análisis completado: {len(resultados)} noticias procesadas")
    return {
        "empresa": empresa,
        "noticias": resultados,
        "resumen": resumen
    }


def calcular_resumen(resultados: list) -> dict:
    if not resultados:
        return None

    total = len(resultados)
    conteos = {"positive": 0, "negative": 0, "neutral": 0}

    for r in resultados:
        conteos[r["sentimiento"]] += 1

    sentimiento_dominante = max(conteos, key=conteos.get)
    confianza_media = sum(r["confianza"] for r in resultados) / total

    return {
        "total_noticias": total,
        "positivas": conteos["positive"],
        "negativas": conteos["negative"],
        "neutrales": conteos["neutral"],
        "sentimiento_dominante": sentimiento_dominante,
        "confianza_media": round(confianza_media, 4),
        "porcentaje_positivo": round(conteos["positive"] / total * 100, 1),
        "porcentaje_negativo": round(conteos["negative"] / total * 100, 1),
        "porcentaje_neutral": round(conteos["neutral"] / total * 100, 1)
    }


if __name__ == "__main__":
    modelo, tokenizer = cargar_modelo_finetuned()

    resultado = analizar_empresa("Apple", modelo, tokenizer)

    print(f"\n=== ANÁLISIS DE {resultado['empresa'].upper()} ===")
    resumen = resultado["resumen"]
    if resumen:
        print(f"\nResumen:")
        print(f"  Total noticias: {resumen['total_noticias']}")
        print(f"  Sentimiento dominante: {resumen['sentimiento_dominante'].upper()}")
        print(f"  Positivas: {resumen['positivas']} ({resumen['porcentaje_positivo']}%)")
        print(f"  Negativas: {resumen['negativas']} ({resumen['porcentaje_negativo']}%)")
        print(f"  Neutrales: {resumen['neutrales']} ({resumen['porcentaje_neutral']}%)")
        print(f"  Confianza media: {resumen['confianza_media']}")

    print(f"\nPrimeras 3 noticias:")
    for noticia in resultado["noticias"][:3]:
        print(f"\n  [{noticia['sentimiento'].upper()}] {noticia['titulo']}")
        print(f"  Fuente: {noticia['fuente']} | Confianza: {noticia['confianza']}")