import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import yaml
import logging
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cargar_config() -> dict:
    with open("config/configuracion.yaml", "r") as f:
        return yaml.safe_load(f)


def limpiar_texto(texto: str) -> str:
    texto = re.sub(r'(https?://|www\.)\S+', '', texto)
    texto = re.sub(r'\b(Visit|Click|Read more|Source)\b.*', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'<.*?>', '', texto)
    texto = re.sub(r'[^\w\s\.\,\!\?\-\%\$\']', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.strip()
    return texto


def preprocesar_texto(texto: str, tokenizer, max_length: int = 512) -> dict:
    texto_limpio = limpiar_texto(texto)
    tokens = tokenizer(
        texto_limpio,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return tokens


def preprocesar_noticias(noticias: list) -> list:
    config = cargar_config()
    nombre_modelo = config["modelo"]["nombre"]
    max_length = config["modelo"]["max_length"]

    logger.info(f"Cargando tokenizer de {nombre_modelo}")
    tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)

    noticias_procesadas = []
    for noticia in noticias:
        texto = noticia.get("texto_traducido", noticia.get("texto_completo", ""))
        texto_limpio = limpiar_texto(texto)
        noticia["texto_preprocesado"] = texto_limpio
        noticias_procesadas.append(noticia)

    logger.info(f"Preprocesadas {len(noticias_procesadas)} noticias")
    return noticias_procesadas


if __name__ == "__main__":
    textos_prueba = [
        "Santander rises 3% after record results!!! Visit: https://example.com",
        "The company <b>reported</b> losses of €2.3 million in Q1.",
        "Apple's stock price fell 5% amid trade war concerns."
    ]

    for texto in textos_prueba:
        limpio = limpiar_texto(texto)
        print(f"Original:  {texto}")
        print(f"Limpio:    {limpio}")
        print()