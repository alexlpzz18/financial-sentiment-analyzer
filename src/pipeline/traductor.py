import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deep_translator import GoogleTranslator
from langdetect import detect
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cargar_config() -> dict:
    with open("config/configuracion.yaml", "r") as f:
        return yaml.safe_load(f)


def detectar_idioma(texto: str) -> str:
    try:
        return detect(texto)
    except Exception:
        return "unknown"


def traducir_a_ingles(texto: str) -> str:
    idioma = detectar_idioma(texto)

    if idioma == "en":
        return texto

    try:
        logger.info(f"Traduciendo texto del {idioma} al inglés")
        traduccion = GoogleTranslator(source="auto", target="en").translate(texto)
        return traduccion
    except Exception as e:
        logger.error(f"Error al traducir: {str(e)}")
        return texto


def traducir_noticias(noticias: list) -> list:
    noticias_traducidas = []

    for noticia in noticias:
        texto_original = noticia["texto_completo"]
        idioma = detectar_idioma(texto_original)

        if idioma != "en":
            texto_traducido = traducir_a_ingles(texto_original)
            noticia["texto_traducido"] = texto_traducido
            noticia["idioma_original"] = idioma
            noticia["traducido"] = True
        else:
            noticia["texto_traducido"] = texto_original
            noticia["idioma_original"] = "en"
            noticia["traducido"] = False

        noticias_traducidas.append(noticia)

    return noticias_traducidas


if __name__ == "__main__":
    noticias_prueba = [
        {
            "titulo": "Santander sube un 3% tras resultados récord",
            "texto_completo": "Santander sube un 3% tras presentar resultados récord en el primer trimestre.",
            "fuente": "Expansión",
            "empresa": "Santander"
        },
        {
            "titulo": "Fitch upgrades Santander outlook to positive",
            "texto_completo": "Fitch upgrades Santander outlook to positive citing strong capital ratios.",
            "fuente": "Reuters",
            "empresa": "Santander"
        }
    ]

    resultado = traducir_noticias(noticias_prueba)
    for noticia in resultado:
        print(f"\nOriginal ({noticia['idioma_original']}): {noticia['texto_completo']}")
        print(f"Traducido: {noticia['texto_traducido']}")
        print(f"Traducido: {noticia['traducido']}")