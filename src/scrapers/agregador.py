import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from src.scrapers.newsapi_scraper import obtener_noticias_empresa
from src.scrapers.yahoo_scraper import obtener_noticias_yahoo
from src.scrapers.rss_scraper import obtener_noticias_rss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def obtener_todas_noticias(empresa: str, max_por_fuente: int = 20) -> list:
    logger.info(f"Agregando noticias de todas las fuentes para: {empresa}")

    todas = []

    noticias_newsapi = obtener_noticias_empresa(empresa, max_articulos=max_por_fuente)
    todas.extend(noticias_newsapi)
    logger.info(f"NewsAPI: {len(noticias_newsapi)} artículos")

    noticias_yahoo = obtener_noticias_yahoo(empresa, max_articulos=max_por_fuente)
    todas.extend(noticias_yahoo)
    logger.info(f"Yahoo Finance: {len(noticias_yahoo)} artículos")

    noticias_rss = obtener_noticias_rss(empresa, max_articulos=max_por_fuente)
    todas.extend(noticias_rss)
    logger.info(f"RSS: {len(noticias_rss)} artículos")

    todas_sin_duplicados = eliminar_duplicados(todas)
    logger.info(f"Total tras eliminar duplicados: {len(todas_sin_duplicados)} artículos")

    return todas_sin_duplicados


def eliminar_duplicados(articulos: list) -> list:
    vistos = set()
    unicos = []

    for articulo in articulos:
        titulo_normalizado = articulo["titulo"].lower().strip()
        if titulo_normalizado not in vistos:
            vistos.add(titulo_normalizado)
            unicos.append(articulo)

    return unicos


if __name__ == "__main__":
    noticias = obtener_todas_noticias("Santander")
    print(f"\nTotal noticias agregadas: {len(noticias)}")
    for noticia in noticias[:5]:
        print(f"\n[{noticia['scraper'].upper()}] {noticia['titulo']}")
        print(f"Fuente: {noticia['fuente']} | Fecha: {noticia['fecha']}")