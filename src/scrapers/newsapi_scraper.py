import os
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def obtener_noticias_empresa(empresa: str, dias: int = 7, max_articulos: int = 20) -> list:
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("NEWSAPI_KEY no encontrada en el archivo .env")

    cliente = NewsApiClient(api_key=api_key)

    try:
        logger.info(f"Buscando noticias sobre: {empresa}")

        respuesta = cliente.get_top_headlines(
            q=empresa,
            page_size=min(max_articulos, 20)
        )

        articulos = []
        for articulo in respuesta.get("articles", []):
            if articulo.get("title") and articulo.get("description"):
                articulos.append({
                    "titulo": articulo["title"],
                    "descripcion": articulo["description"],
                    "texto_completo": f"{articulo['title']}. {articulo['description']}",
                    "fuente": articulo["source"]["name"],
                    "url": articulo["url"],
                    "fecha": articulo["publishedAt"],
                    "empresa": empresa,
                    "scraper": "newsapi"
                })

        logger.info(f"Encontrados {len(articulos)} artículos sobre {empresa}")
        return articulos

    except Exception as e:
        logger.error(f"Error al obtener noticias de NewsAPI: {str(e)}")
        return []


if __name__ == "__main__":
    noticias = obtener_noticias_empresa("Santander", dias=7)
    for noticia in noticias[:3]:
        print(f"\nTítulo: {noticia['titulo']}")
        print(f"Fuente: {noticia['fuente']}")
        print(f"Fecha: {noticia['fecha']}")