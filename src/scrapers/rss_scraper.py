import feedparser
import logging
import yaml
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cargar_feeds() -> dict:
    with open("config/configuracion.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config.get("feeds_rss", {})


def obtener_noticias_rss(empresa: str, max_articulos: int = 20) -> list:
    feeds = cargar_feeds()
    articulos = []

    for nombre_feed, url_feed in feeds.items():
        try:
            logger.info(f"Leyendo feed: {nombre_feed}")
            feed = feedparser.parse(url_feed)

            for entrada in feed.entries:
                titulo = entrada.get("title", "")
                descripcion = entrada.get("summary", "")
                texto = f"{titulo} {descripcion}".lower()

                if empresa.lower() in texto:
                    fecha = entrada.get("published", datetime.now().isoformat())

                    articulos.append({
                        "titulo": titulo,
                        "descripcion": descripcion,
                        "texto_completo": f"{titulo}. {descripcion}",
                        "fuente": nombre_feed,
                        "url": entrada.get("link", ""),
                        "fecha": fecha,
                        "empresa": empresa,
                        "scraper": "rss"
                    })

                    if len(articulos) >= max_articulos:
                        return articulos

        except Exception as e:
            logger.error(f"Error leyendo feed {nombre_feed}: {str(e)}")
            continue

    logger.info(f"Encontrados {len(articulos)} artículos RSS sobre {empresa}")
    return articulos


if __name__ == "__main__":
    noticias = obtener_noticias_rss("Santander")
    for noticia in noticias[:3]:
        print(f"\nTítulo: {noticia['titulo']}")
        print(f"Fuente: {noticia['fuente']}")
        print(f"Fecha: {noticia['fecha']}")