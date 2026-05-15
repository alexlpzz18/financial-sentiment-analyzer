import yfinance as yf
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def buscar_ticker(empresa: str) -> str:
    try:
        resultados = yf.Search(empresa).quotes
        if resultados:
            ticker = resultados[0]["symbol"]
            logger.info(f"Ticker encontrado para '{empresa}': {ticker}")
            return ticker
        else:
            logger.warning(f"No se encontró ticker para '{empresa}'")
            return empresa.upper()
    except Exception as e:
        logger.error(f"Error buscando ticker: {str(e)}")
        return empresa.upper()


def obtener_noticias_yahoo(empresa: str, max_articulos: int = 20) -> list:
    ticker = buscar_ticker(empresa)

    try:
        logger.info(f"Buscando noticias en Yahoo Finance para: {empresa} ({ticker})")

        activo = yf.Ticker(ticker)
        noticias_raw = activo.news

        articulos = []
        for noticia in noticias_raw[:max_articulos]:
            contenido = noticia.get("content", noticia)
    
            titulo = contenido.get("title", "")
            if not titulo:
                continue

            articulos.append({
                "titulo": titulo,
                "descripcion": contenido.get("summary", titulo),
                "texto_completo": f"{titulo}. {contenido.get('summary', '')}",
                "fuente": contenido.get("provider", {}).get("displayName", "Yahoo Finance"),
                "url": contenido.get("canonicalUrl", {}).get("url", ""),
                "fecha": contenido.get("pubDate", datetime.now().isoformat()),
                "empresa": empresa,
                "ticker": ticker,
                "scraper": "yahoo"
            })

        logger.info(f"Encontrados {len(articulos)} artículos para {empresa}")
        return articulos

    except Exception as e:
        logger.error(f"Error al obtener noticias de Yahoo Finance: {str(e)}")
        return []


if __name__ == "__main__":
    noticias = obtener_noticias_yahoo("Santander")
    for noticia in noticias[:3]:
        print(f"\nTítulo: {noticia['titulo']}")
        print(f"Fuente: {noticia['fuente']}")
        print(f"Fecha: {noticia['fecha']}")
        