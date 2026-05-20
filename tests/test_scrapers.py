import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.scrapers.yahoo_scraper import obtener_noticias_yahoo, buscar_ticker
from src.scrapers.rss_scraper import obtener_noticias_rss
from src.scrapers.agregador import obtener_todas_noticias, eliminar_duplicados


def test_buscar_ticker_empresa_conocida():
    ticker = buscar_ticker("Apple")
    assert ticker is not None
    assert len(ticker) > 0


def test_yahoo_devuelve_lista():
    noticias = obtener_noticias_yahoo("Apple", max_articulos=5)
    assert isinstance(noticias, list)


def test_yahoo_estructura_correcta():
    noticias = obtener_noticias_yahoo("Apple", max_articulos=3)
    if noticias:
        campos_requeridos = ["titulo", "descripcion", "texto_completo",
                            "fuente", "url", "fecha", "empresa", "scraper"]
        for campo in campos_requeridos:
            assert campo in noticias[0], f"Falta el campo: {campo}"


def test_rss_devuelve_lista():
    noticias = obtener_noticias_rss("Apple", max_articulos=5)
    assert isinstance(noticias, list)


def test_eliminar_duplicados():
    noticias = [
        {"titulo": "Apple reports record profits", "fuente": "Reuters"},
        {"titulo": "Apple reports record profits", "fuente": "Bloomberg"},
        {"titulo": "Apple launches new iPhone", "fuente": "Reuters"},
    ]
    resultado = eliminar_duplicados(noticias)
    assert len(resultado) == 2


def test_agregador_devuelve_lista():
    noticias = obtener_todas_noticias("Apple", max_por_fuente=5)
    assert isinstance(noticias, list)


def test_agregador_sin_duplicados():
    noticias = obtener_todas_noticias("Apple", max_por_fuente=5)
    titulos = [n["titulo"].lower().strip() for n in noticias]
    assert len(titulos) == len(set(titulos))