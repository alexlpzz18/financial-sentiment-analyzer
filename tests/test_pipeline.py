import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.pipeline.traductor import detectar_idioma, traducir_a_ingles, traducir_noticias
from src.pipeline.preprocesador import limpiar_texto


def test_detectar_idioma_ingles():
    idioma = detectar_idioma("The company reported record profits this quarter.")
    assert idioma == "en"


def test_detectar_idioma_espanol():
    idioma = detectar_idioma("La empresa reportó beneficios récord este trimestre.")
    assert idioma == "es"


def test_traducir_ingles_no_cambia():
    texto = "The company reported record profits."
    traducido = traducir_a_ingles(texto)
    assert traducido == texto


def test_traducir_espanol_a_ingles():
    texto = "La empresa reportó beneficios récord."
    traducido = traducir_a_ingles(texto)
    assert isinstance(traducido, str)
    assert len(traducido) > 0


def test_traducir_noticias_añade_campos():
    noticias = [{
        "titulo": "Santander sube un 3%",
        "texto_completo": "Santander sube un 3% tras resultados récord.",
        "fuente": "Expansión",
        "empresa": "Santander"
    }]
    resultado = traducir_noticias(noticias)
    assert "texto_traducido" in resultado[0]
    assert "idioma_original" in resultado[0]
    assert "traducido" in resultado[0]


def test_limpiar_texto_elimina_urls():
    texto = "Apple rises 3%. Visit: https://example.com for more."
    limpio = limpiar_texto(texto)
    assert "https" not in limpio
    assert "example.com" not in limpio


def test_limpiar_texto_elimina_html():
    texto = "The company <b>reported</b> losses."
    limpio = limpiar_texto(texto)
    assert "<b>" not in limpio
    assert "</b>" not in limpio


def test_limpiar_texto_mantiene_porcentajes():
    texto = "Apple stock fell 5% today."
    limpio = limpiar_texto(texto)
    assert "5%" in limpio