import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.model.finbert_model import predecir_sentimiento, ETIQUETAS

RUTA_MODELO = "models/finbert_finetuned"


@pytest.fixture(scope="module")
def modelo_y_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO)
    modelo = AutoModelForSequenceClassification.from_pretrained(RUTA_MODELO)
    modelo.eval()
    return modelo, tokenizer


def test_modelo_carga_correctamente(modelo_y_tokenizer):
    modelo, tokenizer = modelo_y_tokenizer
    assert modelo is not None
    assert tokenizer is not None


def test_prediccion_devuelve_estructura_correcta(modelo_y_tokenizer):
    modelo, tokenizer = modelo_y_tokenizer
    resultado = predecir_sentimiento(
        "The company reported record profits.",
        modelo, tokenizer
    )
    assert "sentimiento" in resultado
    assert "confianza" in resultado
    assert "probabilidades" in resultado


def test_sentimiento_es_valido(modelo_y_tokenizer):
    modelo, tokenizer = modelo_y_tokenizer
    resultado = predecir_sentimiento(
        "The company reported record profits.",
        modelo, tokenizer
    )
    assert resultado["sentimiento"] in ["positive", "negative", "neutral"]


def test_confianza_entre_0_y_1(modelo_y_tokenizer):
    modelo, tokenizer = modelo_y_tokenizer
    resultado = predecir_sentimiento(
        "The company reported record profits.",
        modelo, tokenizer
    )
    assert 0 <= resultado["confianza"] <= 1


def test_probabilidades_suman_1(modelo_y_tokenizer):
    modelo, tokenizer = modelo_y_tokenizer
    resultado = predecir_sentimiento(
        "The company reported record profits.",
        modelo, tokenizer
    )
    suma = sum(resultado["probabilidades"].values())
    assert abs(suma - 1.0) < 0.01


def test_noticia_positiva(modelo_y_tokenizer):
    modelo, tokenizer = modelo_y_tokenizer
    resultado = predecir_sentimiento(
        "The company reported record profits and raised its dividend.",
        modelo, tokenizer
    )
    assert resultado["sentimiento"] == "positive"


def test_noticia_negativa(modelo_y_tokenizer):
    modelo, tokenizer = modelo_y_tokenizer
    resultado = predecir_sentimiento(
        "The company reported a dramatic decline in profits and issued a profit warning.",
        modelo, tokenizer
    )
    assert resultado["sentimiento"] == "negative"