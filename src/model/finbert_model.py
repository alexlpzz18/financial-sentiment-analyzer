import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import yaml
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ETIQUETAS = {0: "negative", 1: "neutral", 2: "positive"}
ETIQUETAS_INVERTIDAS = {"negative": 0, "neutral": 1, "positive": 2}


def cargar_config() -> dict:
    with open("config/configuracion.yaml", "r") as f:
        return yaml.safe_load(f)


def cargar_modelo_base() -> tuple:
    config = cargar_config()
    nombre_modelo = config["modelo"]["nombre"]

    logger.info(f"Cargando modelo base: {nombre_modelo}")

    tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        nombre_modelo,
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    logger.info("Modelo base cargado correctamente")
    return modelo, tokenizer


def predecir_sentimiento(texto: str, modelo, tokenizer, max_length: int = 512) -> dict:
    modelo.eval()

    inputs = tokenizer(
        texto,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = modelo(**inputs)
        logits = outputs.logits
        probabilidades = torch.softmax(logits, dim=1)[0]
        prediccion = torch.argmax(probabilidades).item()

    return {
        "sentimiento": ETIQUETAS[prediccion],
        "confianza": round(probabilidades[prediccion].item(), 4),
        "probabilidades": {
            "negative": round(probabilidades[0].item(), 4),
            "neutral": round(probabilidades[1].item(), 4),
            "positive": round(probabilidades[2].item(), 4)
        }
    }


if __name__ == "__main__":
    modelo, tokenizer = cargar_modelo_base()

    textos_prueba = [
        "The company reported record profits this quarter.",
        "The company filed for bankruptcy after massive losses.",
        "The company held its annual shareholders meeting."
    ]

    print("\n=== PREDICCIONES CON MODELO BASE (sin fine-tuning) ===\n")
    for texto in textos_prueba:
        resultado = predecir_sentimiento(texto, modelo, tokenizer)
        print(f"Texto: {texto}")
        print(f"Sentimiento: {resultado['sentimiento']} (confianza: {resultado['confianza']})")
        print(f"Probabilidades: {resultado['probabilidades']}")
        print()