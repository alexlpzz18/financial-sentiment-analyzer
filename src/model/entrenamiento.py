import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import wandb
import yaml
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
from datasets import load_dataset

from src.model.finbert_model import ETIQUETAS_INVERTIDAS, cargar_modelo_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cargar_config() -> dict:
    with open("config/configuracion.yaml", "r") as f:
        return yaml.safe_load(f)


class FinancialDataset(Dataset):
    def __init__(self, datos, tokenizer, max_length):
        self.datos = datos
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        item = self.datos[idx]
        texto = item["sentence"] if "sentence" in item else item["text"]
        etiqueta = item["label"]

        encoding = self.tokenizer(
            texto,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(etiqueta, dtype=torch.long)
        }


def calcular_class_weights(config: dict) -> torch.Tensor:
    pesos = config["class_weights"]
    return torch.tensor([
        pesos["negative"],
        pesos["neutral"],
        pesos["positive"]
    ], dtype=torch.float)


def entrenar():
    config = cargar_config()

    torch.manual_seed(config["entrenamiento"]["semilla"])
    np.random.seed(config["entrenamiento"]["semilla"])

    wandb.init(
        project="financial-sentiment-analyzer",
        config={
            "modelo": config["modelo"]["nombre"],
            "epocas": config["entrenamiento"]["epocas"],
            "learning_rate": config["entrenamiento"]["learning_rate"],
            "batch_size": config["entrenamiento"]["batch_size"],
            "dataset": "FinanceMTEB/financial_phrasebank"
        }
    )

    logger.info("Cargando dataset...")
    dataset = load_dataset("FinanceMTEB/financial_phrasebank")

    modelo, tokenizer = cargar_modelo_base()

    max_length = config["modelo"]["max_length"]
    batch_size = config["entrenamiento"]["batch_size"]

    dataset_train = FinancialDataset(dataset["train"], tokenizer, max_length)
    dataset_test = FinancialDataset(dataset["test"], tokenizer, max_length)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    class_weights = calcular_class_weights(config)
    criterio = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizador = AdamW(
        modelo.parameters(),
        lr=config["entrenamiento"]["learning_rate"],
        weight_decay=0.01
    )

    total_steps = len(loader_train) * config["entrenamiento"]["epocas"]
    scheduler = get_linear_schedule_with_warmup(
        optimizador,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    epocas = config["entrenamiento"]["epocas"]

    for epoca in range(epocas):
        modelo.train()
        perdida_total = 0

        for batch in loader_train:
            optimizador.zero_grad()

            outputs = modelo(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            perdida = criterio(outputs.logits, batch["label"])
            perdida.backward()

            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)

            optimizador.step()
            scheduler.step()

            perdida_total += perdida.item()

        perdida_media = perdida_total / len(loader_train)

        modelo.eval()
        predicciones = []
        etiquetas_reales = []

        with torch.no_grad():
            for batch in loader_test:
                outputs = modelo(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                preds = torch.argmax(outputs.logits, dim=1)
                predicciones.extend(preds.numpy())
                etiquetas_reales.extend(batch["label"].numpy())

        f1 = f1_score(etiquetas_reales, predicciones, average="weighted")

        wandb.log({
            "epoca": epoca + 1,
            "perdida_train": perdida_media,
            "f1_weighted": f1
        })

        logger.info(f"Época {epoca+1}/{epocas} - Loss: {perdida_media:.4f} - F1: {f1:.4f}")

    logger.info("\n=== REPORTE FINAL ===")
    print(classification_report(
        etiquetas_reales,
        predicciones,
        target_names=["negative", "neutral", "positive"]
    ))

    os.makedirs("models", exist_ok=True)
    modelo.save_pretrained("models/finbert_finetuned")
    tokenizer.save_pretrained("models/finbert_finetuned")
    logger.info("Modelo guardado en models/finbert_finetuned")

    wandb.finish()


if __name__ == "__main__":
    entrenar()