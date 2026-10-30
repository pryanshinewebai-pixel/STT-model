#!/usr/bin/env python3
"""
Generate Google Colab Notebook for Multilingual STT Model Fine-Tuning.
"""

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "colab_multilingual_stt_training.ipynb"
DESKTOP_PATH = Path.home() / "Desktop" / "colab_multilingual_stt_training.ipynb"

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "accelerator": "GPU",
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎙️ Multilingual STT Model Fine-Tuning (English, Hindi, Gujarati)\n",
                "### Custom Speech-to-Text Model Training with 1,028 Verified Samples\n",
                "\n",
                "This notebook fine-tunes **OpenAI Whisper** on your custom dataset containing **1,028 clean audio clips (~1.5+ hours)** across **English**, **Hindi**, and **Gujarati** using an NVIDIA T4 GPU.\n",
                "\n",
                "---\n",
                "### ⚡ Quick Start:\n",
                "1. Activate GPU: **Runtime** -> **Change runtime type** -> **T4 GPU** -> **Save**.\n",
                "2. Run **Step 1** to install libraries.\n",
                "3. Run **Step 2** and upload `colab_stt_dataset.zip` from your Desktop.\n",
                "4. Click **Runtime** -> **Run all** (or press `Ctrl+F9` / `Cmd+F9`).\n",
                "5. Training takes ~10-15 minutes, after which `finetuned_stt_model.zip` will download automatically!"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Check GPU & Install Required Libraries"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Verify NVIDIA GPU is active\n",
                "!nvidia-smi\n",
                "\n",
                "# Fix DNS resolution if Colab network has a temporary hiccup\n",
                "!echo 'nameserver 8.8.8.8' | tee -a /etc/resolv.conf > /dev/null 2>&1 || true\n",
                "\n",
                "# Install audio and transformer deep learning packages with retries\n",
                "!pip install --retries 5 --timeout 60 -q transformers datasets accelerate evaluate jiwer soundfile librosa torchaudio"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Upload Dataset (`colab_stt_dataset.zip`)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import os, zipfile, shutil\n",
                "from google.colab import files\n",
                "\n",
                "zip_filename = 'colab_stt_dataset.zip'\n",
                "dataset_dir = './dataset'\n",
                "\n",
                "if not os.path.exists(zip_filename):\n",
                "    print('📤 Please upload colab_stt_dataset.zip (from your Desktop):')\n",
                "    uploaded = files.upload()\n",
                "\n",
                "print('📦 Extracting dataset...')\n",
                "os.makedirs(dataset_dir, exist_ok=True)\n",
                "with zipfile.ZipFile(zip_filename, 'r') as z:\n",
                "    z.extractall(dataset_dir)\n",
                "\n",
                "manifest_path = os.path.join(dataset_dir, 'manifest.csv')\n",
                "wav_count = len([f for f in os.listdir(os.path.join(dataset_dir, 'wav_chunks')) if f.endswith('.wav')])\n",
                "print(f'✅ Extracted {wav_count} audio clips and manifest.csv!')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Inspect Dataset Language Distribution"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import pandas as pd\n",
                "\n",
                "df = pd.read_csv(manifest_path)\n",
                "df['full_path'] = df['audio_path'].apply(lambda p: os.path.join(dataset_dir, p))\n",
                "df = df[df['full_path'].apply(os.path.exists)].reset_index(drop=True)\n",
                "\n",
                "print('=== DATASET LANGUAGE BREAKDOWN ===')\n",
                "print(df.groupby('language')['duration'].agg(count='count', total_minutes=lambda x: round(x.sum()/60, 2)))\n",
                "print(f'\\nTotal Samples: {len(df)}')\n",
                "display(df.head())"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Setup Whisper Processor, Tokenizer & Audio Feature Extractor"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import torch\n",
                "import soundfile as sf\n",
                "from dataclasses import dataclass\n",
                "from typing import Any, Dict, List, Union\n",
                "from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration\n",
                "from sklearn.model_selection import train_test_split\n",
                "\n",
                "# Model options: 'openai/whisper-base' (~74M params, recommended) or 'openai/whisper-tiny' (~39M params)\n",
                "MODEL_NAME = 'openai/whisper-base'\n",
                "print(f'Loading Whisper Processor and Tokenizer for {MODEL_NAME}...')\n",
                "\n",
                "feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)\n",
                "tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language='Hindi', task='transcribe')\n",
                "processor = WhisperProcessor.from_pretrained(MODEL_NAME, language='Hindi', task='transcribe')\n",
                "\n",
                "# Split 90% Train / 10% Validation\n",
                "train_df, val_df = train_test_split(df, test_size=0.10, random_state=42, stratify=df['language'])\n",
                "print(f'Train clips: {len(train_df)} | Validation clips: {len(val_df)}')\n",
                "\n",
                "class STTDataset(torch.utils.data.Dataset):\n",
                "    def __init__(self, data_frame, processor):\n",
                "        self.df = data_frame.reset_index(drop=True)\n",
                "        self.processor = processor\n",
                "\n",
                "    def __len__(self):\n",
                "        return len(self.df)\n",
                "\n",
                "    def __getitem__(self, idx):\n",
                "        row = self.df.iloc[idx]\n",
                "        audio, sr = sf.read(row['full_path'], dtype='float32')\n",
                "        if audio.ndim > 1:\n",
                "            audio = audio.mean(axis=1)\n",
                "        \n",
                "        # 80-channel Log-Mel spectrogram\n",
                "        input_features = self.processor.feature_extractor(audio, sampling_rate=sr).input_features[0]\n",
                "        # Tokenize with max_length=448 (Whisper decoder max target positions limit)\n",
                "        labels = self.processor.tokenizer(row['transcript'], truncation=True, max_length=448).input_ids\n",
                "        return {'input_features': input_features, 'labels': labels}\n",
                "\n",
                "train_dataset = STTDataset(train_df, processor)\n",
                "val_dataset = STTDataset(val_df, processor)\n",
                "\n",
                "@dataclass\n",
                "class DataCollatorSpeechSeq2SeqWithPadding:\n",
                "    processor: Any\n",
                "    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:\n",
                "        input_features = [{'input_features': f['input_features']} for f in features]\n",
                "        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')\n",
                "        \n",
                "        label_features = [{'input_ids': f['labels']} for f in features]\n",
                "        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')\n",
                "        \n",
                "        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)\n",
                "        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():\n",
                "            labels = labels[:, 1:]\n",
                "        \n",
                "        # Whisper max target positions guard (strictly <= 448)\n",
                "        if labels.shape[1] > 448:\n",
                "            labels = labels[:, :448]\n",
                "            \n",
                "        batch['labels'] = labels\n",
                "        return batch\n",
                "\n",
                "data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)\n",
                "print('✅ Data pipeline and collator ready!')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Model Setup & Evaluation Metrics (WER / CER)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import evaluate\n",
                "wer_metric = evaluate.load('wer')\n",
                "cer_metric = evaluate.load('cer')\n",
                "\n",
                "def compute_metrics(pred):\n",
                "    pred_ids = pred.predictions\n",
                "    label_ids = pred.label_ids\n",
                "    label_ids[label_ids == -100] = tokenizer.pad_token_id\n",
                "    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)\n",
                "    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)\n",
                "    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)\n",
                "    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)\n",
                "    return {'wer': wer, 'cer': cer}\n",
                "\n",
                "model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)\n",
                "model.config.forced_decoder_ids = None\n",
                "model.config.suppress_tokens = []\n",
                "model.config.use_cache = False\n",
                "\n",
                "print('✅ Model loaded and ready for fine-tuning!')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Fine-Tune Model with Seq2SeqTrainer (GPU Accelerated 🚀)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer\n",
                "\n",
                "training_args = Seq2SeqTrainingArguments(\n",
                "    output_dir='./whisper_finetuned',\n",
                "    per_device_train_batch_size=8,\n",
                "    gradient_accumulation_steps=2,\n",
                "    learning_rate=1e-4,\n",
                "    warmup_steps=100,\n",
                "    num_train_epochs=10,\n",
                "    gradient_checkpointing=True,\n",
                "    fp16=True,\n",
                "    eval_strategy='epoch',\n",
                "    save_strategy='epoch',\n",
                "    per_device_eval_batch_size=8,\n",
                "    predict_with_generate=True,\n",
                "    generation_max_length=225,\n",
                "    logging_steps=25,\n",
                "    report_to=['tensorboard'],\n",
                "    load_best_model_at_end=True,\n",
                "    metric_for_best_model='wer',\n",
                "    greater_is_better=False,\n",
                ")\n",
                "\n",
                "trainer = Seq2SeqTrainer(\n",
                "    args=training_args,\n",
                "    model=model,\n",
                "    train_dataset=train_dataset,\n",
                "    eval_dataset=val_dataset,\n",
                "    data_collator=data_collator,\n",
                "    compute_metrics=compute_metrics,\n",
                "    processing_class=processor.feature_extractor,\n",
                ")\n",
                "\n",
                "print('🚀 Starting Training on NVIDIA T4 GPU...')\n",
                "trainer.train()"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 7: Test Predictions on English, Hindi & Gujarati Audio"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "print('=== MODEL PREDICTIONS ON VALIDATION AUDIOS ===\\n')\n",
                "for lang in ['en', 'hi', 'gu']:\n",
                "    sample_rows = val_df[val_df['language'] == lang].head(2)\n",
                "    for _, r in sample_rows.iterrows():\n",
                "        audio, sr = sf.read(r['full_path'], dtype='float32')\n",
                "        if audio.ndim > 1:\n",
                "            audio = audio.mean(axis=1)\n",
                "        inputs = processor(audio, sampling_rate=sr, return_tensors='pt').input_features.to('cuda')\n",
                "        with torch.no_grad():\n",
                "            predicted_ids = model.generate(inputs)\n",
                "        pred_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]\n",
                "        print(f'[{lang.upper()}] File: {r[\"file_name\"]}')\n",
                "        print(f'  Actual Audio Text: {r[\"transcript\"]}')\n",
                "        print(f'  Model Predicted  : {pred_text}\\n')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 8: Save & Auto-Download Fine-Tuned Model (`finetuned_stt_model.zip`)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "SAVE_DIR = './finetuned_stt_model'\n",
                "print(f'Saving trained model weights and tokenizer to {SAVE_DIR}...')\n",
                "model.save_pretrained(SAVE_DIR)\n",
                "processor.save_pretrained(SAVE_DIR)\n",
                "\n",
                "output_zip = 'finetuned_stt_model.zip'\n",
                "print(f'Packaging {output_zip}...')\n",
                "shutil.make_archive('finetuned_stt_model', 'zip', SAVE_DIR)\n",
                "\n",
                "print('🎉 Training complete! Downloading your trained model now...')\n",
                "files.download(output_zip)"
            ],
            "execution_count": None,
            "outputs": []
        }
    ]
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

shutil.copy(NOTEBOOK_PATH, DESKTOP_PATH)

print(f"✅ Google Colab Notebook created at: {NOTEBOOK_PATH}")
print(f"🖥️ Copied to Desktop at: {DESKTOP_PATH}")
