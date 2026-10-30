#!/usr/bin/env python3
"""
Generate a Kaggle-Optimized Jupyter Notebook for Whisper-Small (244M) Conversational Fine-Tuning.

Optimized for:
  - Architecture: openai/whisper-small (244M parameters)
  - Kaggle GPU: T4 x 2 or P100 (16GB VRAM with FP16 and Gradient Checkpointing)
  - Learning Rate: 1.5e-5 with Cosine warmup (prevents catastrophic forgetting)
  - Training Epochs: 8
  - Batch Size: 8 (Gradient Accumulation 2 -> Effective Batch Size 16)
  - Auto-extracts conversational_stt_dataset.zip
  - Auto-zips final model to /kaggle/working/finetuned_whisper_small.zip for 1-click download
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_NB_PATH = PROJECT_ROOT / "kaggle_whisper_small_training.ipynb"
DESKTOP_NB_PATH = Path.home() / "Desktop" / "kaggle_whisper_small_training.ipynb"

kaggle_notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎙️ Kaggle: Whisper-Small (244M) High-Accuracy Multilingual STT Training\n",
                "### Professional Fine-Tuning for Gujarati, Hindi & English Conversational Speech\n",
                "\n",
                "This notebook upgrades the model to **`openai/whisper-small` (244M parameters)**\n",
                "and trains on our curated **Pristine Conversational Dataset**.\n",
                "\n",
                "---\n",
                "### ⚡ IMPORTANT KAGGLE SETUP (Must do before running):\n",
                "1. **Enable Internet**: Right sidebar ➔ **Notebook options** ➔ Toggle **Internet: ON**.\n",
                "2. **Enable GPU**: Right sidebar ➔ **Accelerator** ➔ Select **GPU T4 x 2** or **GPU P100**.\n",
                "3. **Upload Dataset**: Upload `conversational_stt_dataset.zip` via **+ Add Data** or upload directly into `/kaggle/working`.\n",
                "4. **Run All Cells**: Click **Run All**! Training takes ~40-45 minutes on GPU."
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
                "# Install speech and transformer packages\n",
                "!pip install --timeout 60 --retries 5 -q transformers datasets accelerate soundfile librosa torchaudio\n",
                "print('✅ Required libraries installed successfully!')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Locate & Extract Conversational Dataset"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import os, zipfile, glob, shutil\n",
                "\n",
                "WORKING_DIR = '/kaggle/working'\n",
                "dataset_dir = os.path.join(WORKING_DIR, 'dataset')\n",
                "os.makedirs(dataset_dir, exist_ok=True)\n",
                "\n",
                "# Look for zip file in /kaggle/input or /kaggle/working\n",
                "zip_candidates = glob.glob('/kaggle/input/**/conversational_stt_dataset*.zip', recursive=True) + \\\n",
                "                 glob.glob('/kaggle/working/**/conversational_stt_dataset*.zip', recursive=True) + \\\n",
                "                 glob.glob('/kaggle/input/**/colab_stt_dataset*.zip', recursive=True) + \\\n",
                "                 glob.glob('/kaggle/working/**/colab_stt_dataset*.zip', recursive=True) + \\\n",
                "                 glob.glob('/kaggle/input/**/*.zip', recursive=True)\n",
                "\n",
                "target_zip = None\n",
                "for z in zip_candidates:\n",
                "    if 'stt' in z.lower() or 'dataset' in z.lower() or 'conversational' in z.lower():\n",
                "        target_zip = z\n",
                "        break\n",
                "\n",
                "if target_zip and os.path.exists(target_zip):\n",
                "    print(f'📦 Found dataset zip at: {target_zip}')\n",
                "    print('Extracting files...')\n",
                "    with zipfile.ZipFile(target_zip, 'r') as zf:\n",
                "        zf.extractall(dataset_dir)\n",
                "    print('✅ Extraction complete!')\n",
                "else:\n",
                "    print('Checking if files are already extracted...')\n",
                "\n",
                "# Find manifest.csv\n",
                "manifest_candidates = glob.glob(f'{dataset_dir}/**/manifest*.csv', recursive=True) + \\\n",
                "                      glob.glob('/kaggle/input/**/manifest*.csv', recursive=True)\n",
                "\n",
                "if not manifest_candidates:\n",
                "    raise FileNotFoundError('❌ manifest.csv not found! Please upload conversational_stt_dataset.zip')\n",
                "\n",
                "manifest_path = manifest_candidates[0]\n",
                "manifest_dir = os.path.dirname(manifest_path)\n",
                "print(f'✅ Found manifest at: {manifest_path}')\n",
                "print(f'Dataset root directory: {manifest_dir}')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Load & Verify Dataset Manifest"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import pandas as pd\n",
                "\n",
                "df = pd.read_csv(manifest_path)\n",
                "print(f'Total samples in dataset: {len(df):,}')\n",
                "print(df.head(5)[['audio_path', 'transcript', 'language', 'duration']])\n",
                "\n",
                "# Resolve absolute audio paths\n",
                "def resolve_audio_path(p):\n",
                "    p_str = str(p).strip()\n",
                "    if os.path.exists(p_str):\n",
                "        return p_str\n",
                "    fname = os.path.basename(p_str)\n",
                "    c1 = os.path.join(manifest_dir, 'wav_chunks', fname)\n",
                "    if os.path.exists(c1):\n",
                "        return c1\n",
                "    c2 = os.path.join(manifest_dir, fname)\n",
                "    if os.path.exists(c2):\n",
                "        return c2\n",
                "    matches = glob.glob(f'{manifest_dir}/**/{fname}', recursive=True)\n",
                "    if matches:\n",
                "        return matches[0]\n",
                "    return p_str\n",
                "\n",
                "df['resolved_path'] = df['audio_path'].apply(resolve_audio_path)\n",
                "valid_df = df[df['resolved_path'].apply(os.path.exists)].copy()\n",
                "print(f'✅ Valid audio clips ready for training: {len(valid_df):,} / {len(df):,}')\n",
                "\n",
                "print('\\nBreakdown by Language:')\n",
                "for lang, count in valid_df['language'].value_counts().items():\n",
                "    dur = valid_df[valid_df['language'] == lang]['duration'].sum()\n",
                "    print(f'  - {lang.upper()}: {count:,} clips ({dur/3600:.2f} Hours)')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Load Whisper-Small Processor & Define Dataset Loader"
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
                "from transformers import WhisperProcessor, WhisperForConditionalGeneration\n",
                "from torch.utils.data import Dataset\n",
                "\n",
                "MODEL_NAME = 'openai/whisper-small'\n",
                "print(f'Loading {MODEL_NAME} Processor...')\n",
                "processor = WhisperProcessor.from_pretrained(MODEL_NAME)\n",
                "\n",
                "class STTConversationalDataset(Dataset):\n",
                "    def __init__(self, df, processor):\n",
                "        self.df = df.reset_index(drop=True)\n",
                "        self.processor = processor\n",
                "\n",
                "    def __len__(self):\n",
                "        return len(self.df)\n",
                "\n",
                "    def __getitem__(self, idx):\n",
                "        row = self.df.iloc[idx]\n",
                "        audio_path = row['resolved_path']\n",
                "        transcript = str(row['transcript']).strip()\n",
                "\n",
                "        audio_data, sr = sf.read(audio_path)\n",
                "        if audio_data.ndim > 1:\n",
                "            audio_data = audio_data.mean(axis=1)\n",
                "\n",
                "        input_features = self.processor.feature_extractor(\n",
                "            audio_data, sampling_rate=16000, return_tensors='pt'\n",
                "        ).input_features[0]\n",
                "\n",
                "        labels = self.processor.tokenizer(transcript).input_ids\n",
                "        return {'input_features': input_features, 'labels': labels}\n",
                "\n",
                "@dataclass\n",
                "class DataCollatorSpeechSeq2SeqWithPadding:\n",
                "    processor: Any\n",
                "    decoder_start_token_id: int\n",
                "\n",
                "    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:\n",
                "        input_features = [{'input_features': f['input_features']} for f in features]\n",
                "        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')\n",
                "\n",
                "        label_features = [{'input_ids': f['labels']} for f in features]\n",
                "        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')\n",
                "\n",
                "        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)\n",
                "        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():\n",
                "            labels = labels[:, 1:]\n",
                "\n",
                "        batch['labels'] = labels\n",
                "        return batch\n",
                "\n",
                "# 90% Train / 10% Validation Split\n",
                "train_df = valid_df.sample(frac=0.90, random_state=42)\n",
                "val_df = valid_df.drop(train_df.index)\n",
                "\n",
                "train_dataset = STTConversationalDataset(train_df, processor)\n",
                "val_dataset = STTConversationalDataset(val_df, processor)\n",
                "data_collator = DataCollatorSpeechSeq2SeqWithPadding(\n",
                "    processor=processor,\n",
                "    decoder_start_token_id=processor.tokenizer.bos_token_id\n",
                ")\n",
                "print(f'Train samples: {len(train_dataset):,} | Validation samples: {len(val_dataset):,}')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Initialize Whisper-Small Model (244M Params)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "print(f'🚀 Initializing {MODEL_NAME} with 244M parameters...')\n",
                "model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)\n",
                "model.config.forced_decoder_ids = None\n",
                "model.config.suppress_tokens = []\n",
                "model.config.use_cache = False\n",
                "\n",
                "total_params = sum(p.numel() for p in model.parameters())\n",
                "print(f'✅ Model successfully loaded! Total parameters: {total_params:,}')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Professional Fine-Tuning Pipeline (Gentle LR: 1.5e-5, Cosine Warmup, 8 Epochs)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperForConditionalGeneration\n",
                "\n",
                "# Defensive check: re-initialize model if kernel restarted\n",
                "if 'model' not in globals() or model is None:\n",
                "    MODEL_NAME = 'openai/whisper-small'\n",
                "    print(f'Loading {MODEL_NAME} for training...')\n",
                "    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)\n",
                "    model.config.forced_decoder_ids = None\n",
                "    model.config.suppress_tokens = []\n",
                "    model.config.use_cache = False\n",
                "\n",
                "OUTPUT_MODEL_DIR = '/kaggle/working/whisper_small_finetuned'\n",
                "\n",
                "# Hyperparameters tailored to prevent catastrophic forgetting\n",
                "training_args = Seq2SeqTrainingArguments(\n",
                "    output_dir=OUTPUT_MODEL_DIR,\n",
                "    per_device_train_batch_size=8,\n",
                "    gradient_accumulation_steps=2,  # Effective batch size = 16\n",
                "    learning_rate=1.5e-5,           # Gentle and stable\n",
                "    warmup_steps=150,\n",
                "    lr_scheduler_type='cosine',\n",
                "    weight_decay=0.01,\n",
                "    num_train_epochs=8,\n",
                "    gradient_checkpointing=True,    # Fits 244M in 16GB VRAM\n",
                "    fp16=True,                      # Fast GPU acceleration\n",
                "    eval_strategy='epoch',\n",
                "    save_strategy='epoch',\n",
                "    per_device_eval_batch_size=8,\n",
                "    predict_with_generate=False,    # Fast validation loss tracking\n",
                "    logging_steps=25,\n",
                "    load_best_model_at_end=True,\n",
                "    metric_for_best_model='loss',\n",
                "    greater_is_better=False,\n",
                "    save_total_limit=2,\n",
                "    report_to=['tensorboard'],\n",
                ")\n",
                "\n",
                "trainer = Seq2SeqTrainer(\n",
                "    args=training_args,\n",
                "    model=model,\n",
                "    train_dataset=train_dataset,\n",
                "    eval_dataset=val_dataset,\n",
                "    data_collator=data_collator,\n",
                ")\n",
                "\n",
                "print('🚀 Starting Training on Kaggle GPU...')\n",
                "trainer.train()\n",
                "print('🎉 Training Complete!')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 7: Export Best Model & Create 1-Click Download ZIP"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Save best model and processor\n",
                "FINAL_SAVE_DIR = '/kaggle/working/finetuned_whisper_small'\n",
                "print(f'Saving final model to: {FINAL_SAVE_DIR}...')\n",
                "trainer.save_model(FINAL_SAVE_DIR)\n",
                "processor.save_pretrained(FINAL_SAVE_DIR)\n",
                "\n",
                "# Package into 1-click download zip\n",
                "ZIP_OUTPUT = '/kaggle/working/finetuned_whisper_small.zip'\n",
                "print(f'Packaging into {ZIP_OUTPUT}...')\n",
                "with zipfile.ZipFile(ZIP_OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
                "    for root, dirs, files in os.walk(FINAL_SAVE_DIR):\n",
                "        for file in files:\n",
                "            fpath = os.path.join(root, file)\n",
                "            arcname = os.path.relpath(fpath, FINAL_SAVE_DIR)\n",
                "            zf.write(fpath, arcname)\n",
                "\n",
                "zip_size_mb = os.path.getsize(ZIP_OUTPUT) / (1024 * 1024)\n",
                "print(f'\\n🎉 SUCCESS! Fine-tuned Whisper-Small Model is ready!')\n",
                "print(f'📁 Download Location: {ZIP_OUTPUT} ({zip_size_mb:.2f} MB)')\n",
                "print('👉 In Kaggle: Look at the right sidebar under Output section ➔ Click the 3 dots on finetuned_whisper_small.zip ➔ Click Download!')"
            ],
            "execution_count": None,
            "outputs": []
        }
    ]
}


def export_notebook():
    print(f"Exporting Kaggle Whisper-Small Notebook to:")
    print(f"  1. {LOCAL_NB_PATH}")
    print(f"  2. {DESKTOP_NB_PATH}")

    with open(LOCAL_NB_PATH, "w", encoding="utf-8") as f:
        json.dump(kaggle_notebook, f, indent=2)

    with open(DESKTOP_NB_PATH, "w", encoding="utf-8") as f:
        json.dump(kaggle_notebook, f, indent=2)

    print("✅ Export complete! Ready to upload to Kaggle.")


if __name__ == "__main__":
    export_notebook()
