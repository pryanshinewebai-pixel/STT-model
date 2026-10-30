#!/usr/bin/env python3
"""
Generate a 100% Kaggle-Optimized Jupyter Notebook for Multilingual STT Training.

Tailored for Kaggle:
  - No `google.colab` imports (which fail on Kaggle)
  - Uses Kaggle directories: `/kaggle/working/` and `/kaggle/input/`
  - Compatible with Kaggle T4 x 2 and P100 GPUs
  - Auto-zips model into `/kaggle/working/finetuned_stt_model.zip` for 1-click Output download
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_NB_PATH = PROJECT_ROOT / "kaggle_multilingual_stt_training.ipynb"
DESKTOP_NB_PATH = Path.home() / "Desktop" / "kaggle_multilingual_stt_training.ipynb"

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
                "# 🎙️ Kaggle: Multilingual STT Model Training (20+ Hours)\n",
                "### Fine-Tuning OpenAI Whisper-Base on 7,206 Samples (Gujarati, Hindi, English)\n",
                "\n",
                "This notebook is specially optimized for **Kaggle GPU (T4 x 2 or P100)**.\n",
                "\n",
                "---\n",
                "### ⚡ IMPORTANT KAGGLE SETUP (Must do before running):\n",
                "1. **Enable Internet**: In the right sidebar ➔ **Notebook options** ➔ Toggle **Internet: ON**.\n",
                "2. **Enable GPU**: Right sidebar ➔ **Accelerator** ➔ Select **GPU T4 x 2** or **GPU P100**.\n",
                "3. **Upload Dataset**: Upload `colab_stt_dataset.zip` using the **Upload** button in `/kaggle/working` or add as a Kaggle Dataset.\n",
                "4. Run all cells! Training takes ~35-40 minutes on Kaggle GPU."
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
                "# Verify NVIDIA GPU is active on Kaggle\n",
                "!nvidia-smi\n",
                "\n",
                "# Install speech and transformer packages\n",
                "!pip install --timeout 60 --retries 5 -q transformers datasets accelerate evaluate jiwer soundfile librosa torchaudio\n",
                "print('✅ Required libraries installed successfully!')"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Locate & Extract Dataset (10+ Hours / 4,393 Clips)"
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
                "# Look for zip in /kaggle/working or /kaggle/input\n",
                "zip_candidates = glob.glob('/kaggle/input/**/colab_stt_dataset*.zip', recursive=True) + \\\n",
                "                 glob.glob('/kaggle/working/**/colab_stt_dataset*.zip', recursive=True) + \\\n",
                "                 glob.glob('/kaggle/input/**/*.zip', recursive=True)\n",
                "\n",
                "target_zip = None\n",
                "for z in zip_candidates:\n",
                "    if 'stt' in z.lower() or 'dataset' in z.lower():\n",
                "        target_zip = z\n",
                "        break\n",
                "\n",
                "if target_zip and os.path.exists(target_zip):\n",
                "    print(f'📦 Found dataset zip at: {target_zip}')\n",
                "    print('Extracting files...')\n",
                "    with zipfile.ZipFile(target_zip, 'r') as zf:\n",
                "        zf.extractall(dataset_dir)\n",
                "else:\n",
                "    # Check if files are already extracted or in input\n",
                "    input_manifests = glob.glob('/kaggle/input/**/manifest.csv', recursive=True)\n",
                "    if input_manifests:\n",
                "        dataset_dir = os.path.dirname(input_manifests[0])\n",
                "        print(f'📂 Using Kaggle Input directory: {dataset_dir}')\n",
                "    else:\n",
                "        raise FileNotFoundError('Please upload colab_stt_dataset.zip into /kaggle/working/ or add as Input Dataset!')\n",
                "\n",
                "manifest_path = os.path.join(dataset_dir, 'manifest.csv')\n",
                "if not os.path.exists(manifest_path):\n",
                "    # Check if inside a subfolder\n",
                "    sub = glob.glob(os.path.join(dataset_dir, '**/manifest.csv'), recursive=True)\n",
                "    if sub:\n",
                "        manifest_path = sub[0]\n",
                "        dataset_dir = os.path.dirname(manifest_path)\n",
                "\n",
                "print(f'✅ Dataset ready at {dataset_dir} | Manifest: {manifest_path}')"
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
                "df['full_path'] = df['audio_path'].apply(lambda p: os.path.join(dataset_dir, p) if not os.path.isabs(p) else p)\n",
                "\n",
                "# Verify all files exist\n",
                "df = df[df['full_path'].apply(os.path.exists)].reset_index(drop=True)\n",
                "\n",
                "print('=== 10+ HOURS DATASET BREAKDOWN ===')\n",
                "print(df.groupby('language')['duration'].agg(count='count', total_hours=lambda x: round(x.sum()/3600, 2)))\n",
                "total_hrs = round(df['duration'].sum() / 3600, 2)\n",
                "print(f'\\nTotal Samples: {len(df)} | Total Duration: {total_hrs} Hours')\n",
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
                "# Recommended for Kaggle GPU: 'openai/whisper-base' (~74M params)\n",
                "MODEL_NAME = 'openai/whisper-base'\n",
                "print(f'Loading Processor & Tokenizer for {MODEL_NAME}...')\n",
                "\n",
                "feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)\n",
                "tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language='Hindi', task='transcribe')\n",
                "processor = WhisperProcessor.from_pretrained(MODEL_NAME, language='Hindi', task='transcribe')\n",
                "\n",
                "# 90% Train / 10% Validation\n",
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
                "        # Peak volume normalize\n",
                "        max_amp = float(abs(audio).max()) if len(audio) > 0 else 0.0\n",
                "        if max_amp > 1e-5:\n",
                "            audio = (audio / max_amp) * 0.95\n",
                "        \n",
                "        input_features = self.processor.feature_extractor(audio, sampling_rate=sr).input_features[0]\n",
                "        labels = self.processor.tokenizer(row['transcript'], max_length=448, truncation=True).input_ids\n",
                "        return {'input_features': input_features, 'labels': labels}\n",
                "\n",
                "train_dataset = STTDataset(train_df, processor)\n",
                "val_dataset = STTDataset(val_df, processor)\n",
                "\n",
                "@dataclass\n",
                "class DataCollatorSpeechSeq2SeqWithPadding:\n",
                "    processor: Any\n",
                "\n",
                "    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:\n",
                "        input_features = [{'input_features': f['input_features']} for f in features]\n",
                "        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')\n",
                "        label_features = [{'input_ids': f['labels']} for f in features]\n",
                "        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')\n",
                "        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)\n",
                "        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():\n",
                "            labels = labels[:, 1:]\n",
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
                "# 1. Load Whisper-Base Model\n",
                "MODEL_NAME = 'openai/whisper-base'\n",
                "model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)\n",
                "model.config.forced_decoder_ids = None\n",
                "model.config.suppress_tokens = []\n",
                "model.config.use_cache = False\n",
                "print(f'✅ {MODEL_NAME} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params) loaded on Kaggle!')\n",
                "\n",
                "# 2. Setup Evaluation Metric (Optional: uses WER if available, else loss)\n",
                "compute_metrics = None\n",
                "try:\n",
                "    import evaluate\n",
                "    wer_metric = evaluate.load('wer')\n",
                "    cer_metric = evaluate.load('cer')\n",
                "    def compute_metrics(pred):\n",
                "        pred_ids = pred.predictions\n",
                "        label_ids = pred.label_ids\n",
                "        label_ids[label_ids == -100] = tokenizer.pad_token_id\n",
                "        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)\n",
                "        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)\n",
                "        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)\n",
                "        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)\n",
                "        return {'wer': wer, 'cer': cer}\n",
                "    print('✅ WER/CER metric loaded.')\n",
                "except Exception as e:\n",
                "    print(f'ℹ️ Notice: evaluate not loaded ({e}). Training will evaluate via validation loss.')\n"
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
                "# Ensure model and processor are defined even if previous cells were skipped\n",
                "if 'model' not in globals() or model is None:\n",
                "    MODEL_NAME = 'openai/whisper-base'\n",
                "    print(f'Loading {MODEL_NAME} for training...')\n",
                "    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)\n",
                "    model.config.forced_decoder_ids = None\n",
                "    model.config.suppress_tokens = []\n",
                "    model.config.use_cache = False\n",
                "\n",
                "if 'compute_metrics' not in globals():\n",
                "    compute_metrics = None\n",
                "\n",
                "OUTPUT_MODEL_DIR = '/kaggle/working/whisper_finetuned'\n",
                "\n",
                "training_args = Seq2SeqTrainingArguments(\n",
                "    output_dir=OUTPUT_MODEL_DIR,\n",
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
                "## Step 7: Save & Package Fine-Tuned Model for Easy Download"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Save best model and processor\n",
                "save_dir = '/kaggle/working/whisper_finetuned_final'\n",
                "trainer.save_model(save_dir)\n",
                "processor.save_pretrained(save_dir)\n",
                "\n",
                "# Fix tokenizer_config.json extra_special_tokens for local compatibility\n",
                "import json\n",
                "tcfg_path = os.path.join(save_dir, 'tokenizer_config.json')\n",
                "if os.path.exists(tcfg_path):\n",
                "    with open(tcfg_path) as f:\n",
                "        tcfg = json.load(f)\n",
                "    if isinstance(tcfg.get('extra_special_tokens'), list):\n",
                "        tcfg['extra_special_tokens'] = {}\n",
                "        with open(tcfg_path, 'w') as f:\n",
                "            json.dump(tcfg, f, indent=2)\n",
                "\n",
                "# Zip model into /kaggle/working/finetuned_stt_model.zip\n",
                "zip_output_path = '/kaggle/working/finetuned_stt_model.zip'\n",
                "print(f'📦 Compressing fine-tuned model into {zip_output_path}...')\n",
                "with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
                "    for root, dirs, files in os.walk(save_dir):\n",
                "        for file in files:\n",
                "            file_path = os.path.join(root, file)\n",
                "            zf.write(file_path, arcname=file)\n",
                "\n",
                "print(f'🎉 Success! Model zip created at: {zip_output_path} ({os.path.getsize(zip_output_path)/(1024*1024):.1f} MB)')\n",
                "print('👉 You can now download finetuned_stt_model.zip directly from Kaggle Right Sidebar (Output tab)!')"
            ],
            "execution_count": None,
            "outputs": []
        }
    ]
}

with open(LOCAL_NB_PATH, "w", encoding="utf-8") as f:
    json.dump(kaggle_notebook, f, indent=2)

with open(DESKTOP_NB_PATH, "w", encoding="utf-8") as f:
    json.dump(kaggle_notebook, f, indent=2)

print(f"✅ Kaggle Notebook created at: {LOCAL_NB_PATH}")
print(f"🖥️ Copied to Desktop at: {DESKTOP_NB_PATH}")
