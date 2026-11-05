"""
Evaluate Character Error Rate (CER), Word Error Rate (WER), and Character Accuracy
for the Trained Proprietary Conformer/Transformer STT Model.
"""

import os
import sys
import re
import json
import math
import torch
import librosa
import soundfile as sf
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Conv2dSubsampling(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, conv_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.out_proj = nn.Linear(conv_channels * (input_dim // 4), hidden_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        return self.out_proj(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ProprietarySTTModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, num_layers=6, num_heads=4, vocab_size=196, dropout=0.1, blank_id=2):
        super().__init__()
        self.blank_id = blank_id
        self.subsampling = Conv2dSubsampling(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ctc_linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, feature_lengths):
        x = self.subsampling(features)
        out_lengths = (feature_lengths // 4).clamp(min=1)
        x = self.pos_encoder(x)
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= out_lengths.unsqueeze(1)
        encoded = self.layer_norm(self.transformer(x, src_key_padding_mask=mask))
        logits = self.ctc_linear(encoded)
        return F.log_softmax(logits, dim=-1), out_lengths

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def run_evaluation(num_samples_per_lang=25, lm=None):
    from model.language_model import MultilingualLanguageModel
    if lm is None:
        lm = MultilingualLanguageModel('models/proprietary/lm_data.json')

    vocab_path = 'models/proprietary/vocab.json'
    model_path = 'models/proprietary/proprietary_stt_model.pt'
    manifest_path = 'data/chunked_dataset/manifest.csv'

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id2char = {idx: token for token, idx in vocab.items()}

    ckpt = torch.load(model_path, map_location='cpu')
    model = ProprietarySTTModel(vocab_size=len(vocab))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80)

    df = pd.read_csv(manifest_path)
    train_df = df.sample(frac=0.90, random_state=42)
    val_df = df.drop(train_df.index)

    test_subsets = []
    for lang in ['gu', 'hi', 'en']:
        sub = val_df[val_df['language'] == lang].head(num_samples_per_lang)
        test_subsets.append(sub)
    eval_df = pd.concat(test_subsets).reset_index(drop=True)

    results = []
    for _, row in eval_df.iterrows():
        audio_path = os.path.join('data/chunked_dataset', row['audio_path'])
        if not os.path.exists(audio_path):
            continue
        try:
            audio, sr = sf.read(audio_path)
            if audio.ndim > 1: audio = audio.mean(axis=1)
            if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except Exception:
            continue

        wf = torch.from_numpy(audio).float()
        if wf.abs().max() > 0: wf = (wf / wf.abs().max()) * 0.95
        mel = mel_transform(wf)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        norm_mel = (log_mel - log_mel.mean(dim=-1, keepdim=True)) / (log_mel.std(dim=-1, keepdim=True) + 1e-5)

        with torch.no_grad():
            log_probs, _ = model(norm_mel.unsqueeze(0), torch.tensor([norm_mel.size(-1)]))
            pred_ids = log_probs[0].argmax(dim=-1).tolist()

        # Raw CTC Greedy Decoding
        collapsed, prev = [], None
        for tid in pred_ids:
            if tid in (0, 2):
                prev = tid; continue
            if tid != prev:
                collapsed.append(tid)
                prev = tid
        pred_raw = ''.join([id2char.get(i, '') for i in collapsed]).replace('|', ' ').strip()
        
        # Language Model Enhanced Decoding
        pred_lm = lm.correct_transcript(pred_raw)

        ref_raw = str(row['transcript']).lower().strip()
        ref_clean = re.sub(r'[^\w\s\u0A80-\u0AFF\u0900-\u097F]', '', ref_raw)
        ref_clean = re.sub(r'\s+', ' ', ref_clean).strip()

        # CER & WER Raw
        dist_raw = levenshtein(ref_clean, pred_raw)
        cer_raw = min(1.0, dist_raw / max(1, len(ref_clean)))
        ref_words = ref_clean.split()
        w_dist_raw = levenshtein(ref_words, pred_raw.split())
        wer_raw = min(1.0, w_dist_raw / max(1, len(ref_words)))

        # CER & WER with LM
        dist_lm = levenshtein(ref_clean, pred_lm)
        cer_lm = min(1.0, dist_lm / max(1, len(ref_clean)))
        w_dist_lm = levenshtein(ref_words, pred_lm.split())
        wer_lm = min(1.0, w_dist_lm / max(1, len(ref_words)))

        results.append({
            'lang': row['language'],
            'ref': ref_clean,
            'pred_raw': pred_raw,
            'pred_lm': pred_lm,
            'cer_raw': cer_raw,
            'cer_lm': cer_lm,
            'wer_raw': wer_raw,
            'wer_lm': wer_lm
        })

    res_df = pd.DataFrame(results)
    return res_df

if __name__ == '__main__':
    from model.language_model import MultilingualLanguageModel
    lm = MultilingualLanguageModel('models/proprietary/lm_data.json')
    res = run_evaluation(num_samples_per_lang=20)
    print("\n" + "="*70)
    print("      PROPRIETARY STT MODEL - EVALUATION REPORT (BASELINE vs LM)")
    print("="*70)
    for lang, g in res.groupby('lang'):
        m_cer_raw = g['cer_raw'].mean()
        m_cer_lm = g['cer_lm'].mean()
        acc_raw = max(0.0, 1.0 - m_cer_raw) * 100
        acc_lm = max(0.0, 1.0 - m_cer_lm) * 100
        m_wer_raw = g['wer_raw'].mean()
        m_wer_lm = g['wer_lm'].mean()
        name = {'gu': 'Gujarati', 'hi': 'Hindi', 'en': 'English'}.get(lang, lang)
        print(f"{name:10} | Raw CER: {m_cer_raw*100:5.2f}% -> LM CER: {m_cer_lm*100:5.2f}% | Accuracy: {acc_raw:5.2f}% -> {acc_lm:5.2f}%")
    
    tot_cer_raw = res['cer_raw'].mean()
    tot_cer_lm = res['cer_lm'].mean()
    tot_acc_raw = max(0.0, 1.0 - tot_cer_raw) * 100
    tot_acc_lm = max(0.0, 1.0 - tot_cer_lm) * 100
    print("-" * 70)
    print(f"{'OVERALL':10} | Raw CER: {tot_cer_raw*100:5.2f}% -> LM CER: {tot_cer_lm*100:5.2f}% | Accuracy: {tot_acc_raw:5.2f}% -> {tot_acc_lm:5.2f}%")
    print("="*70)
