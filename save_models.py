# ============================================================
# save_models.py  —  Run this ONCE inside your Jupyter notebooks
# to export all trained models in the correct format.
# ============================================================
# HOW TO USE:
#   Copy the relevant block into your notebook AFTER training.
#   Run it. The model file will appear in backend/models/
# ============================================================

import os, pickle
MODEL_DIR = "backend/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────
# 1.  R1_MLP.ipynb  →  after training `model`
# ──────────────────────────────────────────────────────────
# model.save(os.path.join(MODEL_DIR, "mlp_model.keras"))
# print("✅ MLP saved")

# ──────────────────────────────────────────────────────────
# 2.  R1_CNN.ipynb  →  after training `model` (best model)
# ──────────────────────────────────────────────────────────
# model.save(os.path.join(MODEL_DIR, "cnn_model.keras"))
# print("✅ CNN saved")

# ──────────────────────────────────────────────────────────
# 3.  pretrained_cnn.ipynb  →  after training
# ──────────────────────────────────────────────────────────
# model.save(os.path.join(MODEL_DIR, "pretrained_cnn_model.keras"))
# print("✅ Pretrained CNN saved")

# ──────────────────────────────────────────────────────────
# 4.  R2_RNN.ipynb  →  after training + save scaler once
# ──────────────────────────────────────────────────────────
# model.save(os.path.join(MODEL_DIR, "rnn_model.keras"))
# with open(os.path.join(MODEL_DIR, "audio_scaler.pkl"), "wb") as f:
#     pickle.dump(scaler, f)          # the StandardScaler fitted on training MFCC
# print("✅ RNN + scaler saved")

# ──────────────────────────────────────────────────────────
# 5.  R2_LSTM.ipynb
# ──────────────────────────────────────────────────────────
# model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
# print("✅ LSTM saved")

# ──────────────────────────────────────────────────────────
# 6.  R2_GRU.ipynb
# ──────────────────────────────────────────────────────────
# model.save(os.path.join(MODEL_DIR, "gru_model.keras"))
# print("✅ GRU saved")

# ──────────────────────────────────────────────────────────
# 7.  Autoencoder_MiniProject3.ipynb  →  encoder-decoder
# ──────────────────────────────────────────────────────────
# autoencoder_model.save(os.path.join(MODEL_DIR, "autoencoder_model.keras"))
# print("✅ Autoencoder saved")

# ──────────────────────────────────────────────────────────
# 8.  GAN  →  save generator only
# ──────────────────────────────────────────────────────────
# generator.save(os.path.join(MODEL_DIR, "gan_generator.keras"))
# print("✅ GAN Generator saved")

print("\n📁 All models are saved in:", MODEL_DIR)
print("📋 Files present:", os.listdir(MODEL_DIR))
