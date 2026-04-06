// ============================================================
// src/App.jsx  —  React Frontend for Marine Species Detection
// ============================================================
import { useState, useRef } from "react";

const API = "http://localhost:8000";

const MODELS = [
  { id: "cnn",           label: "CNN",           icon: "👁️", type: "image",    acc: "93.94%" },
  { id: "mlp",           label: "MLP",           icon: "🧠", type: "image",    acc: "79.67%" },
  { id: "pretrained_cnn",label: "Pretrained CNN",icon: "🔬", type: "image",    acc: "8.83%"  },
  { id: "rnn",           label: "RNN",           icon: "🌀", type: "audio",    acc: "85.75%" },
  { id: "lstm",          label: "LSTM",          icon: "🧬", type: "audio",    acc: "92.96%" },
  { id: "gru",           label: "GRU",           icon: "⚡", type: "audio",    acc: "94.38%" },
  { id: "autoencoder",   label: "Autoencoder",   icon: "🗜️", type: "image",    acc: "SSIM 0.99" },
  { id: "gan",           label: "GAN",           icon: "🎨", type: "generate", acc: "AUC 0.97" },
];

export default function App() {
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);
  const [file, setFile]         = useState(null);
  const [preview, setPreview]   = useState(null);
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState(null);
  const [ganSeed, setGanSeed]   = useState(42);
  const inputRef = useRef(null);

  // ── handle file pick ──
  const handleFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    if (selectedModel.type === "image") {
      setPreview(URL.createObjectURL(f));
    } else if (selectedModel.type === "audio") {
      setPreview(URL.createObjectURL(f));
    }
  };

  // ── switch model ──
  const switchModel = (m) => {
    setSelectedModel(m);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  // ── call API ──
  const runPrediction = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      let res;

      if (selectedModel.id === "gan") {
        // GAN: no file, just seed
        res = await fetch(`${API}/predict/gan?seed=${ganSeed}`, { method: "POST" });
      } else {
        if (!file) throw new Error("Please upload a file first.");
        const form = new FormData();
        form.append("file", file);
        res = await fetch(`${API}/predict/${selectedModel.id}`, { method: "POST", body: form });
      }

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `Server error ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const acceptType = selectedModel.type === "audio" ? "audio/*" : "image/*";
  const canRun = selectedModel.type === "generate" || !!file;

  return (
    <div style={styles.page}>
      {/* ── HEADER ── */}
      <header style={styles.header}>
        <div style={styles.logo}>🐋 Marine Species AI</div>
        <div style={styles.subtitle}>24AI636 · Deep Learning · Yehaasary KM · CB.SC.P2AIE25032</div>
      </header>

      <div style={styles.container}>
        <h2 style={styles.sectionTitle}>🎮 Interactive Prediction Demo</h2>
        <p style={styles.desc}>Upload a real image or audio file — your trained model will predict the marine species.</p>

        {/* ── MODEL SELECTOR ── */}
        <div style={styles.modelGrid}>
          {MODELS.map((m) => (
            <button
              key={m.id}
              onClick={() => switchModel(m)}
              style={{
                ...styles.modelBtn,
                ...(selectedModel.id === m.id ? styles.modelBtnActive : {}),
              }}
            >
              <span style={{ fontSize: "1.5rem" }}>{m.icon}</span>
              <span style={styles.modelBtnLabel}>{m.label}</span>
              <span style={styles.modelBtnAcc}>{m.acc}</span>
            </button>
          ))}
        </div>

        {/* ── INPUT PANEL ── */}
        <div style={styles.card}>
          {/* Image upload */}
          {selectedModel.type === "image" && (
            <div
              style={styles.dropZone}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); }}
            >
              {preview ? (
                <img src={preview} alt="preview" style={styles.preview} />
              ) : (
                <>
                  <div style={{ fontSize: "3rem" }}>🖼️</div>
                  <p style={styles.dropText}>Drop an image or click to browse</p>
                  <p style={styles.dropSub}>JPG · PNG · WEBP</p>
                </>
              )}
              <input
                ref={inputRef}
                type="file"
                accept={acceptType}
                style={{ display: "none" }}
                onChange={(e) => handleFile(e.target.files[0])}
              />
            </div>
          )}

          {/* Audio upload */}
          {selectedModel.type === "audio" && (
            <div style={styles.dropZone} onClick={() => inputRef.current?.click()}>
              <div style={{ fontSize: "3rem" }}>🎵</div>
              <p style={styles.dropText}>{file ? `✅ ${file.name}` : "Drop an audio file or click to browse"}</p>
              <p style={styles.dropSub}>WAV · MP3 · OGG</p>
              {preview && (
                <audio controls src={preview} style={{ marginTop: 12, width: "100%" }} />
              )}
              <input
                ref={inputRef}
                type="file"
                accept={acceptType}
                style={{ display: "none" }}
                onChange={(e) => handleFile(e.target.files[0])}
              />
            </div>
          )}

          {/* GAN */}
          {selectedModel.type === "generate" && (
            <div style={{ ...styles.dropZone, cursor: "default" }}>
              <div style={{ fontSize: "3rem" }}>🎲</div>
              <p style={styles.dropText}>GAN — Generate a marine animal image</p>
              <label style={styles.seedLabel}>
                Latent seed:&nbsp;
                <input
                  type="number" min="1" max="9999" value={ganSeed}
                  onChange={(e) => setGanSeed(Number(e.target.value))}
                  style={styles.seedInput}
                />
              </label>
            </div>
          )}
        </div>

        {/* ── ERROR ── */}
        {error && (
          <div style={styles.errorBox}>
            ⚠️ {error}
            {error.includes("not found") && (
              <div style={{ marginTop: 8, fontSize: ".8rem" }}>
                Make sure you've saved the model file → <code>backend/models/{selectedModel.id}_model.keras</code>
              </div>
            )}
          </div>
        )}

        {/* ── RUN BUTTON ── */}
        <button
          onClick={runPrediction}
          disabled={!canRun || loading}
          style={{ ...styles.runBtn, opacity: (!canRun || loading) ? 0.6 : 1 }}
        >
          {loading ? "⏳ Running..." : "⚡ RUN PREDICTION"}
        </button>

        {/* ── RESULTS ── */}
        {result && <ResultPanel result={result} model={selectedModel} />}
      </div>
    </div>
  );
}

// ── Result display component ──
function ResultPanel({ result, model }) {
  if (model.id === "autoencoder") {
    return (
      <div style={styles.resultCard}>
        <h3 style={styles.resultTitle}>🗜️ Autoencoder Reconstruction</h3>
        <div style={styles.statsRow}>
          <StatPill label="MSE"    value={result.mse} />
          <StatPill label="PSNR"   value={`${result.psnr_db} dB`} />
          <StatPill label="SSIM"   value={result.ssim} color="#1de9b6" />
          <StatPill label="Quality" value={result.quality} color="#f0c040" />
        </div>
      </div>
    );
  }

  if (model.id === "gan") {
    return (
      <div style={styles.resultCard}>
        <h3 style={styles.resultTitle}>🎨 GAN Generated Image</h3>
        {result.image_base64 && (
          <img
            src={`data:image/png;base64,${result.image_base64}`}
            alt="GAN generated"
            style={{ width: 180, height: 180, borderRadius: 10, border: "2px solid #1de9b6", imageRendering: "pixelated" }}
          />
        )}
        <p style={{ color: "#7ab8d4", marginTop: 8, fontSize: ".82rem" }}>
          64×64 synthesized from latent z (seed={result.seed})
        </p>
      </div>
    );
  }

  // Classification result
  return (
    <div style={styles.resultCard}>
      <h3 style={styles.resultTitle}>📊 Prediction Results — {result.model}</h3>

      {/* Top prediction */}
      <div style={styles.topPred}>
        <div style={{ fontSize: ".7rem", letterSpacing: ".15em", color: "#1de9b6", marginBottom: 6 }}>🏆 TOP PREDICTION</div>
        <div style={{ fontFamily: "'Cinzel', serif", fontSize: "1.8rem", color: "#fff", marginBottom: 4 }}>{result.top1}</div>
        <div style={{ color: "#1de9b6", fontFamily: "monospace", fontSize: "1rem", marginBottom: 12 }}>
          Confidence: {result.confidence}%
        </div>
        <ConfidenceBar value={result.confidence} />
      </div>

      {/* Top-5 */}
      {result.top5 && (
        <div style={{ marginTop: 20 }}>
          <div style={{ fontSize: ".7rem", letterSpacing: ".12em", color: "#7ab8d4", marginBottom: 10 }}>TOP-5 SPECIES</div>
          {result.top5.map((p, i) => (
            <div key={i} style={styles.predRow}>
              <span style={{ color: i === 0 ? "#1de9b6" : "#d6eef8", flex: 1, fontSize: ".84rem" }}>
                {i === 0 ? "🏆 " : `${i + 1}. `}{p.species}
              </span>
              <span style={{ color: "#1de9b6", fontFamily: "monospace", fontSize: ".78rem", minWidth: 52, textAlign: "right" }}>
                {p.confidence}%
              </span>
              <div style={styles.barTrack}>
                <div style={{ ...styles.barFill, width: `${p.confidence}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ConfidenceBar({ value }) {
  return (
    <div style={{ height: 8, background: "rgba(255,255,255,.07)", borderRadius: 4, overflow: "hidden" }}>
      <div style={{ height: "100%", width: `${value}%`, background: "linear-gradient(90deg,#0d7377,#1de9b6)", borderRadius: 4, transition: "width 1s ease" }} />
    </div>
  );
}

function StatPill({ label, value, color = "#1de9b6" }) {
  return (
    <div style={styles.statPill}>
      <span style={{ color, fontFamily: "monospace", fontSize: "1rem", display: "block" }}>{value}</span>
      <span style={{ fontSize: ".65rem", color: "#7ab8d4", textTransform: "uppercase", letterSpacing: ".08em" }}>{label}</span>
    </div>
  );
}

// ── Styles ──
const styles = {
  page: { minHeight: "100vh", background: "#020d1a", color: "#d6eef8", fontFamily: "'Raleway', sans-serif" },
  header: { background: "rgba(6,20,40,.9)", borderBottom: "1px solid rgba(29,233,182,.15)", padding: "18px 40px", display: "flex", flexDirection: "column", gap: 4 },
  logo: { fontFamily: "'Cinzel', serif", fontSize: "1.2rem", color: "#1de9b6", letterSpacing: ".1em" },
  subtitle: { fontSize: ".72rem", color: "#7ab8d4", letterSpacing: ".06em" },
  container: { maxWidth: 900, margin: "0 auto", padding: "48px 24px" },
  sectionTitle: { fontFamily: "'Cinzel', serif", fontSize: "1.8rem", color: "#f0faff", marginBottom: 12 },
  desc: { color: "#7ab8d4", marginBottom: 32, lineHeight: 1.7 },

  modelGrid: { display: "flex", flexWrap: "wrap", gap: 10, marginBottom: 24 },
  modelBtn: { display: "flex", flexDirection: "column", alignItems: "center", gap: 4, padding: "12px 16px", background: "rgba(10,32,64,.6)", border: "1px solid rgba(29,233,182,.15)", borderRadius: 12, cursor: "pointer", transition: "all .25s", minWidth: 90 },
  modelBtnActive: { background: "rgba(29,233,182,.12)", borderColor: "rgba(29,233,182,.5)" },
  modelBtnLabel: { fontSize: ".8rem", color: "#f0faff", fontFamily: "'Raleway', sans-serif" },
  modelBtnAcc: { fontSize: ".65rem", color: "#1de9b6", fontFamily: "monospace" },

  card: { background: "rgba(6,20,40,.7)", border: "1px solid rgba(29,233,182,.15)", borderRadius: 16, padding: 24, marginBottom: 20 },
  dropZone: { border: "2px dashed rgba(29,233,182,.35)", borderRadius: 12, padding: 32, textAlign: "center", cursor: "pointer", transition: "all .3s" },
  dropText: { fontFamily: "'Cinzel', serif", color: "#f0faff", marginBottom: 6, fontSize: "1rem" },
  dropSub: { color: "#7ab8d4", fontSize: ".76rem" },
  preview: { maxHeight: 240, maxWidth: "100%", borderRadius: 10, border: "2px solid rgba(29,233,182,.3)" },

  seedLabel: { display: "flex", alignItems: "center", justifyContent: "center", gap: 10, color: "#7ab8d4", marginTop: 16, fontSize: ".85rem" },
  seedInput: { background: "rgba(10,32,64,.8)", border: "1px solid rgba(29,233,182,.3)", borderRadius: 6, color: "#1de9b6", padding: "4px 10px", fontFamily: "monospace", width: 80, textAlign: "center" },

  errorBox: { background: "rgba(255,80,80,.1)", border: "1px solid rgba(255,80,80,.3)", borderRadius: 10, padding: "14px 18px", color: "#ff9090", marginBottom: 16, fontSize: ".85rem" },

  runBtn: { display: "block", width: "100%", padding: "16px", background: "linear-gradient(135deg,#0d7377,#1de9b6)", color: "#020d1a", fontFamily: "'Cinzel', serif", fontSize: "1rem", letterSpacing: ".1em", border: "none", borderRadius: 12, cursor: "pointer", fontWeight: 700, marginBottom: 28, transition: "all .25s" },

  resultCard: { background: "rgba(6,20,40,.7)", border: "1px solid rgba(29,233,182,.2)", borderRadius: 16, padding: 28 },
  resultTitle: { fontFamily: "'Cinzel', serif", color: "#f0faff", marginBottom: 20, fontSize: "1.1rem" },
  topPred: { background: "rgba(13,115,119,.2)", border: "2px solid rgba(29,233,182,.35)", borderRadius: 12, padding: 22, textAlign: "center" },

  predRow: { display: "flex", alignItems: "center", gap: 10, marginBottom: 10 },
  barTrack: { flex: 1.5, height: 5, background: "rgba(255,255,255,.06)", borderRadius: 3, overflow: "hidden" },
  barFill: { height: "100%", background: "linear-gradient(90deg,#0d7377,#14a5b5)", borderRadius: 3, transition: "width 1s ease" },

  statsRow: { display: "flex", flexWrap: "wrap", gap: 12, marginTop: 12 },
  statPill: { background: "rgba(29,233,182,.07)", border: "1px solid rgba(29,233,182,.15)", borderRadius: 10, padding: "12px 18px", textAlign: "center", minWidth: 100 },
};
