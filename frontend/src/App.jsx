import React, { useEffect, useState } from "react";
import { fetchConfig, generateCaption } from "./api.js";
import ImageUploader from "./components/ImageUploader.jsx";
import CaptionResult from "./components/CaptionResult.jsx";
import RetrievedImages from "./components/RetrievedImages.jsx";
import CandidateCaptions from "./components/CandidateCaptions.jsx";
import ProcessingInfo from "./components/ProcessingInfo.jsx";

const FALLBACK_CONFIG = {
  default_top_k: 5,
  default_model: "trigram",
  supported_models: ["unigram", "bigram", "trigram"],
  allowed_top_k_range: [1, 10],
  max_upload_mb: 10,
};

export default function App() {
  const [config, setConfig] = useState(FALLBACK_CONFIG);
  const [file, setFile] = useState(null);
  const [topK, setTopK] = useState(FALLBACK_CONFIG.default_top_k);
  const [model, setModel] = useState(FALLBACK_CONFIG.default_model);
  const [maxLength, setMaxLength] = useState(20);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchConfig()
      .then((config) => {
        setConfig(config);
        setTopK(config.default_top_k);
        setModel(config.default_model);
      })
      .catch(() => {
        setError("Backend is not reachable. Start it with: python -m uvicorn backend.main:app --reload");
      });
  }, []);

  const canGenerate = file && !loading;

  async function handleGenerate() {
    if (!canGenerate) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(
        await generateCaption(file, {
          top_k: topK,
          model,
          max_length: maxLength,
          min_length: 3,
        })
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="page-header">
        <h1>Classical ML Image Caption Generator</h1>
        <p className="subtitle">
          Classical computer vision + statistical NLP — no LLMs, no neural networks.
        </p>
      </header>

      <section className="card">
        <h2>Upload an image</h2>
        <ImageUploader file={file} onFileSelected={setFile} config={config} />
        <div className="controls">
          <label>
            Retrieval Top-K
            <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
              {[1, 3, 5, 10].map((k) => (
                <option key={k} value={k}>
                  Top-{k}
                </option>
              ))}
            </select>
          </label>
          <label>
            Language model
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {config.supported_models.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Max caption length
            <input
              type="number"
              min={1}
              max={30}
              value={maxLength}
              onChange={(event) => setMaxLength(Number(event.target.value))}
            />
          </label>
        </div>
        <button className="generate" onClick={handleGenerate} disabled={!canGenerate}>
          {loading ? "Generating caption..." : "Generate Caption"}
        </button>
        {loading && <p className="loading">Extracting visual features, retrieving training images, decoding…</p>}
        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <>
          <CaptionResult caption={result.generated_caption} />
          <RetrievedImages retrieved={result.retrieved_images} />
          <CandidateCaptions
            captionCount={result.candidate_caption_count}
            wordCount={result.candidate_word_count}
            sample={result.candidate_captions_sample}
          />
          <ProcessingInfo processing={result.processing} languageModel={result.language_model} topK={result.top_k} />
        </>
      )}

      <section className="card methodology">
        <h2>Methodology</h2>
        <ol>
          <li>HOG + HSV colour histogram + LBP (classical handcrafted features)</li>
          <li>Per-family standardization, feature fusion, PCA (fitted on training data only)</li>
          <li>Cosine-similarity retrieval over 6,000 training images</li>
          <li>Captions of the retrieved training images as candidate pool</li>
          <li>Trigram n-gram language model with Laplace smoothing (greedy decoding)</li>
        </ol>
        <p className="note">
          The visual descriptors measure colour, texture and edge statistics — they do not
          recognise objects. Captions are generated from <em>visually similar</em> training
          images and may therefore be generic or imperfect.
        </p>
      </section>
    </div>
  );
}
