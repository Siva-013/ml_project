import React, { useEffect, useState } from "react";
import { fetchConfig, generateCaption } from "./api.js";
import ImageUploader from "./components/ImageUploader.jsx";
import ClassicalCaption from "./components/ClassicalCaption.jsx";
import VisionCaption from "./components/VisionCaption.jsx";
import RetrievedImages from "./components/RetrievedImages.jsx";
import CandidateCaptions from "./components/CandidateCaptions.jsx";
import ProcessingInfo from "./components/ProcessingInfo.jsx";

const FALLBACK_CONFIG = {
  default_top_k: 5,
  default_model: "trigram",
  supported_models: ["unigram", "bigram", "trigram"],
  allowed_top_k_range: [1, 10],
  max_upload_mb: 10,
  vision_api: { configured: false, provider: "OpenAI", model: "" },
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
        setError(
          "Backend is not reachable. Start it with: python -m uvicorn backend.main:app --reload"
        );
      });
  }, []);

  const canGenerate = file && !loading;
  const vision = result ? result.vision_api : config.vision_api;

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
        <h1>Classical ML + Vision-Language API Image Captioning</h1>
        <p className="subtitle">
          Two independent captioning methods for one image: the classical
          pipeline (HOG + colour histogram + LBP → PCA → retrieval → trigram
          LM) and an external vision-language API — clearly labelled, side by
          side.
        </p>
        <p className={`vision-status ${vision?.configured ? "connected" : "disconnected"}`}>
          Vision API: {vision?.configured ? "Connected" : "Not configured"}
          {vision?.model ? ` (${vision.provider} · ${vision.model})` : ""}
        </p>
      </header>

      <section className="card">
        <h2>Image Caption Generator</h2>
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
            Classical language model
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
          {loading ? "Analyzing image..." : "Generate Caption"}
        </button>
        {loading && (
          <p className="loading">
            Analyzing image — running the classical pipeline and querying the
            vision API…
          </p>
        )}
        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <>
          <section className="comparison-grid">
            <ClassicalCaption classical={result.classical} />
            <VisionCaption vision={result.vision_api} />
          </section>
          <RetrievedImages retrieved={result.classical.retrieved_images} />
          <CandidateCaptions
            captionCount={result.classical.candidate_caption_count}
            wordCount={result.classical.candidate_word_count}
            sample={result.classical.candidate_captions_sample}
          />
          <ProcessingInfo
            processing={result.classical.processing}
            languageModel={result.classical.model_label}
            topK={result.classical.top_k}
          />
        </>
      )}

      <section className="card methodology">
        <h2>Methodology</h2>
        <ol>
          <li>
            <strong>Classical pipeline:</strong> HOG + HSV colour histogram +
            LBP → per-family standardization → fusion → PCA (training-fitted)
            → cosine retrieval over 6,000 training images → their training
            captions → trigram n-gram language model (greedy decoding).
          </li>
          <li>
            <strong>Vision-language API:</strong> the same uploaded image is
            sent from this page → FastAPI backend → OpenAI vision API → its
            caption is returned verbatim.
          </li>
        </ol>
        <p className="note">
          The classical descriptors measure colour, texture and edge
          statistics — they do not recognise objects, so the classical caption
          can be generic or wrong. The vision-language API is an external
          hosted model: it is <em>not</em> part of the classical ML baseline
          and was not trained by this project. Both outputs are shown exactly
          as produced.
        </p>
      </section>
    </div>
  );
}
