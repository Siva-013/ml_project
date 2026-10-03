import React from "react";

export default function CandidateCaptions({ captionCount, wordCount, sample }) {
  if (captionCount === undefined) return null;
  return (
    <section className="card">
      <h2>Caption Candidates (visual evidence)</h2>
      <p className="hint">
        The training captions of the retrieved images — the decoder’s candidate
        pool. Sample:
      </p>
      <ul className="candidates">
        {(sample || []).map((caption, index) => (
          <li key={index}>“{caption}”</li>
        ))}
      </ul>
      <p>
        {captionCount} candidate captions · {wordCount} candidate words
      </p>
    </section>
  );
}
