import React from "react";

export default function ClassicalCaption({ classical }) {
  if (!classical) return null;
  return (
    <section className="card caption-box classical">
      <h2>Classical ML System</h2>
      <p className="caption">“{classical.caption}”</p>
      <dl className="meta">
        <div>
          <dt>Model</dt>
          <dd>{classical.model_label}</dd>
        </div>
        <div>
          <dt>Retrieval Top-K</dt>
          <dd>{classical.top_k}</dd>
        </div>
        <div>
          <dt>Candidate captions</dt>
          <dd>{classical.candidate_caption_count}</dd>
        </div>
        <div>
          <dt>Candidate words</dt>
          <dd>{classical.candidate_word_count}</dd>
        </div>
      </dl>
      <p className="tag classical-tag">classical ML pipeline (Phases 2–8)</p>
    </section>
  );
}
