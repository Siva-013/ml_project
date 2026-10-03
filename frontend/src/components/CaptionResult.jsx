import React from "react";

export default function CaptionResult({ caption }) {
  if (!caption) return null;
  return (
    <section className="card caption-result">
      <h2>Generated Caption</h2>
      <p className="caption">“{caption}”</p>
    </section>
  );
}
