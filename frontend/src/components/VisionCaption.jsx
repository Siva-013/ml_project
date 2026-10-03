import React from "react";

export default function VisionCaption({ vision }) {
  if (!vision) return null;
  return (
    <section className="card caption-box vision">
      <h2>Vision-Language API</h2>
      {vision.caption ? (
        <p className="caption">“{vision.caption}”</p>
      ) : (
        <p className="caption muted">No API caption available.</p>
      )}
      <dl className="meta">
        <div>
          <dt>Provider</dt>
          <dd>{vision.provider}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{vision.model}</dd>
        </div>
      </dl>
      {vision.error && <p className="error">Vision API: {vision.error}</p>}
      <p className="tag vision-tag">external vision-language API (hosted model)</p>
    </section>
  );
}
