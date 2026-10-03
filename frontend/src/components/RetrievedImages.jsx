import React from "react";

export default function RetrievedImages({ retrieved }) {
  if (!retrieved || retrieved.length === 0) return null;
  return (
    <section className="card">
      <h2>Retrieved Similar Training Images</h2>
      <p className="hint">
        Cosine-similarity neighbours from the 6,000-image training gallery —
        these images supplied the candidate captions.
      </p>
      <div className="retrieved-grid">
        {retrieved.map((item) => (
          <figure key={item.image_id}>
            <img src={`/api/gallery/${encodeURIComponent(item.image_id)}`} alt={item.image_id} />
            <figcaption>
              <span className="image-id">{item.image_id}</span>
              <span className="similarity">cosine: {item.similarity.toFixed(4)}</span>
            </figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}
