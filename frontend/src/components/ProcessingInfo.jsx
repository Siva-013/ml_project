import React from "react";

export default function ProcessingInfo({ processing, languageModel, topK }) {
  if (!processing) return null;
  return (
    <section className="card">
      <h2>Processing Details</h2>
      <table>
        <tbody>
          <tr>
            <th>Language model</th>
            <td>{languageModel}</td>
          </tr>
          <tr>
            <th>Retrieval Top-K</th>
            <td>{topK}</td>
          </tr>
          <tr>
            <th>Feature dimension (PCA)</th>
            <td>{processing.feature_dimension}</td>
          </tr>
          <tr>
            <th>Similarity metric</th>
            <td>{processing.similarity_metric}</td>
          </tr>
          <tr>
            <th>Processing time</th>
            <td>{processing.elapsed_ms} ms</td>
          </tr>
        </tbody>
      </table>
    </section>
  );
}
