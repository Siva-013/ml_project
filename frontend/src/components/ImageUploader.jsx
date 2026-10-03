import React, { useMemo } from "react";

const ALLOWED = ".jpg,.jpeg,.png";

export default function ImageUploader({ file, onFileSelected, config }) {
  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file]
  );
  const maxSizeMb = config.max_upload_mb ?? 10;

  function handleFileChange(event) {
    const selected = event.target.files && event.target.files[0];
    onFileSelected(selected || null);
  }

  return (
    <div className="uploader">
      <input
        type="file"
        accept={ALLOWED}
        onChange={handleFileChange}
        aria-label="Choose an image"
      />
      <p className="hint">
        Supported formats: JPG / JPEG / PNG — up to {maxSizeMb} MB. The image is only
        sent when you click “Generate Caption”.
      </p>
      {file && (
        <div className="preview">
          {previewUrl && <img src={previewUrl} alt="Selected preview" />}
          <p>
            {file.name} — {(file.size / 1024).toFixed(1)} KB
          </p>
        </div>
      )}
    </div>
  );
}
