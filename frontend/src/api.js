// API client for the FastAPI backend. In development, Vite proxies /api to
// http://127.0.0.1:8000 (see vite.config.js), so requests are same-origin.

export async function fetchConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) throw new Error("Backend configuration unavailable");
  return response.json();
}

export async function generateCaption(file, params) {
  const form = new FormData();
  form.append("image", file);
  Object.entries(params).forEach(([key, value]) => form.append(key, String(value)));

  const response = await fetch("/api/caption", { method: "POST", body: form });
  if (!response.ok) {
    let message = "Unable to generate a caption for this image. Please try another image.";
    try {
      const error = await response.json();
      if (error && error.detail) message = error.detail;
    } catch {
      // keep the friendly default message
    }
    throw new Error(message);
  }
  return response.json();
}
