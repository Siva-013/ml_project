import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App.jsx";
import * as api from "../api.js";

vi.mock("../api.js", () => ({
  fetchConfig: vi.fn(),
  generateCaption: vi.fn(),
}));

const CONFIG = {
  default_top_k: 5,
  default_model: "trigram",
  supported_models: ["unigram", "bigram", "trigram"],
  allowed_top_k_range: [1, 10],
  max_upload_mb: 10,
  vision_api: { configured: true, provider: "OpenAI", model: "gpt-4o-mini" },
};

const SUCCESS_RESPONSE = {
  success: true,
  classical: {
    caption: "a man in a field",
    model: "trigram",
    model_label: "Trigram N-gram",
    top_k: 5,
    candidate_caption_count: 25,
    candidate_word_count: 117,
    candidate_captions_sample: ["a man is playing a guitar on stage"],
    retrieved_images: [
      { image_id: "img_1.jpg", similarity: 0.1645 },
      { image_id: "img_2.jpg", similarity: 0.1634 },
    ],
    processing: { feature_dimension: 1804, similarity_metric: "cosine", elapsed_ms: 250 },
  },
  vision_api: {
    caption: "A dog running through a grassy field.",
    provider: "OpenAI",
    model: "gpt-4o-mini",
    error: null,
  },
};

function imageFile() {
  return new File([new Uint8Array([1, 2, 3])], "photo.jpg", { type: "image/jpeg" });
}

async function selectImage(user) {
  const input = screen.getByLabelText(/choose an image/i);
  await user.upload(input, imageFile());
}

beforeEach(() => {
  vi.clearAllMocks();
  api.fetchConfig.mockResolvedValue(CONFIG);
});

describe("App", () => {
  it("renders the page with a connected vision API status", async () => {
    render(<App />);
    expect(
      screen.getByRole("heading", { name: /classical ml \+ vision-language api/i })
    ).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/Vision API: Connected/i)).toBeInTheDocument());
  });

  it("shows a preview after image selection", async () => {
    const user = userEvent.setup();
    render(<App />);
    await selectImage(user);
    expect(screen.getByText(/photo\.jpg/i)).toBeInTheDocument();
    expect(screen.getByAltText(/selected preview/i)).toBeInTheDocument();
  });

  it("shows a loading state with the button disabled while generating", async () => {
    let resolveGeneration;
    api.generateCaption.mockReturnValue(
      new Promise((resolve) => {
        resolveGeneration = resolve;
      })
    );
    const user = userEvent.setup();
    render(<App />);
    await selectImage(user);

    const button = screen.getByRole("button", { name: /generate caption/i });
    await user.click(button);
    expect(button).toBeDisabled();
    expect(
      screen.getByText(/running the classical pipeline and querying the vision api/i)
    ).toBeInTheDocument();

    resolveGeneration(SUCCESS_RESPONSE);
    await waitFor(() => expect(button).toBeEnabled());
  });

  it("renders both captions side by side with clear labels", async () => {
    api.generateCaption.mockResolvedValue(SUCCESS_RESPONSE);
    const user = userEvent.setup();
    render(<App />);
    await selectImage(user);
    await user.click(screen.getByRole("button", { name: /generate caption/i }));

    await waitFor(() =>
      expect(screen.getByText(/a man in a field/i)).toBeInTheDocument()
    );
    expect(screen.getByText(/A dog running through a grassy field\./i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Classical ML System" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Vision-Language API" })).toBeInTheDocument();
    // each block is clearly attributed
    expect(screen.getByText(/classical ml pipeline \(phases 2–8\)/i)).toBeInTheDocument();
    expect(screen.getByText(/external vision-language api \(hosted model\)/i)).toBeInTheDocument();
    // generation used the real pipeline result values
    expect(screen.getAllByText("Trigram N-gram").length).toBeGreaterThan(0);
    expect(screen.getByText("25")).toBeInTheDocument();
    expect(screen.getByText(/img_1\.jpg/i)).toBeInTheDocument();
  });

  it("renders the vision API error without losing the classical caption", async () => {
    api.generateCaption.mockResolvedValue({
      ...SUCCESS_RESPONSE,
      vision_api: {
        caption: null,
        provider: "OpenAI",
        model: "gpt-4o-mini",
        error: "Vision API rate limit reached",
      },
    });
    const user = userEvent.setup();
    render(<App />);
    await selectImage(user);
    await user.click(screen.getByRole("button", { name: /generate caption/i }));

    await waitFor(() =>
      expect(screen.getByText(/Vision API: Vision API rate limit reached/i)).toBeInTheDocument()
    );
    expect(screen.getByText(/a man in a field/i)).toBeInTheDocument(); // classical intact
    expect(screen.getByText(/No API caption available\./i)).toBeInTheDocument();
  });

  it("shows a friendly error when generation fails", async () => {
    api.generateCaption.mockRejectedValue(
      new Error("Unable to generate a caption for this image. Please try another image.")
    );
    const user = userEvent.setup();
    render(<App />);
    await selectImage(user);
    await user.click(screen.getByRole("button", { name: /generate caption/i }));

    await waitFor(() =>
      expect(
        screen.getByText(/Unable to generate a caption for this image/i)
      ).toBeInTheDocument()
    );
  });
});
