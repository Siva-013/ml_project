import "@testing-library/jest-dom/vitest";

// jsdom does not implement object URLs
if (typeof URL.createObjectURL === "undefined") {
  URL.createObjectURL = () => "blob:mock";
}
