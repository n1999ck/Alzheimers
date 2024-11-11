import { render, screen } from "@testing-library/react";
import App from "../App.js";

test("renders anything", () => {
  render(<App />);
  const headerElement = screen.getByText(/Disease Predictor/i);
  expect(headerElement).toBeInTheDocument();
});
