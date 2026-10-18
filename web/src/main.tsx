import { MotionConfig } from "framer-motion";
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { HealthProvider } from "./lib/health";
import "./index.css";

// Router basename follows wherever the app is mounted (<base> is injected in index.html).
const basename = new URL(document.baseURI).pathname.replace(/\/$/, "");

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <MotionConfig reducedMotion="user">
      <BrowserRouter basename={basename}>
        <HealthProvider>
          <ErrorBoundary>
            <App />
          </ErrorBoundary>
        </HealthProvider>
      </BrowserRouter>
    </MotionConfig>
  </React.StrictMode>,
);
