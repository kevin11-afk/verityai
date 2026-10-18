import { lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "@/components/Shell";
import { Overview } from "@/pages/Overview";

const Analyze = lazy(() => import("@/pages/Analyze").then((m) => ({ default: m.Analyze })));
const ReportPage = lazy(() => import("@/pages/ReportPage").then((m) => ({ default: m.ReportPage })));
const Reports = lazy(() => import("@/pages/Reports").then((m) => ({ default: m.Reports })));
const Research = lazy(() => import("@/pages/Research").then((m) => ({ default: m.Research })));
const Knowledge = lazy(() => import("@/pages/Knowledge").then((m) => ({ default: m.Knowledge })));
const Evaluation = lazy(() => import("@/pages/Evaluation").then((m) => ({ default: m.Evaluation })));
const Methodology = lazy(() => import("@/pages/Methodology").then((m) => ({ default: m.Methodology })));

export function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route index element={<Overview />} />
        <Route path="analyze" element={<Analyze />} />
        <Route path="reports" element={<Reports />} />
        <Route path="reports/:id" element={<ReportPage />} />
        <Route path="research" element={<Research />} />
        <Route path="knowledge" element={<Knowledge />} />
        <Route path="evaluation" element={<Evaluation />} />
        <Route path="methodology" element={<Methodology />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
