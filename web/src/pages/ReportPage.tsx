import { ArrowLeft, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "@/api/client";
import { FrameChart } from "@/components/charts";
import { PredictionPanel } from "@/components/Prediction";
import { DownloadButtons, EvidenceSection } from "@/components/ReportParts";
import { Badge, Button, Dialog, KeyValue, Notice, PageHeader, Skeleton } from "@/components/ui";
import { categoryTone, classLabel, formatBytes, formatDate } from "@/lib/format";
import { useAsync, useDocumentTitle } from "@/lib/hooks";

export function ReportPage() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const { data: r, error, loading } = useAsync((s) => api.report(id, s), [id]);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [delErr, setDelErr] = useState<string | null>(null);
  useDocumentTitle(r ? `Report · ${r.file.name}` : "Report");

  async function remove() {
    setBusy(true);
    try { await api.deleteReport(id); nav("/reports"); } catch (e) { setDelErr((e as Error).message); setBusy(false); setConfirm(false); }
  }

  if (loading) return <div className="space-y-4" aria-busy="true"><Skeleton className="h-8 w-80" /><Skeleton className="h-4 w-64" /><Skeleton className="h-72 w-full" /></div>;
  if (error || !r) {
    const notFound = error instanceof ApiError && error.status === 404;
    return (
      <>
        <Link to="/reports" className="link mb-6 inline-flex items-center gap-1 text-sm"><ArrowLeft className="h-4 w-4" aria-hidden />All reports</Link>
        <Notice kind={notFound ? "warn" : "error"} title={notFound ? "Report not found" : "Could not load the report"}>
          {error?.message ?? "Unknown error."}{notFound && " Reports can be lost when a server without persistent storage restarts."}
        </Notice>
      </>
    );
  }

  return (
    <article>
      <Link to="/reports" className="no-print link mb-5 inline-flex items-center gap-1 text-sm"><ArrowLeft className="h-4 w-4" aria-hidden />All reports</Link>
      <PageHeader eyebrow={`Forensic report · ${r.kind}`} title={r.file.name}
        description={<>Generated {formatDate(r.created_at)} · report <span className="font-mono text-xs">{r.id}</span></>}
        actions={<><DownloadButtons id={r.id} /><Button size="sm" variant="ghost" onClick={() => setConfirm(true)}><Trash2 className="h-4 w-4" aria-hidden />Delete</Button></>} />
      {delErr && <div className="mb-4"><Notice kind="error">{delErr}</Notice></div>}

      <Notice kind="warn" title="Screening signal, not a forensic determination">{r.uncertainty_warning}</Notice>

      <div className="mt-8 grid gap-8 lg:grid-cols-2">
        <section aria-labelledby="file-h" className="panel p-5">
          <h2 id="file-h" className="eyebrow mb-4">File</h2>
          <KeyValue items={[
            ["Name", r.file.name], ["Type", r.file.mime], ["Size", formatBytes(r.file.size_bytes)],
            ...(r.file.width ? [["Dimensions", `${r.file.width} × ${r.file.height} px`] as [string, string]] : []),
            ...(r.file.duration_seconds != null ? [["Duration", `${r.file.duration_seconds.toFixed(1)} s`] as [string, string]] : []),
            ["SHA-256", <span key="h" className="break-all font-mono text-xs">{r.file.sha256}</span>],
          ]} />
        </section>
        <section aria-labelledby="meth-h" className="panel p-5">
          <h2 id="meth-h" className="eyebrow mb-4">Analysis</h2>
          <KeyValue items={[
            ["Detector", r.detector.version_label], ["Model id", <span key="m" className="font-mono text-xs">{r.detector.name}</span>],
            ["Method", r.analysis_method],
            ["Policy thresholds", `AI ≥ ${r.detector.thresholds.ai.toFixed(2)} · Real ≥ ${r.detector.thresholds.real.toFixed(2)} · otherwise inconclusive`],
            ["Result", <Badge key="b" tone={categoryTone(r.result.category)}>{r.result.category}</Badge>],
            ["Application", `VerityAI v${r.app_version}`],
          ]} />
        </section>
      </div>

      <div className="mt-8 space-y-8">
        <PredictionPanel result={r.result} detector={r.detector} experimental={r.kind === "video"} />
        <p className="rounded-md bg-canvas p-4 text-sm text-ink-2"><span className="font-semibold">How to read this: </span>{r.explanation.detector_summary}</p>

        {r.video && (
          <section aria-labelledby="frames-h">
            <h2 id="frames-h" className="mb-1 text-base font-semibold">Frame-level information</h2>
            <p className="mb-3 text-sm text-ink-3">{r.video.aggregation_method}</p>
            <FrameChart frames={r.video.frames} threshold={r.detector.thresholds.ai} />
            <div className="panel mt-4 overflow-x-auto">
              <table className="w-full min-w-[480px] text-left text-sm">
                <caption className="sr-only">Per-frame detector scores</caption>
                <thead className="border-b border-line bg-sunken/50 text-xs text-ink-3"><tr><th className="px-4 py-2 font-medium">Frame</th><th className="px-4 py-2 font-medium">Time</th><th className="px-4 py-2 font-medium">Top class</th><th className="px-4 py-2 font-medium">AI</th><th className="px-4 py-2 font-medium">Real</th></tr></thead>
                <tbody>{r.video.frames.map((f) => (
                  <tr key={f.index} className="border-b border-line last:border-0"><td className="px-4 py-2 font-mono">{f.index}</td><td className="px-4 py-2 font-mono tnum">{f.timestamp_seconds.toFixed(2)} s</td><td className="px-4 py-2">{classLabel(f.predicted_class)}</td><td className="px-4 py-2 font-mono tnum">{(f.scores.ai ?? 0).toFixed(3)}</td><td className="px-4 py-2 font-mono tnum">{(f.scores.real ?? 0).toFixed(3)}</td></tr>
                ))}</tbody>
              </table>
            </div>
          </section>
        )}

        <EvidenceSection report={r} />

        <section aria-labelledby="lim-h">
          <h2 id="lim-h" className="mb-3 text-base font-semibold">Limitations and uncertainty</h2>
          <ul className="space-y-2 text-sm text-ink-2">{r.limitations.map((l) => <li key={l} className="flex gap-3"><span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-ink-3" aria-hidden />{l}</li>)}</ul>
        </section>
      </div>

      <Dialog open={confirm} onClose={() => setConfirm(false)} title="Delete this report?"
        footer={<><Button variant="ghost" onClick={() => setConfirm(false)}>Cancel</Button><Button variant="danger" loading={busy} onClick={remove}>Delete report</Button></>}>
        This permanently removes the saved report for <span className="font-medium">{r.file.name}</span> from the server. The original file was never stored.
      </Dialog>
    </article>
  );
}
