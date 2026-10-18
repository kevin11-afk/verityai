import { motion } from "framer-motion";
import { ArrowRight, FileImage, Film, RotateCcw, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "@/api/client";
import type { Report } from "@/api/types";
import { Dropzone } from "@/components/Dropzone";
import { FrameChart } from "@/components/charts";
import { PredictionPanel } from "@/components/Prediction";
import { DownloadButtons, EvidenceSection } from "@/components/ReportParts";
import { Button, ButtonLink, Notice, PageHeader, ProgressBar } from "@/components/ui";
import { formatBytes } from "@/lib/format";
import { useDocumentTitle } from "@/lib/hooks";
import { useHealth } from "@/lib/health";

type Phase = "idle" | "uploading" | "analyzing" | "done" | "error";
const FALLBACK = { image: [".jpg", ".jpeg", ".png", ".webp"], video: [".mp4", ".mov"], imageMb: 15, videoMb: 60 };

export function Analyze() {
  useDocumentTitle("Analyze media");
  const { health } = useHealth();
  const limits = health?.limits;
  const imageExt = limits?.image_extensions ?? FALLBACK.image;
  const videoExt = limits?.video_extensions ?? FALLBACK.video;
  const imageMb = limits?.max_image_mb ?? FALLBACK.imageMb;
  const videoMb = limits?.max_video_mb ?? FALLBACK.videoMb;

  const [file, setFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState<number | null>(0);
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryable, setRetryable] = useState(false);
  const [clientIssue, setClientIssue] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);
  useEffect(() => () => abort.current?.abort(), []);

  const kindOf = (f: File): "image" | "video" | null => {
    const ext = "." + (f.name.split(".").pop() ?? "").toLowerCase();
    return imageExt.includes(ext) ? "image" : videoExt.includes(ext) ? "video" : null;
  };
  const kind = file ? kindOf(file) : null;

  function choose(f: File) {
    setReport(null); setError(null); setPhase("idle"); setProgress(0);
    const k = kindOf(f);
    if (!k) { setFile(null); setClientIssue(`“${f.name}” is not a supported type. Use ${[...imageExt, ...videoExt].join(", ")}.`); return; }
    const max = k === "image" ? imageMb : videoMb;
    if (f.size > max * 1024 * 1024) { setFile(null); setClientIssue(`“${f.name}” is ${formatBytes(f.size)}, over the ${max} MB limit for ${k}s.`); return; }
    if (f.size === 0) { setFile(null); setClientIssue("That file is empty."); return; }
    setClientIssue(null); setFile(f);
  }

  function reset() { abort.current?.abort(); setFile(null); setReport(null); setError(null); setPhase("idle"); setClientIssue(null); setProgress(0); }

  async function run() {
    if (!file || !kind) return;
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    setPhase("uploading"); setProgress(0); setError(null); setReport(null);
    try {
      const r = await api.analyze(kind, file, (f) => { setProgress(f); if (f >= 1) { setPhase("analyzing"); setProgress(null); } }, ctl.signal);
      setReport(r); setPhase("done");
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setError(e instanceof ApiError ? e.message : (e as Error).message);
      // Retrying only helps when the server was unreachable or failed; a 4xx validation error would repeat.
      setRetryable(!(e instanceof ApiError) || e.status === 0 || e.status >= 500);
      setPhase("error");
    }
  }

  const busy = phase === "uploading" || phase === "analyzing";

  return (
    <>
      <PageHeader eyebrow="Workspace" title="Analyze media"
        description="Upload an image or short video. The detector's output and the retrieved research are shown separately, so you can see what the model said and what the literature says." />

      <div className="grid gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="space-y-4">
          {!file ? (
            <Dropzone accept={[...imageExt, ...videoExt]} onFile={choose} disabled={busy}
              hint={<>Images {imageExt.join(" ")} up to {imageMb} MB · Video {videoExt.join(" ")} up to {videoMb} MB and {limits?.max_video_seconds ?? 30} s</>} />
          ) : (
            <div className="panel overflow-hidden">
              <div className="flex aspect-[4/3] items-center justify-center bg-ink/[0.04]">
                {kind === "image" && previewUrl && <img src={previewUrl} alt={`Preview of ${file.name}`} className="max-h-full max-w-full object-contain" />}
                {kind === "video" && previewUrl && <video src={previewUrl} controls muted className="max-h-full max-w-full" aria-label={`Preview of ${file.name}`} />}
              </div>
              <div className="flex items-center gap-3 border-t border-line px-4 py-3">
                {kind === "image" ? <FileImage className="h-4 w-4 shrink-0 text-ink-3" aria-hidden /> : <Film className="h-4 w-4 shrink-0 text-ink-3" aria-hidden />}
                <div className="min-w-0 flex-1"><p className="truncate text-sm font-medium">{file.name}</p><p className="text-xs text-ink-3">{kind} · {formatBytes(file.size)}</p></div>
                <button onClick={reset} aria-label="Remove file" className="rounded p-1.5 text-ink-3 hover:bg-sunken hover:text-ink"><X className="h-4 w-4" /></button>
              </div>
            </div>
          )}
          {clientIssue && <Notice kind="warn" title="File not accepted">{clientIssue}</Notice>}
          {kind === "video" && <Notice kind="warn" title="Video analysis is experimental">It runs the still-image detector on up to {limits?.max_video_frames ?? 8} sampled frames and averages them. It is not a temporal deepfake detector.</Notice>}
          {busy && <ProgressBar value={progress} label={phase === "uploading" ? "Uploading" : "Analyzing: running the detector and retrieving research"} />}
          <div className="flex gap-2">
            <Button variant="primary" onClick={run} disabled={!file || busy} loading={busy}>{busy ? "Working…" : report ? "Analyze again" : "Run analysis"}</Button>
            {(file || report) && <Button variant="ghost" onClick={reset}><RotateCcw className="h-4 w-4" aria-hidden />Start over</Button>}
          </div>
          <p className="text-xs text-ink-3">
            Files are processed on the server, deleted right after analysis, and never stored. Only the report (including a SHA-256 of the file) is kept. See Methodology & privacy.
          </p>
        </div>

        <div className="min-w-0 space-y-6" aria-live="polite">
          {phase === "error" && error && <Notice kind="error" title="Analysis failed" action={retryable ? <Button size="sm" onClick={run}>Retry</Button> : undefined}>{error}</Notice>}
          {!report && phase !== "error" && (
            <div className="flex h-full min-h-64 items-center justify-center rounded-lg border border-dashed border-line-strong p-8 text-center text-sm text-ink-3">
              {busy ? "Analysis in progress…" : "Results appear here: the model prediction, its raw scores, and related research evidence."}
            </div>
          )}
          {report && (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }} className="space-y-6">
              <PredictionPanel result={report.result} detector={report.detector} experimental={report.kind === "video"} />
              {report.video && <FrameChart frames={report.video.frames} threshold={report.detector.thresholds.ai} />}
              <EvidenceSection report={report} />
              <div className="flex flex-wrap items-center gap-2 border-t border-line pt-5">
                <ButtonLink to={`/reports/${report.id}`} variant="primary" size="sm">Open full report<ArrowRight className="h-4 w-4" aria-hidden /></ButtonLink>
                <DownloadButtons id={report.id} />
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </>
  );
}
