import { ChevronDown, ExternalLink, FileSearch } from "lucide-react";
import { useState } from "react";
import type { Passage, RetrievalDiag } from "@/api/types";
import { Badge, cx } from "@/components/ui";
import { safeUrl } from "@/lib/format";

export function CitationChip({ n, onClick }: { n: number; onClick?: () => void }) {
  const cls = "mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-brand-soft px-1 align-text-top font-mono text-[11px] font-semibold text-brand-ink";
  return onClick
    ? <button type="button" onClick={onClick} className={cx(cls, "hover:bg-brand hover:text-white")} aria-label={`Go to source ${n}`}>{n}</button>
    : <span className={cls}>{n}</span>;
}

/** Retrieved text is untrusted: rendered strictly as plain text (React escapes it), never as HTML. */
export function PassageCard({ p, open, onToggle }: { p: Passage; open: boolean; onToggle: () => void }) {
  const url = safeUrl(p.source_url);
  return (
    <li id={`source-${p.citation_id}`} className="scroll-mt-24 border-b border-line last:border-0">
      <button type="button" onClick={onToggle} aria-expanded={open} className="flex w-full items-start gap-3 px-4 py-3 text-left hover:bg-canvas">
        <CitationChip n={p.citation_id} />
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-semibold">{p.title}</span>
          <span className="block truncate font-mono text-xs text-ink-3">{p.source_file}{p.page_number ? ` · p.${p.page_number}` : ""}</span>
          <span className="mt-1.5 flex flex-wrap gap-1.5">
            {p.vector_rank != null && <Badge>semantic rank {p.vector_rank}</Badge>}
            {p.bm25_rank != null && <Badge>keyword rank {p.bm25_rank}</Badge>}
          </span>
        </span>
        <ChevronDown className={cx("mt-0.5 h-4 w-4 shrink-0 text-ink-3 transition-transform", open && "rotate-180")} aria-hidden />
      </button>
      {open && (
        <div className="px-4 pb-4 pl-[3.25rem]">
          <blockquote className="whitespace-pre-wrap break-words border-l-2 border-line-strong pl-3 text-[13px] leading-relaxed text-ink-2">{p.text.trim()}</blockquote>
          <dl className="mt-3 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[11px] text-ink-3">
            <div>fused score {p.rrf_score.toFixed(4)}</div>
            {p.vector_distance != null && <div>vector distance {p.vector_distance.toFixed(3)}</div>}
            <div>chunk {p.chunk_id.split(":").slice(1).join(":")}</div>
          </dl>
          {url && <a href={url} target="_blank" rel="noopener noreferrer" className="link mt-2 inline-flex items-center gap-1 text-xs">Original source <ExternalLink className="h-3 w-3" aria-hidden /></a>}
        </div>
      )}
    </li>
  );
}

export function PassageList({ passages }: { passages: Passage[] }) {
  const [open, setOpen] = useState<Set<number>>(new Set());
  const toggle = (id: number) => setOpen((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  return (
    <ul className="panel overflow-hidden">
      {passages.map((p) => <PassageCard key={p.chunk_id} p={p} open={open.has(p.citation_id)} onToggle={() => toggle(p.citation_id)} />)}
    </ul>
  );
}

export function NoEvidence({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-dashed border-line-strong bg-canvas p-4 text-sm">
      <FileSearch className="mt-0.5 h-4 w-4 shrink-0 text-ink-3" aria-hidden />
      <div><p className="font-semibold">Insufficient evidence</p><p className="mt-0.5 text-ink-3">{message}</p></div>
    </div>
  );
}

export function DiagnosticsPanel({ d }: { d: RetrievalDiag }) {
  const best = d.best_vector_distance;
  const scale = Math.max(1, d.distance_threshold * 1.4, best ?? 0);
  return (
    <div className="rounded-lg border border-line bg-canvas p-4 text-sm">
      <p className="eyebrow mb-3">Retrieval diagnostics</p>
      <div className="mb-4">
        <div className="mb-1 flex justify-between text-xs text-ink-3">
          <span>Best vector distance (lower is closer)</span>
          <span className="font-mono tnum">{best != null ? best.toFixed(3) : "n/a"} / threshold {d.distance_threshold.toFixed(2)}</span>
        </div>
        <div className="relative h-2 rounded-full bg-sunken" aria-hidden>
          {best != null && <div className={cx("h-full rounded-full", best <= d.distance_threshold ? "bg-teal" : "bg-signal-ai")} style={{ width: `${(best / scale) * 100}%` }} />}
          <div className="absolute -top-1 h-4 border-l-2 border-ink/70" style={{ left: `${(d.distance_threshold / scale) * 100}%` }} />
        </div>
      </div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-[13px] sm:grid-cols-3">
        <div><dt className="text-ink-3">Relevance gate</dt><dd className="font-medium">{d.abstained ? "Refused" : d.gate_passed_by === "lexical" ? "Passed (key terms)" : "Passed (vector)"}</dd></div>
        <div><dt className="text-ink-3">Corpus chunks</dt><dd className="font-mono">{d.corpus_chunks}</dd></div>
        <div><dt className="text-ink-3">Passages returned</dt><dd className="font-mono">top {d.top_k}</dd></div>
        <div><dt className="text-ink-3">Vector candidates</dt><dd className="font-mono">{d.vector_candidates}</dd></div>
        <div><dt className="text-ink-3">BM25 candidates</dt><dd className="font-mono">{d.bm25_candidates}</dd></div>
        <div><dt className="text-ink-3">Key-term coverage</dt><dd className="font-mono">{d.lexical_coverage != null ? `${Math.round(d.lexical_coverage * 100)}%` : "n/a"}</dd></div>
        <div><dt className="text-ink-3">Weak passages dropped</dt><dd className="font-mono">{d.weak_filtered ?? 0}</dd></div>
      </dl>
      {d.abstain_reason && <p className="mt-3 text-[13px] text-ink-2">{d.abstain_reason}</p>}
    </div>
  );
}
