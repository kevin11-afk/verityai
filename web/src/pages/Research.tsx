import { motion } from "framer-motion";
import { Loader2, Send, Sparkles } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { api } from "@/api/client";
import type { AskResponse } from "@/api/types";
import { CitationChip, DiagnosticsPanel, NoEvidence, PassageCard } from "@/components/Evidence";
import { Badge, Button, Notice, PageHeader } from "@/components/ui";
import { useDocumentTitle } from "@/lib/hooks";

interface Turn { id: number; q: string; status: "loading" | "done" | "error"; res?: AskResponse; err?: string }
const EXAMPLES = [
  "Why do detectors fail on images from new generators?",
  "What is the difference between frame-level and video-level deepfake detection?",
  "Why wasn't the sdxl-detector chosen?",
  "What are Content Credentials?",
];

function Answer({ res }: { res: AskResponse }) {
  const [open, setOpen] = useState<Set<number>>(new Set());
  const [diag, setDiag] = useState(false);
  const toggle = (id: number) => setOpen((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  const jump = (id: number) => {
    setOpen((s) => new Set(s).add(id));
    requestAnimationFrame(() => document.getElementById(`source-${id}`)?.scrollIntoView({ behavior: "smooth", block: "center" }));
  };
  const docs = Array.from(res.passages.reduce((m, p) => m.set(p.source_file, [...(m.get(p.source_file) ?? []), p.citation_id]), new Map<string, number[]>()));

  return (
    <div className="space-y-5">
      {res.abstained ? <NoEvidence message={res.notice ?? "Nothing in the knowledge base is relevant enough to answer this."} /> : (
        <>
          <div className="rounded-lg border border-line bg-surface p-5 shadow-panel">
            <p className="eyebrow mb-3 flex items-center gap-1.5"><Sparkles className="h-3.5 w-3.5" aria-hidden />Evidence-grounded answer</p>
            {res.answer.length > 0 ? (
              <ul className="space-y-3">
                {res.answer.map((c, i) => <li key={i} className="text-[15px] leading-relaxed">{c.text}<CitationChip n={c.citation_id} onClick={() => jump(c.citation_id)} /></li>)}
              </ul>
            ) : <p className="text-sm text-ink-2">{res.notice}</p>}
            <p className="mt-4 border-t border-line pt-3 text-xs text-ink-3">Every sentence above is quoted word-for-word from the numbered source it cites; nothing is generated or paraphrased.</p>
          </div>
          <div>
            <p className="eyebrow mb-2">Relevant documents</p>
            <div className="flex flex-wrap gap-2">
              {docs.map(([file, ids]) => (
                <button key={file} onClick={() => jump(ids[0] as number)} className="rounded-md border border-line bg-surface px-2.5 py-1 font-mono text-xs hover:bg-sunken">{file} <span className="text-ink-3">· {ids.length}</span></button>
              ))}
            </div>
          </div>
          <div>
            <p className="eyebrow mb-2">Sources</p>
            <ul className="panel overflow-hidden">
              {res.passages.map((p) => <PassageCard key={p.chunk_id} p={p} open={open.has(p.citation_id)} onToggle={() => toggle(p.citation_id)} />)}
            </ul>
          </div>
        </>
      )}
      <div>
        <button onClick={() => setDiag((d) => !d)} aria-expanded={diag} className="link text-sm">{diag ? "Hide" : "Show"} retrieval diagnostics</button>
        {diag && <div className="mt-2"><DiagnosticsPanel d={res.diagnostics} /></div>}
      </div>
    </div>
  );
}

export function Research() {
  useDocumentTitle("Research");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const idRef = useRef(0);
  const endRef = useRef<HTMLDivElement>(null);
  const aborts = useRef<AbortController[]>([]);
  const busy = turns.some((t) => t.status === "loading");

  useEffect(() => () => aborts.current.forEach((a) => a.abort()), []);
  useEffect(() => { if (turns.length) endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [turns.length]);

  async function ask(question: string) {
    const q = question.trim();
    if (q.length < 3 || busy) return;
    const id = ++idRef.current;
    const ctl = new AbortController();
    aborts.current.push(ctl);
    setTurns((t) => [...t, { id, q, status: "loading" }]);
    setText("");
    try {
      const res = await api.ask(q, 5, ctl.signal);
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, status: "done", res } : x)));
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, status: "error", err: (e as Error).message } : x)));
    }
  }
  const submit = (e: FormEvent) => { e.preventDefault(); void ask(text); };

  return (
    <>
      <PageHeader eyebrow="Intelligence" title="Research"
        description="Ask about AI-media detection. Answers are assembled only from retrieved passages in the knowledge base, each with a citation. When nothing relevant exists, it says so." />

      <div className="mx-auto max-w-3xl">
        {turns.length === 0 && (
          <div className="mb-8">
            <p className="eyebrow mb-3">Try asking</p>
            <div className="grid gap-2 sm:grid-cols-2">
              {EXAMPLES.map((e) => <button key={e} onClick={() => void ask(e)} className="rounded-lg border border-line bg-surface p-3.5 text-left text-sm leading-snug hover:border-line-strong hover:bg-canvas">{e}</button>)}
            </div>
            <p className="mt-6 text-xs text-ink-3">The knowledge base covers detector methods, generalization, video/deepfake detection, model cards, content provenance and evaluation methodology. Add your own documents in Knowledge base.</p>
          </div>
        )}

        <div className="space-y-10" aria-live="polite">
          {turns.map((t) => (
            <motion.article key={t.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              <h2 className="mb-4 flex gap-3 text-[17px] font-semibold leading-snug"><Badge className="mt-0.5 h-fit">Q</Badge>{t.q}</h2>
              {t.status === "loading" && <p className="flex items-center gap-2 text-sm text-ink-3"><Loader2 className="h-4 w-4 animate-spin" aria-hidden />Searching the knowledge base…</p>}
              {t.status === "error" && <Notice kind="error" title="The question could not be answered" action={<Button size="sm" onClick={() => void ask(t.q)}>Retry</Button>}>{t.err}</Notice>}
              {t.status === "done" && t.res && <Answer res={t.res} />}
            </motion.article>
          ))}
          <div ref={endRef} />
        </div>

        <form onSubmit={submit} className="sticky bottom-4 mt-8 flex items-end gap-2 rounded-xl border border-line-strong bg-surface p-2 shadow-pop focus-within:border-brand focus-within:ring-2 focus-within:ring-brand/30">
          <label htmlFor="q" className="sr-only">Your question</label>
          <textarea id="q" rows={1} value={text} onChange={(e) => setText(e.target.value)} maxLength={500} placeholder="Ask a question about AI-media detection…"
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void ask(text); } }}
            className="max-h-32 min-h-10 flex-1 resize-none bg-transparent px-3 py-2 text-[15px] outline-none placeholder:text-ink-3" />
          <Button type="submit" variant="primary" disabled={busy || text.trim().length < 3} loading={busy} aria-label="Ask"><Send className="h-4 w-4" aria-hidden /></Button>
        </form>
        <p className="mt-2 text-center text-xs text-ink-3">Enter to send · Shift+Enter for a new line · {text.length}/500</p>
      </div>
    </>
  );
}
