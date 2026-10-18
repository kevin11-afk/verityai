import { FileText, Lock, RefreshCw, Search, Trash2 } from "lucide-react";
import { useState, type FormEvent } from "react";
import { api } from "@/api/client";
import type { AskResponse, KBDocument } from "@/api/types";
import { Dropzone } from "@/components/Dropzone";
import { DiagnosticsPanel, NoEvidence, PassageList } from "@/components/Evidence";
import { Badge, Button, Dialog, EmptyState, Notice, PageHeader, ProgressBar, Section, Skeleton } from "@/components/ui";
import { formatBytes } from "@/lib/format";
import { useAsync, useDocumentTitle } from "@/lib/hooks";
import { useHealth } from "@/lib/health";

export function Knowledge() {
  useDocumentTitle("Knowledge base");
  const { refresh: refreshHealth } = useHealth();
  const list = useAsync((s) => api.kbList(s), []);
  const [url, setUrl] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<{ kind: "success" | "warn" | "error"; text: string } | null>(null);
  const [removing, setRemoving] = useState<KBDocument | null>(null);
  const [reindex, setReindex] = useState(false);
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState<{ loading: boolean; res?: AskResponse; err?: string }>({ loading: false });

  async function upload(file: File) {
    setUploading(true); setResult(null); setProgress(0);
    try {
      const r = await api.kbUpload(file, url, (f) => setProgress(f >= 1 ? null : f));
      setResult({ kind: r.status === "duplicate" ? "warn" : "success", text: r.message });
      setUrl(""); list.reload(); refreshHealth();
    } catch (e) { setResult({ kind: "error", text: (e as Error).message }); }
    finally { setUploading(false); setProgress(null); }
  }

  async function confirmRemove() {
    if (!removing) return;
    setBusy(true);
    try { const r = await api.kbRemove(removing.source_file); setResult({ kind: "success", text: r.message }); list.reload(); refreshHealth(); }
    catch (e) { setResult({ kind: "error", text: (e as Error).message }); }
    finally { setBusy(false); setRemoving(null); }
  }

  async function confirmReindex() {
    setBusy(true);
    try { const r = await api.kbReindex(); setResult({ kind: "success", text: `${r.message} ${r.chunks} passages indexed.` }); list.reload(); refreshHealth(); }
    catch (e) { setResult({ kind: "error", text: (e as Error).message }); }
    finally { setBusy(false); setReindex(false); }
  }

  async function runSearch(e: FormEvent) {
    e.preventDefault();
    if (query.trim().length < 2) return;
    setSearch({ loading: true });
    try { setSearch({ loading: false, res: await api.search(query.trim(), 5) }); }
    catch (err) { setSearch({ loading: false, err: (err as Error).message }); }
  }

  const docs = list.data?.documents ?? [];
  return (
    <>
      <PageHeader eyebrow="Intelligence" title="Knowledge base"
        description="The documents the Research workspace and reports draw on. Add your own PDF, text or Markdown files; each is split into passages and indexed for hybrid keyword + semantic search."
        actions={<Button onClick={() => setReindex(true)}><RefreshCw className="h-4 w-4" aria-hidden />Re-index all</Button>} />

      {list.data && <div className="mb-6"><Notice kind={list.data.persistence_mode === "persistent" ? "info" : "warn"} title={list.data.persistence_mode === "persistent" ? "Uploads are stored persistently" : "Uploaded documents may not persist"}>{list.data.persistence_note}</Notice></div>}
      {result && <div className="mb-6"><Notice kind={result.kind === "success" ? "success" : result.kind}>{result.text}</Notice></div>}

      <div className="grid gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div>
          <Section title="Add a document" description="PDF, TXT or Markdown up to 20 MB. Scanned PDFs without text are not supported.">
            <Dropzone accept={[".pdf", ".txt", ".md"]} onFile={(f) => void upload(f)} disabled={uploading} hint="Identical files are detected and not indexed twice.">
              {uploading && <div className="mt-4 text-left"><ProgressBar value={progress} label={progress === null ? "Indexing passages" : "Uploading"} /></div>}
            </Dropzone>
            <label htmlFor="src-url" className="mb-1 mt-4 block text-xs font-semibold text-ink-2">Source URL (optional)</label>
            <input id="src-url" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://… (shown as the link in citations)" disabled={uploading}
              className="h-10 w-full rounded-md border border-line-strong bg-surface px-3 text-sm" inputMode="url" />
            <p className="mt-1 text-xs text-ink-3">Set this before choosing the file.</p>
          </Section>

          <Section title="Test retrieval" description="Run a query against the index without generating an answer, and inspect exactly what comes back and why.">
            <form onSubmit={runSearch} className="flex gap-2">
              <div className="relative flex-1">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-3" aria-hidden />
                <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. frequency artifacts" aria-label="Retrieval test query" maxLength={500}
                  className="h-10 w-full rounded-md border border-line-strong bg-surface pl-9 pr-3 text-sm" />
              </div>
              <Button type="submit" loading={search.loading} disabled={query.trim().length < 2}>Search</Button>
            </form>
          </Section>
        </div>

        <div className="min-w-0">
          <Section title={`Indexed documents${list.data ? ` · ${list.data.documents.length}` : ""}`} description={list.data ? `${list.data.total_chunks} passages in the index.` : undefined}>
            {list.error && <Notice kind="error" title="Could not load documents" action={<Button size="sm" onClick={list.reload}>Retry</Button>}>{list.error.message}</Notice>}
            {list.loading && !list.data && <div className="space-y-2" aria-busy="true">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-14" />)}</div>}
            {list.data && docs.length === 0 && <div className="panel"><EmptyState icon={<FileText className="h-5 w-5" />} title="The knowledge base is empty">Upload a document to start. Reports and research answers will show “insufficient evidence” until then.</EmptyState></div>}
            {docs.length > 0 && (
              <ul className="panel divide-y divide-line">
                {docs.map((d) => (
                  <li key={d.source_file} className="flex items-start gap-3 px-4 py-3">
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-semibold">{d.title}</p>
                      <p className="truncate font-mono text-xs text-ink-3">{d.source_file}{d.size_bytes ? ` · ${formatBytes(d.size_bytes)}` : ""}</p>
                      <div className="mt-1.5 flex flex-wrap gap-1.5">
                        <Badge tone={d.indexed ? "real" : "unsure"}>{d.indexed ? `${d.chunks} passages` : "Not indexed"}</Badge>
                        <Badge>{d.origin === "seed" ? "Built-in" : "Uploaded"}</Badge>
                        {d.duplicate_of && d.duplicate_of !== d.source_file && <Badge tone="unsure">Duplicate of {d.duplicate_of}</Badge>}
                        {d.source_url && <Badge>Has source link</Badge>}
                      </div>
                      {d.sha256 && <p className="mt-1 font-mono text-[10px] text-ink-3" title={d.sha256}>sha256 {d.sha256.slice(0, 16)}…</p>}
                    </div>
                    {d.removable
                      ? <Button size="sm" variant="ghost" onClick={() => setRemoving(d)} aria-label={`Remove ${d.source_file}`}><Trash2 className="h-4 w-4" aria-hidden /></Button>
                      : <span className="mt-1 text-ink-3" title="Built-in documents are read-only"><Lock className="h-4 w-4" aria-label="Read-only" /></span>}
                  </li>
                ))}
              </ul>
            )}
          </Section>
        </div>
      </div>

      {(search.loading || search.res || search.err) && (
        <Section title="Retrieval result" className="mt-2">
          {search.err && <Notice kind="error">{search.err}</Notice>}
          {search.res && (
            <div className="space-y-4">
              {search.res.passages.length ? <PassageList passages={search.res.passages} /> : <NoEvidence message={search.res.notice ?? "Nothing relevant was found."} />}
              <DiagnosticsPanel d={search.res.diagnostics} />
            </div>
          )}
        </Section>
      )}

      <Dialog open={!!removing} onClose={() => setRemoving(null)} title="Remove this document?"
        footer={<><Button variant="ghost" onClick={() => setRemoving(null)}>Cancel</Button><Button variant="danger" loading={busy} onClick={confirmRemove}>Remove document</Button></>}>
        <p><span className="font-mono font-medium">{removing?.source_file}</span> and its {removing?.chunks} indexed passages will be deleted from the server. Research answers and reports will no longer cite it. This cannot be undone.</p>
      </Dialog>
      <Dialog open={reindex} onClose={() => setReindex(false)} title="Re-index everything?"
        footer={<><Button variant="ghost" onClick={() => setReindex(false)}>Cancel</Button><Button variant="primary" loading={busy} onClick={confirmReindex}>Rebuild index</Button></>}>
        The whole index is dropped and rebuilt from every document. Search is briefly unavailable and rebuilding can take a minute on a small server.
      </Dialog>
    </>
  );
}
