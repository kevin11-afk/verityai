import { Download, ExternalLink, FileSearch, Search } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "@/api/client";
import { Badge, Button, ButtonLink, EmptyState, Notice, PageHeader, Skeleton } from "@/components/ui";
import { categoryTone, formatBytes, formatDate, shortId } from "@/lib/format";
import { useAsync, useDebounced, useDocumentTitle } from "@/lib/hooks";

export function Reports() {
  useDocumentTitle("Reports");
  const [q, setQ] = useState("");
  const dq = useDebounced(q);
  const { data, error, loading, reload } = useAsync((s) => api.reports(dq, s), [dq]);

  return (
    <>
      <PageHeader eyebrow="Workspace" title="Reports" description="Every analysis you run is saved as a report on the server. Open one to review the evidence or download it." />
      {data && <div className="mb-5"><Notice kind={data.persistence_mode === "persistent" ? "info" : "warn"} title={data.persistence_mode === "persistent" ? "Saved on persistent storage" : "History may not persist"}>{data.persistence_note}</Notice></div>}

      <div className="relative mb-4 max-w-md">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-3" aria-hidden />
        <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search by file name or result" aria-label="Search reports"
          className="h-10 w-full rounded-md border border-line-strong bg-surface pl-9 pr-3 text-sm" />
      </div>

      {error && <Notice kind="error" title="Could not load reports" action={<Button size="sm" onClick={reload}>Retry</Button>}>{error.message}</Notice>}
      {loading && !data && <div className="space-y-2" aria-busy="true">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-12 w-full" />)}</div>}
      {data && data.reports.length === 0 && (
        <div className="panel"><EmptyState icon={<FileSearch className="h-5 w-5" />} title={dq ? "No reports match your search" : "No reports yet"}
          action={!dq && <ButtonLink to="/analyze" variant="primary">Analyze media</ButtonLink>}>
          {dq ? "Try a different file name or result." : "Run an analysis and its report will be saved here."}
        </EmptyState></div>
      )}
      {data && data.reports.length > 0 && (
        <div className="panel overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-sm">
            <caption className="sr-only">Saved reports</caption>
            <thead className="border-b border-line bg-sunken/50 text-xs text-ink-3">
              <tr>{["Date", "File", "Result", "Detector", "Size", ""].map((h) => <th key={h} scope="col" className="px-4 py-2.5 font-medium">{h}</th>)}</tr>
            </thead>
            <tbody>
              {data.reports.map((r) => (
                <tr key={r.id} className="border-b border-line last:border-0 hover:bg-canvas">
                  <td className="whitespace-nowrap px-4 py-3 text-ink-3">{formatDate(r.created_at)}</td>
                  <td className="px-4 py-3"><Link to={`/reports/${r.id}`} className="font-medium hover:underline">{r.file_name}</Link><span className="ml-2 font-mono text-[11px] text-ink-3">{shortId(r.id)}</span></td>
                  <td className="px-4 py-3"><Badge tone={categoryTone(r.category)}>{r.category}</Badge>{r.kind === "video" && <span className="ml-2 text-xs text-ink-3">video</span>}</td>
                  <td className="max-w-[220px] truncate px-4 py-3 text-ink-3" title={r.detector_version}>{r.detector_version}</td>
                  <td className="whitespace-nowrap px-4 py-3 text-ink-3 tnum">{formatBytes(r.size_bytes)}</td>
                  <td className="whitespace-nowrap px-4 py-3 text-right">
                    <a href={api.reportDownloadUrl(r.id, "json")} download aria-label={`Download JSON for ${r.file_name}`} className="mr-1 inline-flex rounded p-1.5 text-ink-3 hover:bg-sunken hover:text-ink"><Download className="h-4 w-4" /></a>
                    <Link to={`/reports/${r.id}`} aria-label={`Open report for ${r.file_name}`} className="inline-flex rounded p-1.5 text-ink-3 hover:bg-sunken hover:text-ink"><ExternalLink className="h-4 w-4" /></Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
