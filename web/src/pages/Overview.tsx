import { ArrowRight, FlaskConical, MessagesSquare, ScanSearch, UploadCloud } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "@/api/client";
import { Badge, ButtonLink, EmptyState, Notice, PageHeader, Section, Skeleton, Stat } from "@/components/ui";
import { categoryTone, formatDate, pct, shortId } from "@/lib/format";
import { useAsync, useDocumentTitle } from "@/lib/hooks";
import { useHealth } from "@/lib/health";

export function Overview() {
  useDocumentTitle("Overview");
  const { health, error: healthError, loading: healthLoading } = useHealth();
  const reports = useAsync((s) => api.reports("", s), []);
  const evalRes = useAsync((s) => api.evaluation(s), []);

  const prod = evalRes.data?.detector?.models.find((m) => m.is_production)?.variants["raw"];
  const rag = evalRes.data?.rag;
  const recent = reports.data?.reports.slice(0, 6) ?? [];

  return (
    <>
      <PageHeader eyebrow="Workspace" title="Overview"
        description="Screen images and video for signs of AI generation, with every result tied to a transparent detector, its limits, and cited research."
        actions={<>
          <ButtonLink to="/research"><MessagesSquare className="h-4 w-4" aria-hidden />Ask research</ButtonLink>
          <ButtonLink to="/analyze" variant="primary"><ScanSearch className="h-4 w-4" aria-hidden />Analyze media</ButtonLink>
        </>} />

      {healthError && <div className="mb-6"><Notice kind="error" title="The VerityAI server is unreachable">{healthError.message} Use the connection control at the bottom of the sidebar to check the API address.</Notice></div>}

      <section aria-label="System status" className="panel mb-10 grid grid-cols-2 divide-line sm:divide-x lg:grid-cols-4">
        {healthLoading && !health ? [0, 1, 2, 3].map((i) => <div key={i} className="px-5 py-4"><Skeleton className="h-3 w-16" /><Skeleton className="mt-3 h-5 w-28" /></div>) : <>
          <Stat label="Detector" value={health?.detector.name ?? "Unavailable"} sub={health ? (health.detector.loaded ? "Loaded in memory" : "Loads on first analysis") : undefined} mono />
          <Stat label="Knowledge base" value={health ? `${health.knowledge_base_documents} docs · ${health.knowledge_base_chunks} passages` : "n/a"} sub={health ? (health.knowledge_base_ready ? "Index ready" : "Index warming up") : undefined} />
          <Stat label="Report storage" value={health ? (health.persistence_mode === "persistent" ? "Persistent" : "Temporary") : "n/a"} sub={health?.persistence_mode === "ephemeral" ? "May reset on restart" : undefined} />
          <Stat label="System" value={health ? (health.status === "ok" ? "Healthy" : "Warming up") : "Offline"} sub={health ? `API v${health.version}${health.rss_mb ? ` · peak ${Math.round(health.rss_mb)} MB` : ""}` : undefined} />
        </>}
      </section>

      <div className="grid gap-10 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Section title="Recent analyses" actions={recent.length > 0 ? <Link to="/reports" className="link text-sm">All reports</Link> : undefined}>
          {reports.loading && !reports.data && <div className="space-y-2" aria-busy="true">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-12" />)}</div>}
          {reports.error && <Notice kind="error" title="Could not load reports">{reports.error.message}</Notice>}
          {reports.data && recent.length === 0 && (
            <div className="panel">
              <EmptyState icon={<UploadCloud className="h-5 w-5" />} title="No analyses yet" action={<ButtonLink to="/analyze" variant="primary">Analyze your first file</ButtonLink>}>
                Upload an image or a short video. You will get the detector's raw scores, the policy decision, and related research with citations.
              </EmptyState>
            </div>
          )}
          {recent.length > 0 && (
            <ul className="panel divide-y divide-line">
              {recent.map((r) => (
                <li key={r.id}>
                  <Link to={`/reports/${r.id}`} className="flex items-center gap-3 px-4 py-3 hover:bg-canvas">
                    <div className="min-w-0 flex-1"><p className="truncate text-sm font-medium">{r.file_name}</p><p className="text-xs text-ink-3">{formatDate(r.created_at)} · <span className="font-mono">{shortId(r.id)}</span></p></div>
                    <Badge tone={categoryTone(r.category)}>{r.category}</Badge>
                    <ArrowRight className="h-4 w-4 shrink-0 text-ink-3" aria-hidden />
                  </Link>
                </li>
              ))}
            </ul>
          )}
          {reports.data?.persistence_mode === "ephemeral" && recent.length > 0 && <p className="mt-2 text-xs text-ink-3">{reports.data.persistence_note}</p>}
        </Section>

        <Section title="Measured quality" actions={<Link to="/evaluation" className="link inline-flex items-center gap-1 text-sm"><FlaskConical className="h-3.5 w-3.5" aria-hidden />Evaluation lab</Link>}>
          {evalRes.loading && !evalRes.data && <Skeleton className="h-44" />}
          {evalRes.data && (
            <div className="panel divide-y divide-line">
              <div className="p-4">
                <p className="eyebrow">Detector · ROC-AUC</p>
                {prod && evalRes.data.detector ? <>
                  <p className="mt-1 text-2xl font-semibold tnum">{prod.auc.toFixed(2)} <span className="text-sm font-normal text-ink-3">95% CI {prod.auc_ci[0].toFixed(2)}–{prod.auc_ci[1].toFixed(2)}</span></p>
                  <p className="mt-1 text-xs text-ink-3">{evalRes.data.detector.dataset.n_real + evalRes.data.detector.dataset.n_ai} images, one dataset. At the policy threshold, {pct(1 - prod.at_policy_threshold.real_recall, 0)} of real photos were flagged as AI.</p>
                </> : <p className="mt-1 text-lg font-semibold text-ink-3">Not evaluated</p>}
              </div>
              <div className="p-4">
                <p className="eyebrow">Research retrieval · Hit@1</p>
                {rag ? <>
                  <p className="mt-1 text-2xl font-semibold tnum">{pct(rag.retrieval.hit_at_1.rate, 1)} <span className="text-sm font-normal text-ink-3">Hit@5 {pct(rag.retrieval.hit_at_5.rate, 1)} · MRR {rag.retrieval.mrr.toFixed(2)}</span></p>
                  <p className="mt-1 text-xs text-ink-3">{rag.question_set.answerable} labeled questions; {rag.abstention["off_topic"]?.abstained ?? 0}/{rag.abstention["off_topic"]?.n ?? 0} off-topic questions refused.</p>
                </> : <p className="mt-1 text-lg font-semibold text-ink-3">Not evaluated</p>}
              </div>
            </div>
          )}
          <p className="mt-2 text-xs text-ink-3">Computed from stored evaluation artifacts, never typed in. Small samples: read the confidence intervals.</p>
        </Section>
      </div>
    </>
  );
}
