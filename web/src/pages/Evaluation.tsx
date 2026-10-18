import { Download, FlaskConical } from "lucide-react";
import { useMemo, useState } from "react";
import { api } from "@/api/client";
import type { DetectorEval, RagEval } from "@/api/types";
import { AblationChart, CalibrationChart, ConfusionMatrix, PrChart, RocChart, SweepChart, TradeoffChart } from "@/components/charts";
import { Badge, Button, EmptyState, Notice, PageHeader, Section, Skeleton, Tabs, cx } from "@/components/ui";
import { download, formatDate, pct, pctCi } from "@/lib/format";
import { useAsync, useDocumentTitle } from "@/lib/hooks";
import { balancedThreshold, confusionAt, rates, toCsv, tradeoffCurve } from "@/lib/metrics";

const NOT_EVALUATED = (
  <div className="panel"><EmptyState icon={<FlaskConical className="h-5 w-5" />} title="Not evaluated">
    No evaluation artifact was found on the server. Generate real results with <span className="font-mono text-xs">python scripts/export_evaluation.py all</span> (see Methodology for the data it needs).
  </EmptyState></div>
);

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return <div className="px-4 py-3.5"><p className="eyebrow">{label}</p><p className="mt-1 text-xl font-semibold tnum">{value}</p>{sub && <p className="mt-0.5 text-[11px] text-ink-3">{sub}</p>}</div>;
}
const f = (v: number | null, d = 3) => (v === null ? "n/a" : v.toFixed(d));

const EMPTY: number[] = [];

function DetectorLab({ ev }: { ev: DetectorEval }) {
  const production = ev.models.find((m) => m.is_production) ?? ev.models[0];
  const [modelId, setModelId] = useState(production?.id ?? "");
  const [variant, setVariant] = useState("raw");
  const policy = ev.policy.ai_threshold;
  const [thr, setThr] = useState(policy);

  const model = ev.models.find((m) => m.id === modelId) ?? production;
  const v = model?.variants[variant] ?? model?.variants["raw"];
  const labels = v?.samples.labels ?? EMPTY, scores = v?.samples.scores ?? EMPTY;
  const conf = useMemo(() => confusionAt(labels, scores, thr), [labels, scores, thr]);
  const r = rates(conf);
  const curve = useMemo(() => tradeoffCurve(labels, scores), [labels, scores]);
  const balanced = useMemo(() => balancedThreshold(labels, scores), [labels, scores]);
  if (!model || !v) return NOT_EVALUATED;

  const atPolicy = Math.abs(thr - policy) < 1e-9;
  const base = ev.dataset.n_ai / (ev.dataset.n_ai + ev.dataset.n_real);
  const po = v.at_policy_threshold.policy_outcomes;

  const exportMetrics = () => download(`verityai_metrics_${model.id.replace("/", "_")}_${variant}.csv`, toCsv([{
    model: model.id, variant, threshold: thr, n: r.n, tp: conf.tp, fp: conf.fp, tn: conf.tn, fn: conf.fn,
    accuracy: r.accuracy, precision: r.precision, recall: r.recall, specificity: r.specificity, fpr: r.fpr, fnr: r.fnr, roc_auc: v.auc,
    roc_auc_ci_low: v.auc_ci[0], roc_auc_ci_high: v.auc_ci[1], dataset: ev.dataset.name, artifact_commit: ev.git_commit,
  }]), "text/csv");
  const exportSamples = () => download(`verityai_scores_${model.id.replace("/", "_")}_${variant}.csv`, toCsv(labels.map((l, i) => ({ sample: i, label: l === 1 ? "ai" : "real", ai_score: scores[i] ?? null }))), "text/csv");
  const exportJson = () => download("verityai_detector_eval.json", JSON.stringify(ev, null, 2), "application/json");

  return (
    <>
      <div className="mb-6 grid gap-3 text-sm sm:grid-cols-2">
        <div className="rounded-lg border border-line bg-surface p-4">
          <p className="eyebrow mb-1.5">Dataset</p>
          <p className="font-medium">{ev.dataset.name} <span className="text-ink-3">({ev.dataset.split} split)</span></p>
          <p className="text-ink-3">{ev.dataset.sampling}: {ev.dataset.n_real} real, {ev.dataset.n_ai} AI.</p>
          <p className="mt-1 text-xs text-ink-3">{ev.dataset.license_note}</p>
        </div>
        <div className="rounded-lg border border-line bg-surface p-4">
          <p className="eyebrow mb-1.5">Provenance of these numbers</p>
          <p className="text-ink-3">Generated {formatDate(ev.generated_at)} from commit <span className="font-mono">{ev.git_commit}</span>. Policy threshold {policy.toFixed(2)}; production model <span className="font-mono">{ev.policy.production_model}</span>.</p>
          <div className="mt-2 flex flex-wrap gap-2">
            <Button size="sm" onClick={exportMetrics}><Download className="h-3.5 w-3.5" aria-hidden />Metrics CSV</Button>
            <Button size="sm" onClick={exportSamples}><Download className="h-3.5 w-3.5" aria-hidden />Scores CSV</Button>
            <Button size="sm" onClick={exportJson}><Download className="h-3.5 w-3.5" aria-hidden />Full artifact</Button>
          </div>
        </div>
      </div>

      <Notice kind="warn" title="Read before quoting any number">
        <ul className="list-disc space-y-0.5 pl-4">{ev.dataset.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
      </Notice>

      <div className="mt-8 flex flex-wrap items-end gap-x-6 gap-y-3">
        <div>
          <label htmlFor="model" className="mb-1 block text-xs font-semibold text-ink-2">Model</label>
          <select id="model" value={model.id} onChange={(e) => setModelId(e.target.value)} className="h-10 rounded-md border border-line-strong bg-surface px-3 text-sm">
            {ev.models.map((m) => <option key={m.id} value={m.id}>{m.id}{m.is_production ? " (production)" : ""}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="variant" className="mb-1 block text-xs font-semibold text-ink-2">Image variant</label>
          <select id="variant" value={variant} onChange={(e) => setVariant(e.target.value)} className="h-10 rounded-md border border-line-strong bg-surface px-3 text-sm">
            {Object.keys(model.variants).map((k) => <option key={k} value={k}>{k === "raw" ? "Original files" : "Normalized (512 px JPEG)"}</option>)}
          </select>
        </div>
        <div className="min-w-0 flex-1 basis-72">
          <label htmlFor="thr" className="mb-1 flex justify-between text-xs font-semibold text-ink-2"><span>Decision threshold (AI score ≥)</span><span className="font-mono text-sm tnum">{thr.toFixed(2)}</span></label>
          <input id="thr" type="range" min={0.05} max={0.95} step={0.01} value={thr} onChange={(e) => setThr(Number(e.target.value))} className="h-10 w-full accent-[#3F3CBB]" aria-valuetext={`${thr.toFixed(2)}`} />
          <div className="mt-1 flex gap-2">
            <Button size="sm" variant={atPolicy ? "primary" : "secondary"} onClick={() => setThr(policy)}>Policy {policy.toFixed(2)}</Button>
            <Button size="sm" onClick={() => setThr(Math.round(balanced * 100) / 100)}>Balanced {balanced.toFixed(2)}</Button>
          </div>
        </div>
      </div>
      {!atPolicy && <p className="mt-2 text-xs text-signal-unsure">Moving the threshold on the same samples used to measure it overstates real-world performance; the policy value was fixed before measurement. Intervals below are shown only at the policy threshold.</p>}

      {!model.commercial_use && <div className="mt-4"><Notice kind="warn" title="License restricts commercial use">{model.id} is {model.license}. {model.note}.</Notice></div>}

      <section aria-label="Headline metrics" className="panel mt-6 grid grid-cols-2 divide-line sm:grid-cols-3 sm:divide-x lg:grid-cols-6">
        <Metric label="ROC-AUC" value={f(v.auc, 3)} sub={`95% CI ${f(v.auc_ci[0], 2)}–${f(v.auc_ci[1], 2)}`} />
        <Metric label="Avg. precision" value={f(v.average_precision, 3)} />
        <Metric label="Accuracy" value={r.accuracy === null ? "n/a" : pct(r.accuracy, 1)} sub={atPolicy ? pctCi(v.at_policy_threshold.accuracy, v.at_policy_threshold.accuracy_ci) : undefined} />
        <Metric label="AI caught (recall)" value={r.recall === null ? "n/a" : pct(r.recall, 1)} sub={atPolicy ? pctCi(v.at_policy_threshold.ai_recall, v.at_policy_threshold.ai_recall_ci) : undefined} />
        <Metric label="Real passed" value={r.specificity === null ? "n/a" : pct(r.specificity, 1)} sub={atPolicy ? pctCi(v.at_policy_threshold.real_recall, v.at_policy_threshold.real_recall_ci) : undefined} />
        <Metric label="Precision" value={r.precision === null ? "n/a" : pct(r.precision, 1)} sub={`${conf.fp} false positives`} />
      </section>

      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        <RocChart points={v.roc} marker={{ fpr: r.fpr ?? 0, tpr: r.recall ?? 0 }} />
        <PrChart points={v.pr} base={base} marker={r.precision !== null && r.recall !== null ? { recall: r.recall, precision: r.precision } : null} />
        <TradeoffChart curve={curve} threshold={thr} policy={policy} />
        <ConfusionMatrix c={conf} />
        <CalibrationChart bins={v.calibration.bins} valid={v.calibration.valid} />
        <div className="panel p-4">
          <p className="text-sm font-semibold">Calibration summary</p>
          <dl className="mt-3 grid grid-cols-2 gap-3 text-sm">
            <div><dt className="text-ink-3">Expected calibration error</dt><dd className="font-mono text-lg">{v.calibration.ece.toFixed(3)}</dd></div>
            <div><dt className="text-ink-3">Brier score</dt><dd className="font-mono text-lg">{v.calibration.brier.toFixed(3)}</dd></div>
          </dl>
          <p className="mt-3 text-xs text-ink-3">{v.calibration.note}</p>
          {!v.calibration.valid && <p className="mt-2 text-xs text-signal-unsure">Not enough samples per bin for a reliable reliability diagram.</p>}
        </div>
      </div>

      <Section title="What the three-way policy does" className="mt-10" description={`Counts at the fixed policy threshold (${policy.toFixed(2)}), the way the app reports results to users.`}>
        <div className="panel overflow-x-auto"><table className="w-full min-w-[520px] text-left text-sm">
          <caption className="sr-only">Policy outcomes</caption>
          <thead className="border-b border-line bg-sunken/50 text-xs text-ink-3"><tr><th className="px-4 py-2 font-medium">Reported category</th><th className="px-4 py-2 font-medium">Images</th><th className="px-4 py-2 font-medium">Correct when decisive</th></tr></thead>
          <tbody>
            <tr className="border-b border-line"><td className="px-4 py-2.5"><Badge tone="ai">AI-generated suspected</Badge></td><td className="px-4 py-2.5 font-mono">{po.ai_suspected}</td><td className="px-4 py-2.5 font-mono">{po.ai_suspected ? pct(po.ai_suspected_correct / po.ai_suspected, 0) : "n/a"} ({po.ai_suspected_correct}/{po.ai_suspected})</td></tr>
            <tr className="border-b border-line"><td className="px-4 py-2.5"><Badge tone="real">Real or non-synthetic suspected</Badge></td><td className="px-4 py-2.5 font-mono">{po.real_suspected}</td><td className="px-4 py-2.5 font-mono">{po.real_suspected ? pct(po.real_suspected_correct / po.real_suspected, 0) : "n/a"} ({po.real_suspected_correct}/{po.real_suspected})</td></tr>
            <tr><td className="px-4 py-2.5"><Badge tone="unsure">Inconclusive</Badge></td><td className="px-4 py-2.5 font-mono">{po.inconclusive}</td><td className="px-4 py-2.5 text-ink-3">not scored</td></tr>
          </tbody>
        </table></div>
      </Section>

      <Section title="Dataset breakdown" description="Share of images flagged as AI at the policy threshold, by image type and size. Real images should be flagged rarely; AI images often.">
        <div className="panel overflow-x-auto"><table className="w-full min-w-[560px] text-left text-sm">
          <caption className="sr-only">Flag rate by group</caption>
          <thead className="border-b border-line bg-sunken/50 text-xs text-ink-3"><tr><th className="px-4 py-2 font-medium">Group</th><th className="px-4 py-2 font-medium">Images</th><th className="px-4 py-2 font-medium">Flagged as AI</th><th className="px-4 py-2 font-medium">95% CI</th><th className="px-4 py-2 font-medium">Mean AI score</th></tr></thead>
          <tbody>{v.groups.map((g) => (
            <tr key={g.name} className="border-b border-line last:border-0">
              <td className="px-4 py-2.5 font-medium">{g.name}</td><td className="px-4 py-2.5 font-mono">{g.n}</td>
              <td className={cx("px-4 py-2.5 font-mono", g.label === 0 && g.flagged_rate > 0.1 && "text-signal-ai")}>{pct(g.flagged_rate, 0)} ({g.flagged_ai}/{g.n})</td>
              <td className="px-4 py-2.5 font-mono text-ink-3">{g.flagged_ci ? `${pct(g.flagged_ci[0], 0)}–${pct(g.flagged_ci[1], 0)}` : "n/a"}</td>
              <td className="px-4 py-2.5 font-mono">{g.mean_ai_score.toFixed(2)}</td>
            </tr>))}</tbody>
        </table></div>
        <p className="mt-2 text-xs text-ink-3">Real and AI images differ in size here, so a group's rate mixes image type with image size.</p>
      </Section>

      <Section title="Model comparison" description={`Same ${ev.dataset.n_real + ev.dataset.n_ai} images, same policy threshold, original-file variant.`}>
        <div className="panel overflow-x-auto"><table className="w-full min-w-[760px] text-left text-sm">
          <caption className="sr-only">Detector comparison</caption>
          <thead className="border-b border-line bg-sunken/50 text-xs text-ink-3"><tr>{["Model", "ROC-AUC (95% CI)", "Accuracy @ policy", "False pos.", "False neg.", "License", "Commercial use"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr></thead>
          <tbody>{ev.models.map((m) => { const x = m.variants["raw"]; if (!x) return null; const p = x.at_policy_threshold; return (
            <tr key={m.id} className={cx("border-b border-line last:border-0", m.id === model.id && "bg-brand-soft/50")}>
              <td className="px-4 py-2.5"><button onClick={() => setModelId(m.id)} className="font-mono text-[13px] font-medium hover:underline">{m.id}</button>{m.is_production && <Badge className="ml-2" tone="real">Production</Badge>}</td>
              <td className="px-4 py-2.5 font-mono tnum">{x.auc.toFixed(3)} <span className="text-ink-3">({x.auc_ci[0].toFixed(2)}–{x.auc_ci[1].toFixed(2)})</span></td>
              <td className="px-4 py-2.5 font-mono tnum">{pct(p.accuracy, 1)}</td><td className="px-4 py-2.5 font-mono">{p.fp}</td><td className="px-4 py-2.5 font-mono">{p.fn}</td>
              <td className="px-4 py-2.5 text-ink-2">{m.license}</td>
              <td className="px-4 py-2.5">{m.commercial_use ? <Badge tone="real">Allowed</Badge> : <Badge tone="ai">Not allowed</Badge>}</td>
            </tr>); })}</tbody>
        </table></div>
        <p className="mt-2 text-xs text-ink-3">Overlapping confidence intervals mean the difference between two models is not established by this sample.</p>
      </Section>
    </>
  );
}

function RagLab({ ev }: { ev: RagEval }) {
  const [filter, setFilter] = useState<"all" | "problems">("problems");
  const rt = ev.retrieval;
  const rows = ev.rows.filter((r) => filter === "all" || (r.category === "answerable" ? (r.first_rank !== 1) : !r.abstained));
  const sweep = ev.abstention_sweep.map((s) => ({ threshold: s.threshold, answered: s.answerable_answered / s.answerable_total, refused: s.refusals_correct / s.refusals_total }));
  const tile = (label: string, x: { rate: number; ci: [number, number] | null; k: number; n: number }) => <Metric label={label} value={pct(x.rate, 1)} sub={`${x.k}/${x.n} · 95% CI ${x.ci ? `${pct(x.ci[0], 0)}–${pct(x.ci[1], 0)}` : "n/a"}`} />;
  const exportJson = () => download("verityai_rag_eval.json", JSON.stringify(ev, null, 2), "application/json");
  const exportCsv = () => download("verityai_rag_eval_rows.csv", toCsv(ev.rows.map((r) => ({
    id: r.id, category: r.category, question: r.question, expected: r.expected_sources.join("|"), retrieved: r.retrieved_sources.join("|"),
    abstained: r.abstained ? 1 : 0, first_rank: r.first_rank, best_distance: r.best_vector_distance, grounded: r.claims_grounded ? 1 : 0,
  }))), "text/csv");
  return (
    <>
      <div className="mb-6 grid gap-3 text-sm sm:grid-cols-2">
        <div className="rounded-lg border border-line bg-surface p-4"><p className="eyebrow mb-1.5">Question set</p>
          <p className="text-ink-2">{ev.question_set.answerable} answerable · {ev.question_set.unanswerable_in_domain} on-topic but not covered · {ev.question_set.off_topic} off-topic</p>
          <p className="mt-1 text-xs text-ink-3">{ev.config.documents} documents, {ev.config.chunks} passages, top-{ev.config.top_k}, {ev.config.embedding_model}.</p></div>
        <div className="rounded-lg border border-line bg-surface p-4"><p className="eyebrow mb-1.5">Provenance</p>
          <p className="text-ink-3">Generated {formatDate(ev.generated_at)} from commit <span className="font-mono">{ev.git_commit}</span>.</p>
          <div className="mt-2 flex gap-2"><Button size="sm" onClick={exportCsv}><Download className="h-3.5 w-3.5" aria-hidden />Rows CSV</Button><Button size="sm" onClick={exportJson}><Download className="h-3.5 w-3.5" aria-hidden />Full artifact</Button></div></div>
      </div>
      <Notice kind="warn" title="Labeling caveat">{ev.question_set.labeling}</Notice>

      <section aria-label="Retrieval metrics" className="panel mt-6 grid grid-cols-2 divide-line sm:grid-cols-3 sm:divide-x lg:grid-cols-5">
        {tile("Hit@1", rt.hit_at_1)}{tile("Hit@3", rt.hit_at_3)}{tile("Hit@5", rt.hit_at_5)}
        <Metric label="MRR" value={rt.mrr.toFixed(3)} sub={`95% CI ${rt.mrr_ci[0].toFixed(2)}–${rt.mrr_ci[1].toFixed(2)}`} />
        <Metric label="Answers grounded" value={`${ev.groundedness.answers_fully_grounded}/${ev.groundedness.answers}`} sub={`${ev.groundedness.claims_total} claims checked`} />
      </section>

      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        <AblationChart rows={ev.ablation} />
        <SweepChart rows={sweep} current={ev.config.distance_threshold} />
      </div>

      <Section title="Abstention behaviour" className="mt-10" description="The system should refuse questions the knowledge base cannot answer.">
        <div className="panel overflow-x-auto"><table className="w-full min-w-[480px] text-left text-sm">
          <caption className="sr-only">Abstention results</caption>
          <thead className="border-b border-line bg-sunken/50 text-xs text-ink-3"><tr><th className="px-4 py-2 font-medium">Question type</th><th className="px-4 py-2 font-medium">Correctly refused</th><th className="px-4 py-2 font-medium">95% CI</th></tr></thead>
          <tbody>
            {([["off_topic", "Off-topic (unrelated to AI media)"], ["unanswerable_in_domain", "On-topic, not covered by the documents"]] as const).map(([k, label]) => { const a = ev.abstention[k]; return a ? (
              <tr key={k} className="border-b border-line last:border-0"><td className="px-4 py-2.5">{label}</td><td className="px-4 py-2.5 font-mono">{a.abstained}/{a.n} ({pct(a.rate, 0)})</td><td className="px-4 py-2.5 font-mono text-ink-3">{a.ci ? `${pct(a.ci[0], 0)}–${pct(a.ci[1], 0)}` : "n/a"}</td></tr>) : null; })}
            <tr><td className="px-4 py-2.5">Answerable questions wrongly refused</td><td className="px-4 py-2.5 font-mono">{rt.wrongly_abstained.length}/{ev.question_set.answerable}</td><td className="px-4 py-2.5 text-ink-3">{rt.wrongly_abstained.join(", ") || "none"}</td></tr>
          </tbody>
        </table></div>
        <p className="mt-2 text-xs text-ink-3">On-topic questions the documents don't cover are the hardest case: the relevance gate keys on similarity, and nearby passages can look relevant.</p>
      </Section>

      <Section title="Question-level results" actions={<div className="inline-flex rounded-md border border-line bg-sunken p-0.5 text-xs">
        {(["problems", "all"] as const).map((k) => <button key={k} onClick={() => setFilter(k)} aria-pressed={filter === k} className={cx("rounded px-3 py-1 font-medium", filter === k ? "bg-surface shadow-panel" : "text-ink-3")}>{k === "all" ? "All questions" : "Problems only"}</button>)}</div>}>
        {rows.length === 0 ? <p className="text-sm text-ink-3">No problems in this view.</p> : (
          <div className="panel overflow-x-auto"><table className="w-full min-w-[720px] text-left text-sm">
            <caption className="sr-only">Per-question retrieval results</caption>
            <thead className="border-b border-line bg-sunken/50 text-xs text-ink-3"><tr>{["ID", "Question", "Type", "Expected", "Top source", "Outcome"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr></thead>
            <tbody>{rows.map((r) => {
              const ok = r.category === "answerable" ? r.first_rank === 1 : r.abstained;
              return (
                <tr key={r.id} className="border-b border-line align-top last:border-0">
                  <td className="px-4 py-2.5 font-mono text-xs">{r.id}</td><td className="px-4 py-2.5">{r.question}</td>
                  <td className="px-4 py-2.5 text-xs text-ink-3">{r.category.replace(/_/g, " ")}</td>
                  <td className="px-4 py-2.5 font-mono text-xs text-ink-3">{r.expected_sources.join(", ") || "refuse"}</td>
                  <td className="px-4 py-2.5 font-mono text-xs">{r.abstained ? "(refused)" : r.retrieved_sources[0]}</td>
                  <td className="px-4 py-2.5"><Badge tone={ok ? "real" : "ai"}>{r.category === "answerable" ? (r.first_rank === null ? "Missed" : r.first_rank === 1 ? "Rank 1" : `Rank ${r.first_rank}`) : r.abstained ? "Refused" : "Answered anyway"}</Badge></td>
                </tr>);
            })}</tbody>
          </table></div>
        )}
      </Section>
    </>
  );
}

export function Evaluation() {
  useDocumentTitle("Evaluation lab");
  const { data, error, loading, reload } = useAsync((s) => api.evaluation(s), []);
  const [tab, setTab] = useState<"detector" | "retrieval">("detector");
  return (
    <>
      <PageHeader eyebrow="Quality" title="Evaluation lab"
        description="Every figure here is computed from stored evaluation artifacts produced by the project's own scripts. Anything that has not been measured says “Not evaluated”." />
      {error && <Notice kind="error" title="Could not load evaluation results" action={<Button size="sm" onClick={reload}>Retry</Button>}>{error.message}</Notice>}
      {loading && !data && <div className="space-y-4" aria-busy="true"><Skeleton className="h-24" /><Skeleton className="h-72" /></div>}
      {data && (
        <>
          <Tabs label="Evaluation area" value={tab} onChange={setTab} items={[{ id: "detector", label: "Detector" }, { id: "retrieval", label: "Research retrieval" }]} />
          {tab === "detector" ? (data.detector ? <DetectorLab ev={data.detector} /> : NOT_EVALUATED) : (data.rag ? <RagLab ev={data.rag} /> : NOT_EVALUATED)}
        </>
      )}
    </>
  );
}
