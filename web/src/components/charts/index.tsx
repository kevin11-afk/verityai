import {
  Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, LineChart, ReferenceDot, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from "recharts";
import type { CalibrationBin, FrameInfo } from "@/api/types";
import { cx } from "@/components/ui";
import type { Confusion, TradeoffPoint } from "@/lib/metrics";

const C = { grid: "#E4E0D7", axis: "#5A6781", brand: "#3F3CBB", teal: "#0F766E", red: "#A8243A", amber: "#8A5A00", ink: "#0B1B33" };
const axis = { stroke: C.axis, fontSize: 12, tickLine: false } as const;
const fmt2 = (v: number | string) => (typeof v === "number" ? v.toFixed(2) : v);

function Tip({ active, payload, label, labelFmt }: { active?: boolean; payload?: { name?: string; value?: number | string; color?: string; payload?: Record<string, unknown> }[]; label?: number | string; labelFmt?: (l: number | string) => string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-line bg-surface px-3 py-2 text-xs shadow-pop">
      {label !== undefined && <p className="mb-1 font-semibold">{labelFmt ? labelFmt(label) : String(label)}</p>}
      {payload.map((p, i) => (
        <p key={i} className="flex items-center gap-2 text-ink-2">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} aria-hidden />{p.name}: <span className="font-mono">{typeof p.value === "number" ? p.value.toFixed(3) : p.value}</span>
        </p>
      ))}
    </div>
  );
}

export function ChartFrame({ title, caption, height = 280, children, label }: { title: string; caption?: string; height?: number; children: React.ReactNode; label: string }) {
  return (
    <figure className="panel p-4">
      <figcaption className="mb-3"><p className="text-sm font-semibold">{title}</p>{caption && <p className="text-xs text-ink-3">{caption}</p>}</figcaption>
      <div style={{ height }} role="img" aria-label={label}><ResponsiveContainer width="100%" height="100%">{children as React.ReactElement}</ResponsiveContainer></div>
    </figure>
  );
}

export function RocChart({ points, marker }: { points: [number, number, number | null][]; marker: { fpr: number; tpr: number } }) {
  const data = points.map(([fpr, tpr, thr]) => ({ fpr, tpr, thr }));
  return (
    <ChartFrame title="ROC curve" caption="True-positive rate against false-positive rate across every threshold." label="ROC curve; dot marks the selected threshold">
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
        <XAxis type="number" dataKey="fpr" domain={[0, 1]} tickFormatter={fmt2} {...axis} label={{ value: "False-positive rate", position: "insideBottom", offset: -12, fill: C.axis, fontSize: 12 }} />
        <YAxis type="number" domain={[0, 1]} tickFormatter={fmt2} {...axis} label={{ value: "True-positive rate", angle: -90, position: "insideLeft", fill: C.axis, fontSize: 12 }} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={C.axis} strokeDasharray="4 4" />
        <Tooltip content={<Tip labelFmt={(l) => `FPR ${Number(l).toFixed(3)}`} />} />
        <Line name="TPR" type="linear" dataKey="tpr" stroke={C.brand} strokeWidth={2.2} dot={false} isAnimationActive={false} />
        <ReferenceDot x={marker.fpr} y={marker.tpr} r={6} fill={C.red} stroke="#fff" strokeWidth={2} />
      </ComposedChart>
    </ChartFrame>
  );
}

export function PrChart({ points, marker, base }: { points: [number, number, number | null][]; marker: { recall: number; precision: number } | null; base: number }) {
  const data = points.map(([recall, precision]) => ({ recall, precision }));
  return (
    <ChartFrame title="Precision–recall curve" caption={`Dashed line: precision of a random flagger (${(base * 100).toFixed(0)}% of samples are AI).`} label="Precision-recall curve; dot marks the selected threshold">
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
        <XAxis type="number" dataKey="recall" domain={[0, 1]} tickFormatter={fmt2} {...axis} label={{ value: "Recall", position: "insideBottom", offset: -12, fill: C.axis, fontSize: 12 }} />
        <YAxis type="number" domain={[0, 1]} tickFormatter={fmt2} {...axis} label={{ value: "Precision", angle: -90, position: "insideLeft", fill: C.axis, fontSize: 12 }} />
        <ReferenceLine y={base} stroke={C.axis} strokeDasharray="4 4" />
        <Tooltip content={<Tip labelFmt={(l) => `Recall ${Number(l).toFixed(3)}`} />} />
        <Line name="Precision" type="linear" dataKey="precision" stroke={C.teal} strokeWidth={2.2} dot={false} isAnimationActive={false} />
        {marker && <ReferenceDot x={marker.recall} y={marker.precision} r={6} fill={C.red} stroke="#fff" strokeWidth={2} />}
      </ComposedChart>
    </ChartFrame>
  );
}

export function TradeoffChart({ curve, threshold, policy }: { curve: TradeoffPoint[]; threshold: number; policy: number }) {
  return (
    <ChartFrame title="Error trade-off by threshold" caption="Raising the threshold flags fewer real images (lower FPR) but misses more AI images (higher FNR)." label="False-positive and false-negative rates by threshold">
      <LineChart data={curve} margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
        <XAxis type="number" dataKey="threshold" domain={[0, 1]} tickFormatter={fmt2} {...axis} label={{ value: "AI-score threshold", position: "insideBottom", offset: -12, fill: C.axis, fontSize: 12 }} />
        <YAxis domain={[0, 1]} tickFormatter={fmt2} {...axis} />
        <Tooltip content={<Tip labelFmt={(l) => `Threshold ${Number(l).toFixed(2)}`} />} />
        <Legend verticalAlign="top" height={24} iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
        <ReferenceLine x={policy} stroke={C.axis} strokeDasharray="4 4" label={{ value: "policy", position: "top", fontSize: 11, fill: C.axis }} />
        <ReferenceLine x={threshold} stroke={C.red} strokeWidth={2} />
        <Line name="False-positive rate (real flagged as AI)" type="monotone" dataKey="fpr" stroke={C.red} strokeWidth={2} dot={false} isAnimationActive={false} />
        <Line name="False-negative rate (AI missed)" type="monotone" dataKey="fnr" stroke={C.brand} strokeWidth={2} dot={false} isAnimationActive={false} />
      </LineChart>
    </ChartFrame>
  );
}

export function CalibrationChart({ bins, valid }: { bins: CalibrationBin[]; valid: boolean }) {
  const data = bins.map((b) => ({ x: b.mean_score, y: b.fraction_ai, n: b.n }));
  return (
    <ChartFrame title="Calibration (reliability)" caption={valid ? "If scores were calibrated probabilities, points would sit on the diagonal." : "Too few samples per bin for a reliable reading; shown for transparency only."} label="Calibration reliability diagram">
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
        <XAxis type="number" dataKey="x" domain={[0, 1]} tickFormatter={fmt2} {...axis} label={{ value: "Mean predicted AI score", position: "insideBottom", offset: -12, fill: C.axis, fontSize: 12 }} />
        <YAxis type="number" domain={[0, 1]} tickFormatter={fmt2} {...axis} label={{ value: "Observed share that is AI", angle: -90, position: "insideLeft", fill: C.axis, fontSize: 12 }} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={C.axis} strokeDasharray="4 4" />
        <Tooltip content={({ active, payload }) => active && payload?.[0] ? (
          <div className="rounded-md border border-line bg-surface px-3 py-2 text-xs shadow-pop">
            <p className="font-semibold">{(payload[0].payload as { n: number }).n} samples in bin</p>
            <p className="font-mono text-ink-2">score {Number(payload[0].payload.x).toFixed(3)} → observed {Number(payload[0].payload.y).toFixed(3)}</p>
          </div>) : null} />
        <Line type="linear" dataKey="y" stroke={valid ? C.brand : C.axis} strokeWidth={1.5} dot={{ r: 4, fill: valid ? C.brand : C.axis }} isAnimationActive={false} />
        <Scatter dataKey="y" fill="transparent" />
      </ComposedChart>
    </ChartFrame>
  );
}

export function ConfusionMatrix({ c }: { c: Confusion }) {
  const total = c.tp + c.fp + c.tn + c.fn || 1;
  const cell = (label: string, v: number, good: boolean) => (
    <div className={cx("rounded-md border p-4 text-center", good ? "border-teal/25 bg-teal-soft" : "border-signal-ai/20 bg-signal-ai-soft")}>
      <p className="font-mono text-2xl font-semibold tnum">{v}</p>
      <p className="mt-0.5 text-xs font-medium">{label}</p>
      <p className="text-[11px] text-ink-3">{((v / total) * 100).toFixed(1)}% of samples</p>
    </div>
  );
  return (
    <figure className="panel p-4">
      <figcaption className="mb-3"><p className="text-sm font-semibold">Confusion matrix</p><p className="text-xs text-ink-3">Counts at the selected threshold.</p></figcaption>
      <div className="grid grid-cols-[auto_1fr_1fr] items-stretch gap-2 text-xs">
        <div />
        <p className="pb-1 text-center font-semibold text-ink-2">Predicted real</p>
        <p className="pb-1 text-center font-semibold text-ink-2">Predicted AI</p>
        <p className="flex items-center font-semibold text-ink-2 [writing-mode:vertical-rl] rotate-180">Actually real</p>
        {cell("True negative", c.tn, true)}{cell("False positive", c.fp, false)}
        <p className="flex items-center font-semibold text-ink-2 [writing-mode:vertical-rl] rotate-180">Actually AI</p>
        {cell("False negative", c.fn, false)}{cell("True positive", c.tp, true)}
      </div>
    </figure>
  );
}

export function FrameChart({ frames, threshold }: { frames: FrameInfo[]; threshold: number }) {
  const data = frames.map((f) => ({ t: f.timestamp_seconds, ai: f.scores.ai ?? 0 }));
  return (
    <ChartFrame title="AI score by sampled frame" caption="Each bar is the still-image detector run on one frame. Variation between frames is not evidence of manipulation." height={220} label="AI score of each sampled video frame">
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 22, left: 0 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="t" tickFormatter={(v: number) => `${v.toFixed(1)}s`} {...axis} label={{ value: "Time in video", position: "insideBottom", offset: -12, fill: C.axis, fontSize: 12 }} />
        <YAxis domain={[0, 1]} tickFormatter={fmt2} {...axis} />
        <ReferenceLine y={threshold} stroke={C.ink} strokeDasharray="4 4" label={{ value: "policy", position: "right", fontSize: 11, fill: C.axis }} />
        <Tooltip content={<Tip labelFmt={(l) => `t = ${Number(l).toFixed(2)} s`} />} />
        <Bar name="AI score" dataKey="ai" fill={C.brand} radius={[3, 3, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ChartFrame>
  );
}

export function AblationChart({ rows }: { rows: { mode: string; hit_at_1: number; hit_at_5: number; mrr: number }[] }) {
  return (
    <ChartFrame title="Retrieval mode comparison" caption="Same questions, same index; only the ranking method differs." label="Hit@1, Hit@5 and MRR for BM25, vector and hybrid retrieval">
      <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="mode" {...axis} />
        <YAxis domain={[0, 1]} tickFormatter={fmt2} {...axis} />
        <Tooltip content={<Tip />} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar name="Hit@1" dataKey="hit_at_1" fill={C.brand} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        <Bar name="Hit@5" dataKey="hit_at_5" fill={C.teal} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        <Bar name="MRR" dataKey="mrr" fill="#9CA3C4" radius={[3, 3, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ChartFrame>
  );
}

export function SweepChart({ rows, current }: { rows: { threshold: number; answered: number; refused: number }[]; current: number }) {
  return (
    <ChartFrame title="Relevance-gate trade-off" caption="A stricter gate refuses more off-topic questions but also refuses more answerable ones (vector gate only)." label="Share of answerable questions answered and unanswerable questions refused by distance threshold">
      <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
        <CartesianGrid stroke={C.grid} strokeDasharray="3 3" />
        <XAxis type="number" dataKey="threshold" domain={[0.4, 1]} tickFormatter={fmt2} {...axis} label={{ value: "Distance threshold", position: "insideBottom", offset: -12, fill: C.axis, fontSize: 12 }} />
        <YAxis domain={[0, 1]} tickFormatter={fmt2} {...axis} />
        <Tooltip content={<Tip labelFmt={(l) => `Threshold ${Number(l).toFixed(2)}`} />} />
        <Legend verticalAlign="top" height={24} iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
        <ReferenceLine x={current} stroke={C.red} strokeWidth={2} label={{ value: "current", position: "top", fontSize: 11, fill: C.red }} />
        <Line name="Answerable questions answered" type="monotone" dataKey="answered" stroke={C.teal} strokeWidth={2} dot={false} isAnimationActive={false} />
        <Line name="Off-topic / uncovered questions refused" type="monotone" dataKey="refused" stroke={C.brand} strokeWidth={2} dot={false} isAnimationActive={false} />
      </LineChart>
    </ChartFrame>
  );
}
