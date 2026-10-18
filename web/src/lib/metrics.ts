// Pure metric functions over real per-sample scores (label 1 = AI). Used by the
// Evaluation Lab so threshold changes recompute from measured data, not stored examples.

export interface Confusion { tp: number; fp: number; tn: number; fn: number }

export function confusionAt(labels: number[], scores: number[], threshold: number): Confusion {
  const c: Confusion = { tp: 0, fp: 0, tn: 0, fn: 0 };
  for (let i = 0; i < labels.length; i++) {
    const predictedAi = (scores[i] ?? 0) >= threshold;
    const isAi = labels[i] === 1;
    if (predictedAi && isAi) c.tp++;
    else if (predictedAi && !isAi) c.fp++;
    else if (!predictedAi && !isAi) c.tn++;
    else c.fn++;
  }
  return c;
}

const div = (a: number, b: number): number | null => (b === 0 ? null : a / b);

export function rates(c: Confusion) {
  const n = c.tp + c.fp + c.tn + c.fn;
  return {
    n,
    accuracy: div(c.tp + c.tn, n),
    precision: div(c.tp, c.tp + c.fp),
    recall: div(c.tp, c.tp + c.fn),          // true-positive rate (AI caught)
    specificity: div(c.tn, c.tn + c.fp),     // real images correctly passed
    fpr: div(c.fp, c.fp + c.tn),             // real flagged as AI
    fnr: div(c.fn, c.fn + c.tp),             // AI missed
    f1: (() => {
      const p = div(c.tp, c.tp + c.fp), r = div(c.tp, c.tp + c.fn);
      return p === null || r === null || p + r === 0 ? null : (2 * p * r) / (p + r);
    })(),
  };
}

export interface TradeoffPoint { threshold: number; fpr: number; fnr: number }

/** FPR and FNR across thresholds: the cost of moving the decision boundary. */
export function tradeoffCurve(labels: number[], scores: number[], steps = 101): TradeoffPoint[] {
  const out: TradeoffPoint[] = [];
  for (let i = 0; i < steps; i++) {
    const t = i / (steps - 1);
    const r = rates(confusionAt(labels, scores, t));
    out.push({ threshold: t, fpr: r.fpr ?? 0, fnr: r.fnr ?? 0 });
  }
  return out;
}

/** Threshold with minimum FPR + FNR (Youden-style balance) on the given samples. */
export function balancedThreshold(labels: number[], scores: number[]): number {
  let best = { t: 0.5, cost: Infinity };
  for (const p of tradeoffCurve(labels, scores, 201)) {
    const cost = p.fpr + p.fnr;
    if (cost < best.cost) best = { t: p.threshold, cost };
  }
  return best.t;
}

export function toCsv(rows: Record<string, string | number | null>[]): string {
  if (rows.length === 0) return "";
  const first = rows[0] as Record<string, string | number | null>;
  const cols = Object.keys(first);
  const esc = (v: string | number | null) => {
    const s = v === null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c] ?? null)).join(","))].join("\n");
}
