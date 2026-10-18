import { describe, expect, it } from "vitest";
import { balancedThreshold, confusionAt, rates, toCsv, tradeoffCurve } from "./metrics";
import { categoryTone, formatBytes, safeUrl } from "./format";

const labels = [1, 1, 1, 0, 0, 0];
const scores = [0.9, 0.7, 0.4, 0.6, 0.2, 0.1];

describe("confusionAt", () => {
  it("counts each outcome at a threshold", () => {
    expect(confusionAt(labels, scores, 0.65)).toEqual({ tp: 2, fp: 0, tn: 3, fn: 1 });
    expect(confusionAt(labels, scores, 0.5)).toEqual({ tp: 2, fp: 1, tn: 2, fn: 1 });
  });
  it("treats a score equal to the threshold as flagged (matches the backend policy)", () => {
    expect(confusionAt([1], [0.65], 0.65).tp).toBe(1);
  });
});

describe("rates", () => {
  it("computes standard rates and returns null instead of dividing by zero", () => {
    const r = rates({ tp: 2, fp: 1, tn: 2, fn: 1 });
    expect(r.recall).toBeCloseTo(2 / 3);
    expect(r.fpr).toBeCloseTo(1 / 3);
    expect(r.accuracy).toBeCloseTo(4 / 6);
    expect(rates({ tp: 0, fp: 0, tn: 0, fn: 0 }).accuracy).toBeNull();
    expect(rates({ tp: 0, fp: 0, tn: 3, fn: 0 }).precision).toBeNull();
  });
});

describe("tradeoffCurve", () => {
  it("FNR rises and FPR falls as the threshold increases", () => {
    const c = tradeoffCurve(labels, scores, 11);
    expect(c[0]!.fpr).toBe(1);
    expect(c[0]!.fnr).toBe(0);
    expect(c[10]!.fpr).toBe(0);
    expect(c[10]!.fnr).toBe(1);
  });
  it("balancedThreshold minimises FPR + FNR", () => {
    const t = balancedThreshold(labels, scores);
    const r = rates(confusionAt(labels, scores, t));
    expect((r.fpr ?? 1) + (r.fnr ?? 1)).toBeLessThanOrEqual(1 / 3 + 1e-9);
  });
});

describe("helpers", () => {
  it("escapes CSV", () => {
    expect(toCsv([{ a: "x,y", b: 'q"z', c: null }])).toBe('a,b,c\n"x,y","q""z",');
  });
  it("only allows http(s) links", () => {
    expect(safeUrl("https://example.org/a")).toBe("https://example.org/a");
    expect(safeUrl("javascript:alert(1)")).toBeNull();
    expect(safeUrl("data:text/html,<script>")).toBeNull();
    expect(safeUrl(null)).toBeNull();
  });
  it("maps categories to tones and formats bytes", () => {
    expect(categoryTone("AI-generated suspected")).toBe("ai");
    expect(categoryTone("Real or non-synthetic suspected")).toBe("real");
    expect(categoryTone("Inconclusive")).toBe("unsure");
    expect(formatBytes(1536)).toBe("1.5 KB");
  });
});
