import { Cpu } from "lucide-react";
import type { DetectorInfo, ModelResult } from "@/api/types";
import { Badge, ScoreBar } from "@/components/ui";
import { categoryTone, classLabel, type Tone } from "@/lib/format";

const toneOfClass = (cls: string): Tone => (cls.toLowerCase() === "ai" ? "ai" : cls.toLowerCase() === "real" ? "real" : "neutral");

/** The detector's own output. Deliberately labelled and styled apart from retrieved research. */
export function PredictionPanel({ result, detector, experimental }: { result: ModelResult; detector: DetectorInfo; experimental?: boolean }) {
  const tone = categoryTone(result.category);
  const entries = Object.entries(result.scores).sort((a, b) => b[1] - a[1]);
  return (
    <section aria-labelledby="pred-h" className="panel overflow-hidden">
      <div className="flex items-center gap-2 border-b border-line bg-sunken/60 px-5 py-2.5">
        <Cpu className="h-4 w-4 text-ink-3" aria-hidden />
        <h2 id="pred-h" className="eyebrow !text-ink-2">Model prediction · detector output</h2>
      </div>
      <div className="p-5">
        <div className="flex flex-wrap items-center gap-3">
          <Badge tone={tone} className="px-3 py-1 text-[13px]">{result.category}</Badge>
          {experimental && <Badge>Experimental aggregate</Badge>}
        </div>
        <p className="mt-2 text-xs text-ink-3">
          Category is this project's fixed decision policy applied to the raw scores; it is not a probability that the file is authentic or fake.
        </p>
        <div className="mt-5 space-y-4">
          {entries.map(([label, v]) => (
            <ScoreBar key={label} label={classLabel(label)} value={v} tone={toneOfClass(label)}
              threshold={label.toLowerCase() === "ai" ? detector.thresholds.ai : label.toLowerCase() === "real" ? detector.thresholds.real : undefined} />
          ))}
        </div>
        <p className="mt-3 text-xs text-ink-3">The vertical mark shows the policy threshold ({detector.thresholds.ai.toFixed(2)}). {result.score_semantics}</p>
        <details className="mt-4 rounded-md border border-line bg-canvas px-3.5 py-2.5 text-sm">
          <summary className="cursor-pointer font-medium text-ink-2">Decision policy and detector</summary>
          <p className="mt-2 text-ink-2">{result.decision_policy}</p>
          <p className="mt-2 text-ink-3">{detector.version_label} · inference {Math.round(result.latency_ms)} ms</p>
        </details>
      </div>
    </section>
  );
}
