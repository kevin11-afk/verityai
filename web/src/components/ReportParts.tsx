import { Download, FileJson, FileText } from "lucide-react";
import { api } from "@/api/client";
import type { Report } from "@/api/types";
import { DiagnosticsPanel, NoEvidence, PassageList } from "@/components/Evidence";
import { Button, ButtonAnchor } from "@/components/ui";

/** Retrieved research for a report: always visually and verbally separate from the model output. */
export function EvidenceSection({ report }: { report: Report }) {
  const { passages, diagnostics } = report.evidence;
  return (
    <section aria-labelledby="evi-h">
      <div className="mb-3">
        <h2 id="evi-h" className="eyebrow !text-ink-2">Retrieved research · from the knowledge base, not from the detector</h2>
        <p className="mt-1 text-sm text-ink-3">
          These passages were retrieved by search to give context about detector limitations. They do not describe this specific file.
        </p>
      </div>
      {passages.length > 0
        ? <PassageList passages={passages} />
        : <NoEvidence message={report.explanation.no_evidence_notice ?? diagnostics.abstain_reason ?? "No passage was relevant enough to cite."} />}
      <details className="mt-3"><summary className="cursor-pointer text-sm font-medium text-ink-2">Retrieval diagnostics</summary><div className="mt-2"><DiagnosticsPanel d={diagnostics} /></div></details>
    </section>
  );
}

export function DownloadButtons({ id }: { id: string }) {
  return (
    <>
      <ButtonAnchor href={api.reportDownloadUrl(id, "json")} download size="sm"><FileJson className="h-4 w-4" aria-hidden />JSON</ButtonAnchor>
      <ButtonAnchor href={api.reportDownloadUrl(id, "md")} download size="sm"><FileText className="h-4 w-4" aria-hidden />Markdown</ButtonAnchor>
      <Button size="sm" variant="ghost" onClick={() => window.print()}><Download className="h-4 w-4" aria-hidden />Print / PDF</Button>
    </>
  );
}
