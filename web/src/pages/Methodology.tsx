import type { ReactNode } from "react";
import { Notice, PageHeader } from "@/components/ui";
import { useDocumentTitle } from "@/lib/hooks";
import { useHealth } from "@/lib/health";

function Block({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="scroll-mt-24 border-t border-line py-8 first:border-0 first:pt-0">
      <h2 id={`${id}-h`} className="mb-3 text-lg font-semibold">{title}</h2>
      <div className="space-y-3 text-[15px] leading-relaxed text-ink-2">{children}</div>
    </section>
  );
}
const List = ({ items }: { items: ReactNode[] }) => <ul className="space-y-2">{items.map((x, i) => <li key={i} className="flex gap-3"><span className="mt-2.5 h-1 w-1 shrink-0 rounded-full bg-ink-3" aria-hidden />{x}</li>)}</ul>;

const TOC = [["detector", "How the detector works"], ["scores", "What the scores mean"], ["rag", "How retrieval and answers work"], ["evidence", "How evidence is selected"], ["limits", "Known limitations"], ["privacy", "Privacy and data handling"], ["cannot", "What it cannot establish"]];

export function Methodology() {
  useDocumentTitle("Methodology & privacy");
  const { health } = useHealth();
  const t = health?.thresholds;
  const lim = health?.limits;
  return (
    <>
      <PageHeader eyebrow="Reference" title="Methodology & privacy" description="How VerityAI reaches a result, what that result means, and what happens to the files you upload." />
      <div className="grid gap-10 lg:grid-cols-[200px_minmax(0,1fr)]">
        <nav aria-label="On this page" className="no-print hidden lg:block"><ul className="sticky top-8 space-y-1.5 text-sm">
          {TOC.map(([id, label]) => <li key={id}><a href={`#${id}`} className="text-ink-3 hover:text-ink">{label}</a></li>)}
        </ul></nav>
        <div className="max-w-3xl">
          <Block id="detector" title="How the detector works">
            <p>Each image is passed through a pretrained image classifier{health ? <> (<span className="font-mono text-sm">{health.detector.name}</span>, {health.detector_version_label})</> : null}, a Swin-Tiny vision transformer fine-tuned to separate real photographs from AI-generated images. It outputs two numbers, one per class. Nothing in the pipeline is random, simulated or hard-coded: if the model cannot run, the analysis fails with an error rather than inventing a result.</p>
            <p>Video is handled by sampling up to {lim?.max_video_frames ?? 8} evenly spaced frames, running the same image classifier on each, and averaging the scores. That is a frame-level heuristic. It does not look at motion or consistency across frames, so it is not a temporal deepfake detector, and the app labels it experimental.</p>
          </Block>
          <Block id="scores" title="What the scores mean">
            <p>The two scores are the model's raw softmax outputs and sum to 1. They show how strongly the image resembles the patterns the model learned for each class. They are <strong>not calibrated probabilities</strong>: a score of 0.90 does not mean a 90% chance that the image is AI-generated.</p>
            <p>The reported category applies a fixed policy to those scores{t ? <>: <strong>AI-generated suspected</strong> if the AI score is at least {t.ai.toFixed(2)}; <strong>Real or non-synthetic suspected</strong> if the real score is at least {t.real.toFixed(2)}; otherwise <strong>Inconclusive</strong></> : null}. These thresholds were fixed in advance, not tuned on the evaluation data. The Evaluation lab shows what they do on a measured sample, including error rates and confidence intervals.</p>
          </Block>
          <Block id="rag" title="How retrieval and answers work">
            <p>The knowledge base is a set of documents split into passages. A question is matched in two ways at once: keyword search (BM25) finds passages sharing the question's terms, and semantic search (sentence embeddings in a vector index) finds passages with similar meaning. The two rankings are merged with Reciprocal Rank Fusion.</p>
            <p>There is no language model in the loop. An answer is built by choosing the sentences from the retrieved passages that best match the question and quoting them word-for-word with a numbered citation. The system cannot state something that is not in a cited passage. Its weakness is the opposite one: it can only repeat what the documents say, and sentence selection is simple.</p>
          </Block>
          <Block id="evidence" title="How evidence is selected">
            <List items={[
              <>A relevance gate compares the closest passage to the question. If it is too far away, and the question's key terms do not appear together in any passage, the system refuses to answer and says there is not enough evidence.</>,
              <>Passages that pass are ranked by fused keyword and semantic rank. The top passages are shown with their source file, page, and ranking details (open “Retrieval diagnostics”).</>,
              <>In a report, retrieved passages give general context about detector limitations. They are not findings about your file, and the interface keeps them visually separate from the model's prediction.</>,
            ]} />
          </Block>
          <Block id="limits" title="Known limitations">
            <List items={[
              "The detector is a single pretrained classifier with moderate measured accuracy. On the evaluation sample it flagged a meaningful share of real photos as AI and missed some AI images (see the Evaluation lab for exact figures and intervals).",
              "Its training data predates the newest image generators. Images from recent systems may be scored as real; a false “Real suspected” result has been observed in manual testing.",
              "Compression, resizing, screenshots and editing can move scores in either direction. Results on partly edited images are not well defined.",
              "Video analysis is frame-level only and experimental.",
              "The knowledge base is small and hand-curated. Retrieval was evaluated on a limited question set written by the project authors, so real-world retrieval quality may differ.",
              "Evaluation used 200 images from one dataset; confidence intervals are wide.",
            ]} />
          </Block>
          <Block id="privacy" title="Privacy and data handling">
            <List items={[
              <>Uploaded images and videos are sent to the VerityAI server for analysis, held in a temporary file only while the model runs, and deleted immediately afterwards. The media itself is never saved.</>,
              <>What <em>is</em> saved is the report: file name, type, size, dimensions or duration, a SHA-256 fingerprint of the file, the scores and the retrieved passages. {health?.persistence_mode === "ephemeral" ? "On this deployment reports live on temporary disk and can disappear when the server restarts." : "On this deployment reports are kept on persistent storage."} You can delete any report from its page.</>,
              <>Analysis runs entirely on the VerityAI server. No file, file name or result is sent to any third-party service or AI provider. The only outbound requests are one-time downloads of the open-source model files from Hugging Face when the server first starts.</>,
              <>Documents you add to the knowledge base are stored on the server and indexed there. Anyone using this deployment can see and search them, so do not upload confidential material to a shared instance.</>,
              <>Your browser stores nothing except, if you change it, the API address (in local storage).</>,
              <>Retrieved text and uploaded documents are treated as untrusted: displayed as plain text only, never executed, and external links are limited to http(s).</>,
            ]} />
            <p>Limits: images up to {lim?.max_image_mb ?? 15} MB, video up to {lim?.max_video_mb ?? 60} MB and {lim?.max_video_seconds ?? 30} s, documents up to {lim?.max_kb_upload_mb ?? 20} MB. File types are checked by content, not just by name.</p>
          </Block>
          <Block id="cannot" title="What VerityAI cannot establish">
            <Notice kind="warn" title="A screening signal, not proof">A result here cannot prove that a file is authentic, AI-generated or edited, cannot establish who made it, and must not be used as legal or forensic evidence.</Notice>
            <List items={[
              "It cannot verify provenance. Cryptographic content credentials (C2PA) are a stronger signal when present, and VerityAI does not read them.",
              "A “Real or non-synthetic suspected” result is not a certificate of authenticity.",
              "An “AI-generated suspected” result is not an accusation; real photographs are sometimes flagged.",
              "Always corroborate with source checking, metadata review and, where it matters, a qualified human examiner.",
            ]} />
          </Block>
        </div>
      </div>
    </>
  );
}
