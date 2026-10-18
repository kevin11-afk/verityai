import { UploadCloud } from "lucide-react";
import { useRef, useState, type ReactNode } from "react";
import { cx } from "@/components/ui";

export function Dropzone({ accept, onFile, disabled, hint, children }: { accept: string[]; onFile: (f: File) => void; disabled?: boolean; hint: ReactNode; children?: ReactNode }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const pick = (files: FileList | null) => { const f = files?.[0]; if (f) onFile(f); };
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); if (!disabled) pick(e.dataTransfer.files); }}
      className={cx("rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors",
        over ? "border-brand bg-brand-soft" : "border-line-strong bg-surface hover:border-ink-3", disabled && "opacity-60")}
    >
      <UploadCloud className="mx-auto h-8 w-8 text-ink-3" aria-hidden />
      <p className="mt-3 text-sm font-medium">
        Drag a file here, or{" "}
        <button type="button" disabled={disabled} onClick={() => input.current?.click()} className="link font-semibold">browse</button>
      </p>
      <p className="mt-1.5 text-xs text-ink-3">{hint}</p>
      <input ref={input} type="file" className="sr-only" tabIndex={-1} aria-label="Choose a file" accept={accept.join(",")} disabled={disabled}
        onChange={(e) => { pick(e.target.files); e.target.value = ""; }} />
      {children}
    </div>
  );
}
