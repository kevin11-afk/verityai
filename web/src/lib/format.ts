export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export const pct = (v: number, digits = 1): string => `${(v * 100).toFixed(digits)}%`;

export function pctCi(v: number, ci: [number, number] | null | undefined, digits = 0): string {
  return ci ? `${pct(v, digits)} (${pct(ci[0], digits)}–${pct(ci[1], digits)})` : pct(v, digits);
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function shortId(id: string): string { return id.slice(0, 8); }

export type Tone = "ai" | "real" | "unsure" | "neutral";
export function categoryTone(category: string): Tone {
  if (category.startsWith("AI")) return "ai";
  if (category.startsWith("Real")) return "real";
  if (category === "Inconclusive") return "unsure";
  return "neutral";
}

export function citationLabel(p: { title: string; page_number?: number | null; source_file: string }): string {
  return `${p.title}${p.page_number ? `, p.${p.page_number}` : ""} (${p.source_file})`;
}

/** Only http(s) links are ever rendered as links (blocks javascript: and data: URLs). */
export function safeUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  try {
    const u = new URL(url);
    return u.protocol === "https:" || u.protocol === "http:" ? u.toString() : null;
  } catch { return null; }
}

export function download(filename: string, content: string, mime: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Human label for a classifier class name ("ai" -> "AI", "real" -> "Real"). */
export function classLabel(c: string): string {
  return c.toLowerCase() === "ai" ? "AI" : c.charAt(0).toUpperCase() + c.slice(1);
}
