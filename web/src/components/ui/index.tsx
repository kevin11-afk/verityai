import { AlertCircle, AlertTriangle, CheckCircle2, Info, Loader2, X } from "lucide-react";
import { useEffect, useRef, type AnchorHTMLAttributes, type ButtonHTMLAttributes, type ReactNode } from "react";
import { Link, type LinkProps } from "react-router-dom";
import type { Tone } from "@/lib/format";

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

/* ---- Button ---------------------------------------------------------------- */
type Variant = "primary" | "secondary" | "ghost" | "danger";
const variants: Record<Variant, string> = {
  primary: "bg-ink text-white hover:bg-ink-2 disabled:bg-ink/40",
  secondary: "bg-surface text-ink border border-line-strong hover:bg-sunken disabled:text-ink/40",
  ghost: "text-ink-2 hover:bg-sunken disabled:text-ink/40",
  danger: "bg-signal-ai text-white hover:bg-[#8A1D30] disabled:bg-signal-ai/40",
};
const buttonClass = (variant: Variant, size: "sm" | "md", extra?: string) => cx(
  "inline-flex items-center justify-center gap-2 rounded-md font-medium transition-colors",
  size === "sm" ? "h-8 px-3 text-[13px]" : "h-10 px-4 text-sm", variants[variant], extra,
);

/** A real link that looks like a button (never a <button> nested inside an <a>). */
export function ButtonLink({ variant = "secondary", size = "md", className, ...rest }: LinkProps & { variant?: Variant; size?: "sm" | "md" }) {
  return <Link {...rest} className={buttonClass(variant, size, className)} />;
}
export function ButtonAnchor({ variant = "secondary", size = "md", className, ...rest }: AnchorHTMLAttributes<HTMLAnchorElement> & { variant?: Variant; size?: "sm" | "md" }) {
  return <a {...rest} className={buttonClass(variant, size, className)} />;
}

export function Button({ variant = "secondary", size = "md", loading, className, children, disabled, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md"; loading?: boolean }) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={buttonClass(variant, size, className)}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

/* ---- Badge ------------------------------------------------------------------ */
const toneClass: Record<Tone, string> = {
  ai: "bg-signal-ai-soft text-signal-ai border-signal-ai/20",
  real: "bg-signal-real-soft text-signal-real border-signal-real/20",
  unsure: "bg-signal-unsure-soft text-signal-unsure border-signal-unsure/25",
  neutral: "bg-sunken text-ink-2 border-line",
};
export function Badge({ tone = "neutral", children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  return <span className={cx("inline-flex items-center gap-1 rounded border px-2 py-0.5 text-xs font-medium", toneClass[tone], className)}>{children}</span>;
}

/* ---- Notice ----------------------------------------------------------------- */
export function Notice({ kind = "info", title, children, action }: { kind?: "info" | "warn" | "error" | "success"; title?: string; children?: ReactNode; action?: ReactNode }) {
  const map = {
    info: { Icon: Info, cls: "border-brand/20 bg-brand-soft text-ink" },
    warn: { Icon: AlertTriangle, cls: "border-signal-unsure/30 bg-signal-unsure-soft text-ink" },
    error: { Icon: AlertCircle, cls: "border-signal-ai/30 bg-signal-ai-soft text-ink" },
    success: { Icon: CheckCircle2, cls: "border-signal-real/30 bg-signal-real-soft text-ink" },
  }[kind];
  const icon = { info: "text-brand", warn: "text-signal-unsure", error: "text-signal-ai", success: "text-signal-real" }[kind];
  return (
    <div role={kind === "error" ? "alert" : "status"} className={cx("flex gap-3 rounded-md border p-3.5 text-sm", map.cls)}>
      <map.Icon className={cx("mt-0.5 h-4 w-4 shrink-0", icon)} aria-hidden />
      <div className="min-w-0 flex-1">
        {title && <p className="font-semibold">{title}</p>}
        {children && <div className={cx("text-ink-2", title && "mt-0.5")}>{children}</div>}
      </div>
      {action}
    </div>
  );
}

/* ---- Skeleton / Empty ------------------------------------------------------------ */
export const Skeleton = ({ className }: { className?: string }) => <div aria-hidden className={cx("skeleton", className ?? "h-4 w-full")} />;

export function EmptyState({ icon, title, children, action }: { icon: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center px-6 py-12 text-center">
      <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-full bg-sunken text-ink-3" aria-hidden>{icon}</div>
      <h3 className="text-[15px] font-semibold">{title}</h3>
      {children && <p className="mt-1.5 max-w-md text-sm text-ink-3">{children}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

/* ---- Page structure ---------------------------------------------------------------- */
export function PageHeader({ title, description, actions, eyebrow }: { title: string; description?: ReactNode; actions?: ReactNode; eyebrow?: string }) {
  return (
    <header className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b border-line pb-6">
      <div className="min-w-0 max-w-2xl">
        {eyebrow && <p className="eyebrow mb-2">{eyebrow}</p>}
        <h1 className="text-[26px] font-semibold leading-tight">{title}</h1>
        {description && <p className="mt-2 text-[15px] leading-relaxed text-ink-3">{description}</p>}
      </div>
      {actions && <div className="no-print flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Section({ title, description, actions, children, className }: { title: string; description?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cx("mb-10", className)}>
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold">{title}</h2>
          {description && <p className="mt-0.5 text-sm text-ink-3">{description}</p>}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function Stat({ label, value, sub, mono }: { label: string; value: ReactNode; sub?: ReactNode; mono?: boolean }) {
  return (
    <div className="min-w-0 px-5 py-4">
      <p className="eyebrow">{label}</p>
      <p className={cx("mt-1.5 break-words text-[17px] font-semibold leading-snug tnum", mono && "font-mono text-sm")}>{value}</p>
      {sub && <p className="mt-0.5 text-xs text-ink-3">{sub}</p>}
    </div>
  );
}

export function KeyValue({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-2 text-sm">
      {items.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-ink-3">{k}</dt>
          <dd className="min-w-0 break-words font-medium">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/* ---- Meter ------------------------------------------------------------------- */
export function ScoreBar({ label, value, tone, threshold }: { label: string; value: number; tone: Tone; threshold?: number }) {
  const bar = { ai: "bg-signal-ai", real: "bg-signal-real", unsure: "bg-signal-unsure", neutral: "bg-ink-3" }[tone];
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-sm">
        <span className="font-medium">{label}</span>
        <span className="font-mono tnum">{value.toFixed(4)}</span>
      </div>
      <div className="relative h-2.5 rounded-full bg-sunken" role="meter" aria-label={`${label} score`} aria-valuemin={0} aria-valuemax={1} aria-valuenow={Number(value.toFixed(4))}>
        <div className={cx("h-full rounded-full", bar)} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
        {threshold !== undefined && (
          <div className="absolute -top-1 h-4.5 border-l-2 border-ink/70" style={{ left: `${threshold * 100}%`, height: "1.125rem" }} title={`Policy threshold ${threshold}`} />
        )}
      </div>
    </div>
  );
}

/* ---- Tabs -------------------------------------------------------------------- */
export function Tabs<T extends string>({ value, onChange, items, label }: { value: T; onChange: (v: T) => void; items: { id: T; label: string }[]; label: string }) {
  return (
    <div role="tablist" aria-label={label} className="mb-6 inline-flex rounded-md border border-line bg-sunken p-0.5">
      {items.map((t) => (
        <button key={t.id} role="tab" aria-selected={value === t.id} onClick={() => onChange(t.id)}
          className={cx("rounded px-4 py-1.5 text-sm font-medium transition-colors", value === t.id ? "bg-surface text-ink shadow-panel" : "text-ink-3 hover:text-ink")}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

/* ---- Dialog -------------------------------------------------------------------- */
export function Dialog({ open, onClose, title, children, footer }: { open: boolean; onClose: () => void; title: string; children: ReactNode; footer?: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    const node = ref.current;
    const focusables = () => node ? Array.from(node.querySelectorAll<HTMLElement>('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')).filter((el) => !el.hasAttribute("disabled")) : [];
    focusables()[0]?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.stopPropagation(); onClose(); return; }
      if (e.key !== "Tab") return;
      const f = focusables();
      if (f.length === 0) return;
      const first = f[0] as HTMLElement, last = f[f.length - 1] as HTMLElement;
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => { document.removeEventListener("keydown", onKey); document.body.style.overflow = ""; previous?.focus(); };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-ink/40 p-0 sm:items-center sm:p-4" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={ref} role="dialog" aria-modal="true" aria-labelledby="dialog-title" className="w-full max-w-md rounded-t-xl bg-surface p-6 shadow-pop sm:rounded-xl">
        <div className="mb-3 flex items-start justify-between gap-4">
          <h2 id="dialog-title" className="text-lg font-semibold">{title}</h2>
          <button onClick={onClose} aria-label="Close dialog" className="rounded p-1 text-ink-3 hover:bg-sunken hover:text-ink"><X className="h-4 w-4" /></button>
        </div>
        <div className="text-sm text-ink-2">{children}</div>
        {footer && <div className="mt-6 flex justify-end gap-2">{footer}</div>}
      </div>
    </div>
  );
}

export function ProgressBar({ value, label }: { value: number | null; label: string }) {
  return (
    <div>
      <div className="mb-1.5 flex justify-between text-xs text-ink-3"><span>{label}</span>{value !== null && <span className="tnum">{Math.round(value * 100)}%</span>}</div>
      <div className="h-1.5 overflow-hidden rounded-full bg-sunken" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={value === null ? undefined : Math.round(value * 100)}>
        {value === null
          ? <div className="h-full w-1/3 animate-indeterminate rounded-full bg-brand" />
          : <div className="h-full rounded-full bg-brand transition-[width] duration-200" style={{ width: `${value * 100}%` }} />}
      </div>
    </div>
  );
}
