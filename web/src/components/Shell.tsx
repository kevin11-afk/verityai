import { BookOpen, FileText, FlaskConical, LayoutDashboard, Library, Menu, MessagesSquare, ScanSearch, Settings2, X } from "lucide-react";
import { Suspense, useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Skeleton, cx } from "@/components/ui";
import { ConnectionDialog } from "@/components/ConnectionDialog";
import { useHealth } from "@/lib/health";

const NAV = [
  { group: "Workspace", items: [
    { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
    { to: "/analyze", label: "Analyze media", icon: ScanSearch },
    { to: "/reports", label: "Reports", icon: FileText },
  ] },
  { group: "Intelligence", items: [
    { to: "/research", label: "Research", icon: MessagesSquare },
    { to: "/knowledge", label: "Knowledge base", icon: Library },
  ] },
  { group: "Quality", items: [{ to: "/evaluation", label: "Evaluation lab", icon: FlaskConical }] },
  { group: "Reference", items: [{ to: "/methodology", label: "Methodology & privacy", icon: BookOpen }] },
];

function Brand() {
  return (
    <div className="flex items-center gap-3">
      <svg viewBox="0 0 32 32" className="h-8 w-8 shrink-0" aria-hidden>
        <rect width="32" height="32" rx="7" fill="#0B1B33" />
        <path d="M8 10l8 13 8-13" fill="none" stroke="#F7F5F0" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx="16" cy="9" r="2.2" fill="#8C89EA" />
      </svg>
      <div className="leading-tight">
        <p className="text-[15px] font-semibold tracking-tight">VerityAI</p>
        <p className="text-[11px] text-ink-3">Intelligence for media authenticity</p>
      </div>
    </div>
  );
}

function ConnectionChip({ onClick }: { onClick: () => void }) {
  const { health, error, loading } = useHealth();
  const state = loading ? "checking" : error ? "offline" : health?.status === "ok" ? "ok" : "degraded";
  const dot = { checking: "bg-ink-3", offline: "bg-signal-ai", ok: "bg-signal-real", degraded: "bg-signal-unsure" }[state];
  const label = { checking: "Checking server…", offline: "Server unreachable", ok: "Server online", degraded: "Warming up" }[state];
  return (
    <button onClick={onClick} className="group flex w-full items-center gap-2.5 rounded-md border border-line bg-surface px-3 py-2 text-left text-xs hover:bg-sunken" aria-label={`${label}. Open connection settings`}>
      <span className={cx("h-2 w-2 shrink-0 rounded-full", dot)} aria-hidden />
      <span className="min-w-0 flex-1">
        <span className="block font-medium text-ink">{label}</span>
        <span className="block truncate text-ink-3">{health ? `API v${health.version} · Python ${health.python}` : "Check API address"}</span>
      </span>
      <Settings2 className="h-3.5 w-3.5 text-ink-3 group-hover:text-ink" aria-hidden />
    </button>
  );
}

function NavList({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Primary" className="space-y-6">
      {NAV.map((g) => (
        <div key={g.group}>
          <p className="eyebrow mb-1.5 px-3">{g.group}</p>
          <ul className="space-y-0.5">
            {g.items.map(({ to, label, icon: Icon, end }) => (
              <li key={to}>
                <NavLink to={to} end={end} onClick={onNavigate}
                  className={({ isActive }) => cx(
                    "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    isActive ? "bg-brand-soft text-brand-ink" : "text-ink-2 hover:bg-sunken hover:text-ink")}>
                  <Icon className="h-4 w-4 shrink-0" aria-hidden />{label}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

export function Shell() {
  const [drawer, setDrawer] = useState(false);
  const [connOpen, setConnOpen] = useState(false);
  const { pathname } = useLocation();

  useEffect(() => { setDrawer(false); window.scrollTo(0, 0); }, [pathname]);
  useEffect(() => {
    if (!drawer) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setDrawer(false); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [drawer]);

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[264px_1fr]">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-[60] focus:rounded-md focus:bg-ink focus:px-3 focus:py-2 focus:text-sm focus:text-white">Skip to content</a>

      {/* Desktop sidebar */}
      <aside className="no-print sticky top-0 hidden h-screen flex-col border-r border-line bg-canvas px-4 py-6 lg:flex">
        <div className="px-3"><Brand /></div>
        <div className="mt-8 flex-1 overflow-y-auto"><NavList /></div>
        <ConnectionChip onClick={() => setConnOpen(true)} />
      </aside>

      {/* Mobile top bar */}
      <div className="no-print sticky top-0 z-30 flex items-center justify-between border-b border-line bg-canvas/95 px-4 py-3 backdrop-blur lg:hidden">
        <Brand />
        <button onClick={() => setDrawer(true)} aria-label="Open navigation" aria-expanded={drawer} className="rounded-md p-2 hover:bg-sunken"><Menu className="h-5 w-5" /></button>
      </div>
      {drawer && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <div className="absolute inset-0 bg-ink/40" onClick={() => setDrawer(false)} />
          <div className="absolute inset-y-0 left-0 flex w-[280px] max-w-[85vw] flex-col bg-canvas px-4 py-5 shadow-pop">
            <div className="mb-6 flex items-center justify-between px-3">
              <Brand />
              <button onClick={() => setDrawer(false)} aria-label="Close navigation" className="rounded-md p-2 hover:bg-sunken"><X className="h-5 w-5" /></button>
            </div>
            <div className="flex-1 overflow-y-auto"><NavList onNavigate={() => setDrawer(false)} /></div>
            <ConnectionChip onClick={() => { setDrawer(false); setConnOpen(true); }} />
          </div>
        </div>
      )}

      <main id="main" tabIndex={-1} className="min-w-0 px-4 py-8 outline-none sm:px-8 lg:px-12 lg:py-10">
        <div className="mx-auto max-w-[1120px]">
          <ErrorBoundary resetKey={pathname}>
            <Suspense fallback={<div className="space-y-4" aria-busy="true" aria-label="Loading page"><Skeleton className="h-8 w-64" /><Skeleton className="h-4 w-96 max-w-full" /><Skeleton className="h-64 w-full" /></div>}>
              <Outlet />
            </Suspense>
          </ErrorBoundary>
        </div>
      </main>
      <ConnectionDialog open={connOpen} onClose={() => setConnOpen(false)} />
    </div>
  );
}
