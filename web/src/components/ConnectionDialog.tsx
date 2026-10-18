import { useEffect, useState } from "react";
import { api, defaultApiBase, getApiBase, setApiBaseOverride } from "@/api/client";
import { Button, Dialog, Notice } from "@/components/ui";
import { useHealth } from "@/lib/health";

/** Lets a separately hosted frontend point at any VerityAI API (no rebuild needed). */
export function ConnectionDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { refresh, health, error } = useHealth();
  const [value, setValue] = useState("");
  const [test, setTest] = useState<{ ok: boolean; msg: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { if (open) { setValue(getApiBase()); setTest(null); } }, [open]);

  async function check() {
    setBusy(true);
    const previous = getApiBase();
    try {
      setApiBaseOverride(value);
      const h = await api.health();
      setTest({ ok: true, msg: `Connected: VerityAI API v${h.version}.` });
    } catch (e) {
      setApiBaseOverride(previous === defaultApiBase() ? null : previous);
      setTest({ ok: false, msg: (e as Error).message });
    } finally { setBusy(false); }
  }

  function reset() { setApiBaseOverride(null); setValue(defaultApiBase()); setTest(null); refresh(); }

  return (
    <Dialog open={open} onClose={onClose} title="Connection"
      footer={<>
        <Button variant="ghost" onClick={reset}>Use default</Button>
        <Button onClick={async () => { await check(); refresh(); }} loading={busy}>Test and save</Button>
      </>}>
      <p className="mb-3">
        The web app talks to the VerityAI API at the address below. Leave it as is when the API is served from the
        same site; change it only if the API is hosted elsewhere.
      </p>
      <label htmlFor="api-base" className="mb-1 block text-xs font-semibold text-ink-2">API address</label>
      <input id="api-base" value={value} onChange={(e) => setValue(e.target.value)} spellCheck={false}
        className="h-10 w-full rounded-md border border-line-strong bg-surface px-3 font-mono text-[13px]" />
      <p className="mt-2 text-xs text-ink-3">Default for this site: <span className="font-mono">{defaultApiBase()}</span></p>
      <div className="mt-4 space-y-2">
        {test && <Notice kind={test.ok ? "success" : "error"}>{test.msg}</Notice>}
        {!test && error && <Notice kind="error" title="Server unreachable">{error.message}</Notice>}
        {!test && health && <Notice kind="info">Currently connected to API v{health.version}. Saved only in this browser.</Notice>}
      </div>
    </Dialog>
  );
}
