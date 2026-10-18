import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api, ApiError } from "@/api/client";
import type { Health } from "@/api/types";

interface HealthCtx { health: Health | null; error: ApiError | null; loading: boolean; refresh: () => void }
const Ctx = createContext<HealthCtx>({ health: null, error: null, loading: true, refresh: () => {} });

export function HealthProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const ctl = new AbortController();
    const run = () => api.health(ctl.signal).then(
      (h) => { setHealth(h); setError(null); setLoading(false); },
      (e: Error) => { if (e.name !== "AbortError") { setError(e instanceof ApiError ? e : new ApiError(0, "network", e.message)); setLoading(false); } },
    );
    void run();
    const id = setInterval(() => void run(), 30_000);
    return () => { ctl.abort(); clearInterval(id); };
  }, [tick]);

  return <Ctx.Provider value={{ health, error, loading, refresh: () => setTick((t) => t + 1) }}>{children}</Ctx.Provider>;
}

export const useHealth = () => useContext(Ctx);
