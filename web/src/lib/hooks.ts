import { useCallback, useEffect, useRef, useState } from "react";

export interface AsyncState<T> { data: T | null; error: Error | null; loading: boolean; reload: () => void }

/** Runs `fn` on mount and whenever `deps` change; aborts stale requests. */
export function useAsync<T>(fn: (signal: AbortSignal) => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [state, setState] = useState<{ data: T | null; error: Error | null; loading: boolean }>({ data: null, error: null, loading: true });
  const [tick, setTick] = useState(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    const ctl = new AbortController();
    setState((s) => ({ ...s, loading: true, error: null }));
    fnRef.current(ctl.signal).then(
      (data) => { if (!ctl.signal.aborted) setState({ data, error: null, loading: false }); },
      (error: Error) => { if (!ctl.signal.aborted && error.name !== "AbortError") setState({ data: null, error, loading: false }); },
    );
    return () => ctl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { ...state, reload };
}

export function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

export function useDocumentTitle(title: string): void {
  useEffect(() => { document.title = `${title} · VerityAI`; }, [title]);
}
