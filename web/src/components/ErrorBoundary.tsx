import { Component, type ReactNode } from "react";
import { Button, Notice } from "@/components/ui";

const CHUNK_ERROR = /dynamically imported module|Loading chunk|Importing a module script failed|Failed to fetch/i;

/** Catches render-time failures (including a lazy page chunk that cannot be fetched) so the
 *  rest of the app, notably navigation, stays usable instead of unmounting to a blank page. */
export class ErrorBoundary extends Component<{ children: ReactNode; resetKey?: string }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) { return { error }; }

  componentDidUpdate(prev: { resetKey?: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  componentDidCatch(error: Error) { console.error("UI error:", error); }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    const network = CHUNK_ERROR.test(error.message);
    return (
      <div className="mx-auto max-w-xl py-10">
        <Notice kind="error" title={network ? "This page could not be loaded" : "Something went wrong on this page"}
          action={<Button size="sm" onClick={() => window.location.reload()}>Reload</Button>}>
          {network
            ? "The server may be unreachable, or the app was updated while this tab was open. Check your connection and reload."
            : "The rest of the app still works. Try another section, or reload this page."}
        </Notice>
      </div>
    );
  }
}
