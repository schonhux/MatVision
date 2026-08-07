"use client";

/**
 * Dataset progress — makes labeling effort visible rather than guessed at.
 * Surfaces the BUILD_PLAN.md M2 gate (5 fully labeled matches) and lets you pull
 * the leakage-safe export once there's something worth exporting.
 */

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { getDatasetStats, exportDataset, getToken, DatasetStats } from "@/lib/api";

export default function DatasetPage() {
  const router = useRouter();
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStats(await getDatasetStats());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dataset stats");
    }
  }, []);

  useEffect(() => {
    if (!getToken()) { router.replace("/login"); return; }
    queueMicrotask(refresh);
  }, [router, refresh]);

  async function handleExport() {
    try {
      const data = await exportDataset();
      // Download as a file rather than dumping JSON on screen — this is a dataset
      // artifact meant to be fed to training code, not read in a browser.
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `matvision-dataset-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
    }
  }

  return (
    <main className="mx-auto max-w-3xl px-4 py-10">
      <Link href="/dashboard" className="text-sm text-neutral-400 hover:text-neutral-200">
        &larr; Back to matches
      </Link>
      <h1 className="mt-2 mb-6 text-2xl font-semibold">Dataset progress</h1>

      {error && (
        <p className="mb-4 rounded border border-red-800 bg-red-950 px-3 py-2 text-sm text-red-300">{error}</p>
      )}

      {!stats ? (
        <p className="text-neutral-400">Loading…</p>
      ) : (
        <>
          <section className="mb-6 rounded border border-neutral-800 bg-neutral-900 p-4">
            <div className="mb-2 flex items-baseline justify-between">
              <h2 className="font-medium">Labeled matches</h2>
              <span className="text-sm text-neutral-400">
                {stats.m2_gate.current} / {stats.m2_gate.target_matches} target
              </span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded bg-neutral-800">
              <div
                className={stats.m2_gate.met ? "h-full bg-green-500" : "h-full bg-neutral-400"}
                style={{
                  width: `${Math.min(100, (stats.m2_gate.current / stats.m2_gate.target_matches) * 100)}%`,
                }}
              />
            </div>
          </section>

          <section className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-5">
            {[
              ["Matches", stats.total_matches],
              ["Annotated", stats.annotated_matches],
              ["Events", stats.total_events],
              ["Corrections", stats.total_corrections],
              ["Minutes labeled", stats.labeled_minutes],
            ].map(([label, value]) => (
              <div key={label as string} className="rounded border border-neutral-800 bg-neutral-900 p-3">
                <p className="text-2xl font-semibold">{value}</p>
                <p className="text-xs text-neutral-500">{label}</p>
              </div>
            ))}
          </section>

          <section className="mb-6 grid gap-4 sm:grid-cols-2">
            <div className="rounded border border-neutral-800 bg-neutral-900 p-4">
              <h2 className="mb-2 text-sm font-medium">Events by type</h2>
              {Object.keys(stats.events_by_type).length === 0 ? (
                <p className="text-xs text-neutral-500">None yet.</p>
              ) : (
                <ul className="space-y-1 text-sm">
                  {Object.entries(stats.events_by_type)
                    .sort((a, b) => b[1] - a[1])
                    .map(([type, count]) => (
                      <li key={type} className="flex justify-between">
                        <span className="text-neutral-400">{type.replace(/_/g, " ")}</span>
                        <span className="font-mono">{count}</span>
                      </li>
                    ))}
                </ul>
              )}
            </div>

            <div className="rounded border border-neutral-800 bg-neutral-900 p-4">
              <h2 className="mb-2 text-sm font-medium">Match states</h2>
              {Object.keys(stats.states_by_type).length === 0 ? (
                <p className="text-xs text-neutral-500">None yet.</p>
              ) : (
                <ul className="space-y-1 text-sm">
                  {Object.entries(stats.states_by_type)
                    .sort((a, b) => b[1] - a[1])
                    .map(([state, count]) => (
                      <li key={state} className="flex justify-between">
                        <span className="text-neutral-400">{state}</span>
                        <span className="font-mono">{count}</span>
                      </li>
                    ))}
                </ul>
              )}
            </div>
          </section>

          <button
            onClick={handleExport}
            disabled={stats.annotated_matches === 0}
            className="rounded bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-900 disabled:opacity-40"
          >
            Export dataset (leakage-safe splits)
          </button>
          {stats.annotated_matches === 0 && (
            <p className="mt-2 text-xs text-neutral-500">
              Label a match and mark it complete to enable export.
            </p>
          )}
        </>
      )}
    </main>
  );
}
