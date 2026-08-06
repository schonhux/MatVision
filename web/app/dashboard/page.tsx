"use client";

import { useCallback, useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { listMatches, createMatch, clearToken, getToken, Match, ApiError } from "@/lib/api";

const STATUS_LABELS: Record<string, string> = {
  uploaded: "Uploaded",
  validating: "Preparing video",
  transcoding: "Transcoding",
  tracking: "Analyzing positions",
  extracting_pose: "Extracting pose",
  classifying_states: "Classifying states",
  detecting_events: "Detecting events",
  generating_insights: "Building report",
  complete: "Ready",
  failed: "Failed",
};

export default function DashboardPage() {
  const router = useRouter();
  const [matches, setMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setMatches(await listMatches());
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        clearToken();
        router.replace("/login");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to load matches");
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    queueMicrotask(refresh);
  }, [router, refresh]);

  async function handleFileSelected(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setUploading(true);
    try {
      const match = await createMatch(file, file.name);
      router.push(`/matches/${match.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  function handleLogout() {
    clearToken();
    router.push("/login");
  }

  return (
    <main className="mx-auto max-w-4xl px-4 py-10">
      <div className="mb-8 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Your matches</h1>
        <div className="flex items-center gap-3">
          <label className="cursor-pointer rounded bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-900">
            {uploading ? "Uploading..." : "Upload match"}
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/quicktime,video/webm"
              onChange={handleFileSelected}
              disabled={uploading}
              className="hidden"
            />
          </label>
          <Link href="/dataset" className="text-sm text-neutral-400 hover:text-neutral-200">
            Dataset
          </Link>
          <button onClick={handleLogout} className="text-sm text-neutral-400 hover:text-neutral-200">
            Log out
          </button>
        </div>
      </div>

      {error && (
        <p className="mb-4 rounded bg-red-950 border border-red-800 px-3 py-2 text-sm text-red-300">
          {error}
        </p>
      )}

      {loading ? (
        <p className="text-neutral-400">Loading...</p>
      ) : matches.length === 0 ? (
        <p className="text-neutral-400">
          No matches yet. Upload a folkstyle match (stationary camera, full mat visible,
          up to 10 minutes) to get started.
        </p>
      ) : (
        <ul className="divide-y divide-neutral-800">
          {matches.map((m) => (
            <li key={m.id}>
              <Link
                href={`/matches/${m.id}`}
                className="flex items-center justify-between py-4 hover:bg-neutral-900 px-2 -mx-2 rounded"
              >
                <div>
                  <p className="font-medium">{m.title}</p>
                  <p className="text-sm text-neutral-500">
                    {new Date(m.created_at).toLocaleString()}
                    {m.duration_seconds ? ` · ${Math.round(m.duration_seconds)}s` : ""}
                  </p>
                </div>
                <span
                  className={`text-sm rounded-full px-3 py-1 ${
                    m.status === "complete"
                      ? "bg-green-950 text-green-300"
                      : m.status === "failed"
                      ? "bg-red-950 text-red-300"
                      : "bg-neutral-800 text-neutral-300"
                  }`}
                >
                  {STATUS_LABELS[m.status] ?? m.status}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
