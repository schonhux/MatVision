"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  getMatch,
  getVideoUrl,
  listJobs,
  listEvents,
  createEvent,
  cutClip,
  getClipUrl,
  getToken,
  Match,
  Job,
  MatchEvent,
} from "@/lib/api";

const STAGE_LABELS: Record<string, string> = {
  validate: "Preparing video",
  transcode: "Transcoding",
  detect_track: "Identifying wrestlers",
  pose: "Analyzing positions",
  features: "Computing features",
  states: "Classifying match states",
  events: "Detecting scoring events",
  consolidate: "Comparing techniques",
  clips: "Generating clips",
  stats: "Building statistics",
  observations: "Finding patterns",
  report: "Building report",
};

const EVENT_TYPES = ["shot_attempt", "takedown", "escape", "reversal", "restart", "other"];

function formatMs(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function MatchDetailPage() {
  const params = useParams();
  const router = useRouter();
  const matchId = params.id as string;

  const videoRef = useRef<HTMLVideoElement>(null);
  const [match, setMatch] = useState<Match | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [events, setEvents] = useState<MatchEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Manual tagging form state
  const [tagStart, setTagStart] = useState<number | null>(null);
  const [tagEnd, setTagEnd] = useState<number | null>(null);
  const [tagType, setTagType] = useState(EVENT_TYPES[0]);
  const [tagNote, setTagNote] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [m, j, e] = await Promise.all([
        getMatch(matchId),
        listJobs(matchId),
        listEvents(matchId),
      ]);
      setMatch(m);
      setJobs(j);
      setEvents(e);

      if (m.video_keys.original || m.video_keys.analysis_720p) {
        try {
          const { url } = await getVideoUrl(matchId);
          setVideoUrl(url);
        } catch {
          /* not ready yet — fine, we'll get it on the next poll */
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load match");
    }
  }, [matchId]);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    queueMicrotask(refresh);
    const interval = setInterval(() => {
      if (match?.status !== "complete" && match?.status !== "failed") {
        refresh();
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [router, refresh, match?.status]);

  function markStart() {
    if (videoRef.current) setTagStart(Math.round(videoRef.current.currentTime * 1000));
  }

  function markEnd() {
    if (videoRef.current) setTagEnd(Math.round(videoRef.current.currentTime * 1000));
  }

  async function submitTag(e: React.FormEvent) {
    e.preventDefault();
    if (tagStart === null || tagEnd === null) {
      setError("Mark a start and end point on the video first");
      return;
    }
    if (tagEnd <= tagStart) {
      setError("End must be after start");
      return;
    }
    try {
      await createEvent(matchId, {
        type: tagType,
        start_ms: tagStart,
        end_ms: tagEnd,
        note: tagNote || undefined,
      });
      setTagStart(null);
      setTagEnd(null);
      setTagNote("");
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save tag");
    }
  }

  function seekTo(ms: number) {
    if (videoRef.current) {
      videoRef.current.currentTime = ms / 1000;
      videoRef.current.play();
    }
  }

  async function handleCutClip(eventId: string) {
    try {
      await cutClip(matchId, eventId);
      setTimeout(refresh, 3000);
      setTimeout(refresh, 8000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to cut clip");
    }
  }

  async function playClip(eventId: string) {
    try {
      const { url } = await getClipUrl(matchId, eventId);
      window.open(url, "_blank");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Clip not ready yet");
    }
  }

  if (!match) {
    return (
      <main className="mx-auto max-w-5xl px-4 py-10">
        {error ? <p className="text-red-400">{error}</p> : <p className="text-neutral-400">Loading...</p>}
      </main>
    );
  }

  const isProcessing = match.status !== "complete" && match.status !== "failed";

  return (
    <main className="mx-auto max-w-5xl px-4 py-8">
      <Link href="/dashboard" className="text-sm text-neutral-400 hover:text-neutral-200">
        &larr; Back to matches
      </Link>
      <h1 className="mt-2 mb-6 text-2xl font-semibold">{match.title}</h1>

      {error && (
        <p className="mb-4 rounded bg-red-950 border border-red-800 px-3 py-2 text-sm text-red-300">
          {error}
        </p>
      )}

      {/* Pipeline progress */}
      {isProcessing && (
        <div className="mb-6 rounded border border-neutral-800 bg-neutral-900 p-4">
          <p className="mb-2 text-sm text-neutral-400">Processing...</p>
          <ul className="space-y-1 text-sm">
            {jobs.map((j) => (
              <li key={j.id} className="flex items-center gap-2">
                <span
                  className={
                    j.status === "complete"
                      ? "text-green-400"
                      : j.status === "running"
                      ? "text-yellow-400"
                      : j.status === "failed"
                      ? "text-red-400"
                      : "text-neutral-600"
                  }
                >
                  {j.status === "complete" ? "✓" : j.status === "running" ? "●" : j.status === "failed" ? "✗" : "○"}
                </span>
                <span>{STAGE_LABELS[j.stage] ?? j.stage}</span>
                {j.error && <span className="text-red-400 text-xs">— {j.error.slice(0, 100)}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Player */}
      {videoUrl ? (
        <video ref={videoRef} src={videoUrl} controls className="w-full rounded bg-black" />
      ) : (
        <div className="flex h-64 items-center justify-center rounded bg-neutral-900 text-neutral-500">
          Video not ready yet
        </div>
      )}

      {/* Manual tagging */}
      <section className="mt-6 rounded border border-neutral-800 bg-neutral-900 p-4">
        <h2 className="mb-3 font-medium">Tag an event</h2>
        <form onSubmit={submitTag} className="flex flex-wrap items-end gap-3">
          <div>
            <button type="button" onClick={markStart} className="rounded bg-neutral-800 px-3 py-1.5 text-sm">
              Mark start {tagStart !== null && `(${formatMs(tagStart)})`}
            </button>
          </div>
          <div>
            <button type="button" onClick={markEnd} className="rounded bg-neutral-800 px-3 py-1.5 text-sm">
              Mark end {tagEnd !== null && `(${formatMs(tagEnd)})`}
            </button>
          </div>
          <select
            value={tagType}
            onChange={(e) => setTagType(e.target.value)}
            className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm"
          >
            {EVENT_TYPES.map((t) => (
              <option key={t} value={t}>
                {t.replace("_", " ")}
              </option>
            ))}
          </select>
          <input
            type="text"
            placeholder="Note (optional)"
            value={tagNote}
            onChange={(e) => setTagNote(e.target.value)}
            className="flex-1 min-w-[10rem] rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm"
          />
          <button type="submit" className="rounded bg-neutral-100 px-4 py-1.5 text-sm font-medium text-neutral-900">
            Save tag
          </button>
        </form>
      </section>

      {/* Timeline */}
      <section className="mt-6">
        <h2 className="mb-3 font-medium">Timeline</h2>
        {events.length === 0 ? (
          <p className="text-sm text-neutral-500">No events tagged yet.</p>
        ) : (
          <ul className="divide-y divide-neutral-800">
            {events.map((ev) => (
              <li key={ev.id} className="flex items-center justify-between py-3">
                <button onClick={() => seekTo(ev.start_ms)} className="text-left hover:text-neutral-300">
                  <span className="font-mono text-sm text-neutral-500">{formatMs(ev.start_ms)}</span>{" "}
                  <span className="font-medium">{ev.type.replace("_", " ")}</span>
                  {ev.note && <span className="text-neutral-500"> — {ev.note}</span>}
                </button>
                {ev.clip_key ? (
                  <button
                    onClick={() => playClip(ev.id)}
                    className="rounded bg-neutral-800 px-3 py-1 text-xs hover:bg-neutral-700"
                  >
                    Play clip
                  </button>
                ) : (
                  <button
                    onClick={() => handleCutClip(ev.id)}
                    className="rounded bg-neutral-800 px-3 py-1 text-xs hover:bg-neutral-700"
                  >
                    Cut clip
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
