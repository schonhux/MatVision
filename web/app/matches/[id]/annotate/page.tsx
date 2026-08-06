"use client";

/**
 * The annotation console — Layer 2's core deliverable.
 *
 * This is the tool that builds the training dataset (PROJECT_GUIDE.md Layer 2:
 * "the product is the dataset tool"). Everything here exists to make labeling fast
 * and accurate: frame-stepping, keyboard shortcuts, and label forms constrained to
 * the exact vocabularies the models will be trained against.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  getMatch, getVideoUrl, getToken,
  listEvents, createEvent, updateEvent, deleteEvent,
  listStates, createState, deleteState,
  listAthletes, setAthlete, updateMatchAnnotation,
  Match, MatchEvent, StateSegment, MatchAthlete,
  Initiator, Outcome, StateName,
} from "@/lib/api";

const FPS = 30; // analysis copy is normalized to 30fps by the transcode stage
const FRAME_MS = 1000 / FPS;

const EVENT_TYPES = [
  "shot_attempt", "takedown", "defended_shot", "escape", "restart",
  "reversal", "near_fall", "stalemate", "other",
];
const TECHNIQUES = [
  "single_leg", "double_leg", "high_crotch", "ankle_pick", "duck_under",
  "arm_drag", "snap_down", "front_headlock", "other",
];
const STATES: StateName[] = ["neutral", "top", "bottom", "scramble", "stopped"];
const OUTCOMES: Outcome[] = ["successful", "failed", "countered", "stalemate"];

function fmt(ms: number): string {
  const total = ms / 1000;
  const m = Math.floor(total / 60);
  const s = (total % 60).toFixed(2).padStart(5, "0");
  return `${m}:${s}`;
}

export default function AnnotatePage() {
  const params = useParams();
  const router = useRouter();
  const matchId = params.id as string;
  const videoRef = useRef<HTMLVideoElement>(null);

  const [match, setMatch] = useState<Match | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [events, setEvents] = useState<MatchEvent[]>([]);
  const [states, setStates] = useState<StateSegment[]>([]);
  const [athletes, setAthletes] = useState<MatchAthlete[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [currentMs, setCurrentMs] = useState(0);

  const [markIn, setMarkIn] = useState<number | null>(null);
  const [markOut, setMarkOut] = useState<number | null>(null);
  const [mode, setMode] = useState<"event" | "state">("event");

  // Event form
  const [evType, setEvType] = useState(EVENT_TYPES[0]);
  const [evInitiator, setEvInitiator] = useState<Initiator>("user");
  const [evOutcome, setEvOutcome] = useState<Outcome>("successful");
  const [evTechnique, setEvTechnique] = useState("");
  const [evNote, setEvNote] = useState("");

  // State form
  const [stName, setStName] = useState<StateName>("neutral");
  const [stControlling, setStControlling] = useState<Initiator>("user");

  const refresh = useCallback(async () => {
    try {
      const [m, e, s, a] = await Promise.all([
        getMatch(matchId), listEvents(matchId), listStates(matchId, "human"), listAthletes(matchId),
      ]);
      setMatch(m); setEvents(e); setStates(s); setAthletes(a);
      if (m.video_keys.original || m.video_keys.analysis_720p) {
        try {
          const { url } = await getVideoUrl(matchId);
          setVideoUrl(url);
        } catch { /* not ready yet */ }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load match");
    }
  }, [matchId]);

  useEffect(() => {
    if (!getToken()) { router.replace("/login"); return; }
    queueMicrotask(refresh);
  }, [router, refresh]);

  // --- playhead + frame stepping ---------------------------------------------

  const step = useCallback((frames: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.pause();
    v.currentTime = Math.max(0, v.currentTime + (frames * FRAME_MS) / 1000);
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      // Don't hijack typing in form fields.
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      const v = videoRef.current;
      if (!v) return;

      switch (e.key) {
        case ",": case "ArrowLeft": e.preventDefault(); step(e.shiftKey ? -10 : -1); break;
        case ".": case "ArrowRight": e.preventDefault(); step(e.shiftKey ? 10 : 1); break;
        case " ": e.preventDefault(); v.paused ? v.play() : v.pause(); break;
        case "i": setMarkIn(Math.round(v.currentTime * 1000)); break;
        case "o": setMarkOut(Math.round(v.currentTime * 1000)); break;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step]);

  async function submitEvent(e: React.FormEvent) {
    e.preventDefault();
    if (markIn === null || markOut === null) { setError("Set both in (i) and out (o) points"); return; }
    if (markOut <= markIn) { setError("Out point must come after in point"); return; }
    try {
      await createEvent(matchId, {
        type: evType, start_ms: markIn, end_ms: markOut,
        note: evNote || undefined,
        initiator: evInitiator, outcome: evOutcome,
        technique: evTechnique || undefined,
      } as Parameters<typeof createEvent>[1]);
      setMarkIn(null); setMarkOut(null); setEvNote(""); setError(null);
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save event");
    }
  }

  async function submitState(e: React.FormEvent) {
    e.preventDefault();
    if (markIn === null || markOut === null) { setError("Set both in (i) and out (o) points"); return; }
    try {
      await createState(matchId, {
        state: stName, start_ms: markIn, end_ms: markOut,
        controlling: (stName === "top" || stName === "bottom") ? stControlling : undefined,
      });
      setMarkIn(null); setMarkOut(null); setError(null);
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save state segment");
    }
  }

  async function identifyAthlete(role: Initiator) {
    const v = videoRef.current;
    const name = window.prompt(`Name of the "${role}" wrestler?`);
    if (name === null) return;
    try {
      await setAthlete(matchId, {
        role, athlete_name: name || undefined,
        seed_frame_ms: v ? Math.round(v.currentTime * 1000) : undefined,
      });
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to identify athlete");
    }
  }

  async function markComplete() {
    try {
      await updateMatchAnnotation(matchId, { annotation_complete: true });
      setError(null);
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not mark complete");
    }
  }

  async function setVenue() {
    const venue = window.prompt("Venue (used for leakage-safe dataset splitting):", match?.venue ?? "");
    if (venue === null) return;
    try { await updateMatchAnnotation(matchId, { venue }); refresh(); }
    catch (err) { setError(err instanceof Error ? err.message : "Failed to set venue"); }
  }

  function seek(ms: number) {
    if (videoRef.current) videoRef.current.currentTime = ms / 1000;
  }

  if (!match) {
    return <main className="mx-auto max-w-6xl px-4 py-10">
      {error ? <p className="text-red-400">{error}</p> : <p className="text-neutral-400">Loading…</p>}
    </main>;
  }

  return (
    <main className="mx-auto max-w-6xl px-4 py-6">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <Link href={`/matches/${matchId}`} className="text-sm text-neutral-400 hover:text-neutral-200">
            &larr; Back to match
          </Link>
          <h1 className="mt-1 text-xl font-semibold">Annotate: {match.title}</h1>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <button onClick={setVenue} className="rounded bg-neutral-800 px-3 py-1.5">
            Venue: {match.venue ?? "not set"}
          </button>
          {match.annotation_complete ? (
            <span className="rounded-full bg-green-950 px-3 py-1.5 text-green-300">Complete</span>
          ) : (
            <button onClick={markComplete} className="rounded bg-neutral-100 px-3 py-1.5 font-medium text-neutral-900">
              Mark complete
            </button>
          )}
        </div>
      </div>

      {error && (
        <p className="mb-3 rounded border border-red-800 bg-red-950 px-3 py-2 text-sm text-red-300">{error}</p>
      )}

      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <div>
          {videoUrl ? (
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              className="w-full rounded bg-black"
              onTimeUpdate={(e) => setCurrentMs(Math.round(e.currentTarget.currentTime * 1000))}
            />
          ) : (
            <div className="flex h-72 items-center justify-center rounded bg-neutral-900 text-neutral-500">
              Video not ready yet
            </div>
          )}

          {/* Frame-accurate transport */}
          <div className="mt-2 flex flex-wrap items-center gap-2 text-sm">
            <span className="font-mono text-neutral-400">{fmt(currentMs)}</span>
            <span className="font-mono text-xs text-neutral-600">
              frame {Math.round(currentMs / FRAME_MS)}
            </span>
            <button onClick={() => step(-10)} className="rounded bg-neutral-800 px-2 py-1">⏪ 10f</button>
            <button onClick={() => step(-1)} className="rounded bg-neutral-800 px-2 py-1">◀ 1f</button>
            <button onClick={() => step(1)} className="rounded bg-neutral-800 px-2 py-1">1f ▶</button>
            <button onClick={() => step(10)} className="rounded bg-neutral-800 px-2 py-1">10f ⏩</button>
            <span className="ml-2 text-xs text-neutral-600">
              keys: , / . step · shift = 10f · space play · i/o mark
            </span>
          </div>

          {/* In / out points */}
          <div className="mt-3 flex flex-wrap items-center gap-2 rounded border border-neutral-800 bg-neutral-900 p-3 text-sm">
            <button
              onClick={() => setMarkIn(currentMs)}
              className="rounded bg-neutral-800 px-3 py-1.5"
            >
              Mark in (i){markIn !== null && `: ${fmt(markIn)}`}
            </button>
            <button
              onClick={() => setMarkOut(currentMs)}
              className="rounded bg-neutral-800 px-3 py-1.5"
            >
              Mark out (o){markOut !== null && `: ${fmt(markOut)}`}
            </button>
            {markIn !== null && markOut !== null && markOut > markIn && (
              <span className="text-neutral-400">span {((markOut - markIn) / 1000).toFixed(2)}s</span>
            )}
          </div>

          {/* Label forms */}
          <div className="mt-3 rounded border border-neutral-800 bg-neutral-900 p-3">
            <div className="mb-3 flex gap-2 text-sm">
              <button
                onClick={() => setMode("event")}
                className={`rounded px-3 py-1.5 ${mode === "event" ? "bg-neutral-100 text-neutral-900" : "bg-neutral-800"}`}
              >Event</button>
              <button
                onClick={() => setMode("state")}
                className={`rounded px-3 py-1.5 ${mode === "state" ? "bg-neutral-100 text-neutral-900" : "bg-neutral-800"}`}
              >Match state</button>
            </div>

            {mode === "event" ? (
              <form onSubmit={submitEvent} className="flex flex-wrap items-end gap-2 text-sm">
                <select value={evType} onChange={(e) => setEvType(e.target.value)} className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5">
                  {EVENT_TYPES.map((t) => <option key={t} value={t}>{t.replace(/_/g, " ")}</option>)}
                </select>
                <select value={evInitiator} onChange={(e) => setEvInitiator(e.target.value as Initiator)} className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5">
                  <option value="user">by me</option>
                  <option value="opponent">by opponent</option>
                </select>
                <select value={evOutcome} onChange={(e) => setEvOutcome(e.target.value as Outcome)} className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5">
                  {OUTCOMES.map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
                <select value={evTechnique} onChange={(e) => setEvTechnique(e.target.value)} className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5">
                  <option value="">technique…</option>
                  {TECHNIQUES.map((t) => <option key={t} value={t}>{t.replace(/_/g, " ")}</option>)}
                </select>
                <input
                  value={evNote} onChange={(e) => setEvNote(e.target.value)} placeholder="note"
                  className="min-w-[8rem] flex-1 rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5"
                />
                <button type="submit" className="rounded bg-neutral-100 px-3 py-1.5 font-medium text-neutral-900">
                  Save event
                </button>
              </form>
            ) : (
              <form onSubmit={submitState} className="flex flex-wrap items-end gap-2 text-sm">
                <select value={stName} onChange={(e) => setStName(e.target.value as StateName)} className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5">
                  {STATES.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
                {(stName === "top" || stName === "bottom") && (
                  <select value={stControlling} onChange={(e) => setStControlling(e.target.value as Initiator)} className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1.5">
                    <option value="user">me on top</option>
                    <option value="opponent">opponent on top</option>
                  </select>
                )}
                <button type="submit" className="rounded bg-neutral-100 px-3 py-1.5 font-medium text-neutral-900">
                  Save state
                </button>
              </form>
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          <section className="rounded border border-neutral-800 bg-neutral-900 p-3">
            <h2 className="mb-2 text-sm font-medium">Athletes</h2>
            {(["user", "opponent"] as Initiator[]).map((role) => {
              const a = athletes.find((x) => x.role === role);
              return (
                <div key={role} className="mb-1.5 flex items-center justify-between text-sm">
                  <span className="text-neutral-400">{role === "user" ? "Me" : "Opponent"}</span>
                  <button onClick={() => identifyAthlete(role)} className="rounded bg-neutral-800 px-2 py-1 text-xs">
                    {a?.athlete_name ?? "identify…"}
                  </button>
                </div>
              );
            })}
          </section>

          <section className="rounded border border-neutral-800 bg-neutral-900 p-3">
            <h2 className="mb-2 text-sm font-medium">State segments ({states.length})</h2>
            {states.length === 0 ? (
              <p className="text-xs text-neutral-500">None yet.</p>
            ) : (
              <ul className="max-h-56 space-y-1 overflow-y-auto text-sm">
                {states.map((s) => (
                  <li key={s.id} className="flex items-center justify-between gap-2">
                    <button onClick={() => seek(s.start_ms)} className="truncate text-left hover:text-neutral-300">
                      <span className="font-mono text-xs text-neutral-500">{fmt(s.start_ms)}</span>{" "}
                      {s.state}{s.controlling ? ` (${s.controlling})` : ""}
                    </button>
                    <button
                      onClick={async () => { await deleteState(matchId, s.id); refresh(); }}
                      className="text-xs text-neutral-600 hover:text-red-400"
                    >✕</button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="rounded border border-neutral-800 bg-neutral-900 p-3">
            <h2 className="mb-2 text-sm font-medium">Events ({events.length})</h2>
            {events.length === 0 ? (
              <p className="text-xs text-neutral-500">None yet.</p>
            ) : (
              <ul className="max-h-56 space-y-1 overflow-y-auto text-sm">
                {events.map((ev) => (
                  <li key={ev.id} className="flex items-center justify-between gap-2">
                    <button onClick={() => seek(ev.start_ms)} className="truncate text-left hover:text-neutral-300">
                      <span className="font-mono text-xs text-neutral-500">{fmt(ev.start_ms)}</span>{" "}
                      {ev.type.replace(/_/g, " ")}
                      {ev.outcome && <span className="text-neutral-500"> · {ev.outcome}</span>}
                    </button>
                    <button
                      onClick={async () => { await deleteEvent(matchId, ev.id); refresh(); }}
                      className="text-xs text-neutral-600 hover:text-red-400"
                    >✕</button>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}
