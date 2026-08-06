const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("matvision_token");
}

export function setToken(token: string): void {
  localStorage.setItem("matvision_token", token);
}

export function clearToken(): void {
  localStorage.removeItem("matvision_token");
}

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(
  path: string,
  options: RequestInit & { auth?: boolean } = {}
): Promise<T> {
  const { auth = true, headers, ...rest } = options;
  const finalHeaders: Record<string, string> = {
    ...(headers as Record<string, string>),
  };

  if (auth) {
    const token = getToken();
    if (token) finalHeaders["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}${path}`, { ...rest, headers: finalHeaders });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* response wasn't JSON — fall back to statusText */
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}

// --- Auth ---------------------------------------------------------------

export async function signup(email: string, password: string) {
  return request<{ access_token: string }>("/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
    auth: false,
  });
}

export async function login(email: string, password: string) {
  const form = new URLSearchParams();
  form.set("username", email);
  form.set("password", password);
  return request<{ access_token: string }>("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form.toString(),
    auth: false,
  });
}

export async function getCurrentUser() {
  return request<{ id: string; email: string; created_at: string }>("/auth/me");
}

// --- Matches --------------------------------------------------------------

export interface Match {
  id: string;
  title: string;
  style: string;
  status: string;
  duration_seconds: number | null;
  video_keys: Record<string, string>;
  venue: string | null;
  annotation_complete: boolean;
  created_at: string;
  updated_at: string;
}

export async function listMatches() {
  return request<Match[]>("/matches");
}

export async function getMatch(matchId: string) {
  return request<Match>(`/matches/${matchId}`);
}

export async function getVideoUrl(matchId: string) {
  return request<{ url: string; source: string }>(`/matches/${matchId}/video-url`);
}

export async function createMatch(file: File, title: string) {
  const presign = await request<{
    match_id: string;
    upload_url: string;
    object_key: string;
  }>("/matches", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title,
      filename: file.name,
      content_type: file.type || "video/mp4",
      size_bytes: file.size,
    }),
  });

  const uploadRes = await fetch(presign.upload_url, {
    method: "PUT",
    headers: { "Content-Type": file.type || "video/mp4" },
    body: file,
  });
  if (!uploadRes.ok) {
    throw new Error(`Direct upload to storage failed: ${uploadRes.status}`);
  }

  return request<Match>(`/matches/${presign.match_id}/complete`, { method: "POST" });
}

// --- Jobs (pipeline progress) -----------------------------------------------

export interface Job {
  id: string;
  match_id: string;
  stage: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
}

export async function listJobs(matchId: string) {
  return request<Job[]>(`/matches/${matchId}/jobs`);
}

// --- Events ------------------------------------------------

export interface MatchEvent {
  id: string;
  match_id: string;
  type: string;
  start_ms: number;
  end_ms: number;
  note: string | null;
  source: string;
  // Layer 2 annotation labels
  initiator: "user" | "opponent" | null;
  outcome: "successful" | "failed" | "countered" | "stalemate" | null;
  state_before: string | null;
  state_after: string | null;
  opponent_response: string | null;
  technique: string | null;
  detail: Record<string, unknown>;
  annotator_id: string | null;
  clip_key: string | null;
  created_at: string;
}

export async function listEvents(matchId: string) {
  return request<MatchEvent[]>(`/matches/${matchId}/events`);
}

export async function createEvent(
  matchId: string,
  event: { type: string; start_ms: number; end_ms: number; note?: string }
) {
  return request<MatchEvent>(`/matches/${matchId}/events`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(event),
  });
}

export async function cutClip(matchId: string, eventId: string) {
  return request<MatchEvent>(`/matches/${matchId}/events/${eventId}/cut-clip`, {
    method: "POST",
  });
}

export async function getClipUrl(matchId: string, eventId: string) {
  return request<{ url: string }>(`/matches/${matchId}/events/${eventId}/clip-url`);
}

// --- Layer 2: annotation ------------------------------------------------------

export type Initiator = "user" | "opponent";
export type Outcome = "successful" | "failed" | "countered" | "stalemate";
export type StateName = "neutral" | "top" | "bottom" | "scramble" | "stopped";

export interface EventLabels {
  initiator?: Initiator | null;
  outcome?: Outcome | null;
  state_before?: StateName | null;
  state_after?: StateName | null;
  opponent_response?: string | null;
  technique?: string | null;
  detail?: Record<string, unknown>;
}

export async function updateEvent(
  matchId: string,
  eventId: string,
  updates: Partial<{ type: string; start_ms: number; end_ms: number; note: string }> & EventLabels
) {
  return request<MatchEvent>(`/matches/${matchId}/events/${eventId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
}

export async function deleteEvent(matchId: string, eventId: string) {
  return request<void>(`/matches/${matchId}/events/${eventId}`, { method: "DELETE" });
}

export interface StateSegment {
  id: string;
  match_id: string;
  state: StateName;
  start_ms: number;
  end_ms: number;
  controlling: Initiator | null;
  confidence: number | null;
  source: string;
  annotator_id: string | null;
  created_at: string;
}

export async function listStates(matchId: string) {
  return request<StateSegment[]>(`/matches/${matchId}/states`);
}

export async function createState(
  matchId: string,
  segment: { state: StateName; start_ms: number; end_ms: number; controlling?: Initiator }
) {
  return request<StateSegment>(`/matches/${matchId}/states`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(segment),
  });
}

export async function updateState(
  matchId: string,
  segmentId: string,
  updates: Partial<{ state: StateName; start_ms: number; end_ms: number; controlling: Initiator }>
) {
  return request<StateSegment>(`/matches/${matchId}/states/${segmentId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
}

export async function deleteState(matchId: string, segmentId: string) {
  return request<void>(`/matches/${matchId}/states/${segmentId}`, { method: "DELETE" });
}

export interface MatchAthlete {
  id: string;
  match_id: string;
  role: Initiator;
  athlete_name: string | null;
  singlet_color: string | null;
  seed_frame_ms: number | null;
  seed_bbox: Record<string, number>;
  created_at: string;
}

export async function listAthletes(matchId: string) {
  return request<MatchAthlete[]>(`/matches/${matchId}/athletes`);
}

export async function setAthlete(
  matchId: string,
  athlete: {
    role: Initiator;
    athlete_name?: string;
    singlet_color?: string;
    seed_frame_ms?: number;
    seed_bbox?: { x1: number; y1: number; x2: number; y2: number };
  }
) {
  return request<MatchAthlete>(`/matches/${matchId}/athletes`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(athlete),
  });
}

export async function updateMatchAnnotation(
  matchId: string,
  updates: { venue?: string; annotation_complete?: boolean }
) {
  return request<Match>(`/matches/${matchId}/annotation`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
}

// --- Layer 2: dataset ------------------------------------------------------

export interface DatasetStats {
  total_matches: number;
  annotated_matches: number;
  total_events: number;
  total_state_segments: number;
  labeled_minutes: number;
  events_by_type: Record<string, number>;
  states_by_type: Record<string, number>;
  m2_gate: { target_matches: number; current: number; met: boolean };
}

export async function getDatasetStats() {
  return request<DatasetStats>("/datasets/stats");
}

export async function exportDataset(seed = 42) {
  return request<Record<string, unknown>>(`/datasets/export?seed=${seed}`);
}

export { API_URL, ApiError };
