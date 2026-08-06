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

export { API_URL, ApiError };
