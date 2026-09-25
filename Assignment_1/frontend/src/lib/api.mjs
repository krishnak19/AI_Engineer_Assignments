const DEFAULT_API_BASE_URL = "http://localhost:8000";
export class ApiError extends Error { constructor(message, status) { super(message); this.name = "ApiError"; this.status = status; } }
export function getApiBaseUrl() { return (process.env.NEXT_PUBLIC_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/$/, ""); }
export async function apiRequest(path, options = {}, fetcher = fetch) {
  const headers = { "Content-Type": "application/json" };
  if (options.token) headers.Authorization = `Bearer ${options.token}`;
  let response;
  try { response = await fetcher(`${getApiBaseUrl()}${path}`, { method: options.method || "GET", headers, ...(options.body === undefined ? {} : { body: JSON.stringify(options.body) }) }); }
  catch { throw new ApiError("The MediBot API is unavailable. Check that the backend is running.", 0); }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const fallback = response.status === 401 ? "Your session is invalid or has expired. Please sign in again." : response.status === 403 ? "Your role does not allow this request." : `The API returned an error (${response.status}).`;
    throw new ApiError(typeof data.detail === "string" ? data.detail : fallback, response.status);
  }
  return data;
}
