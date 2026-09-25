"use client";
import { FormEvent, useEffect, useState } from "react";
import { ApiError, apiRequest } from "@/lib/api.mjs";
const ROLES = ["doctor", "nurse", "billing_executive", "technician", "admin"] as const;
type Role = typeof ROLES[number];
type Session = { token: string; username: string; role: Role };
type Source = { label: string; source_document: string; section_title: string; collection: string };
type Result = { answer: string; retrieval_type: "document" | "sql"; sources: Source[]; role: Role };
const KEY = "medibot-demo-session";
const message = (error: unknown) => error instanceof ApiError ? error.message : "Unable to reach the MediBot API. Please try again.";

export default function Home() {
  const [session, setSession] = useState<Session | null>(null), [ready, setReady] = useState(false);
  const [health, setHealth] = useState("checking"), [username, setUsername] = useState(""), [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>("doctor"), [collections, setCollections] = useState<string[]>([]), [question, setQuestion] = useState("");
  const [result, setResult] = useState<Result | null>(null), [error, setError] = useState(""), [loading, setLoading] = useState("");
  useEffect(() => {
    try { const saved = sessionStorage.getItem(KEY); if (saved) setSession(JSON.parse(saved)); } catch { sessionStorage.removeItem(KEY); } finally { setReady(true); }
    apiRequest("/health").then(() => setHealth("online")).catch(() => setHealth("offline"));
  }, []);
  useEffect(() => {
    if (!session) return;
    setLoading("collections"); setError("");
    apiRequest(`/collections/${encodeURIComponent(session.role)}`, { token: session.token }).then((data: { collections: string[] }) => setCollections(data.collections))
      .catch((e: unknown) => { if (e instanceof ApiError && e.status === 401) logout(); setError(message(e)); }).finally(() => setLoading(""));
  }, [session]);
  function logout() { sessionStorage.removeItem(KEY); setSession(null); setCollections([]); setResult(null); setQuestion(""); setPassword(""); }
  async function login(e: FormEvent) {
    e.preventDefault(); setError("");
    if (!username.trim() || !password) return setError("Enter a username and password.");
    setLoading("login");
    try {
      const data = await apiRequest("/login", { method: "POST", body: { username: username.trim(), password, role } }) as { access_token: string };
      const next = { token: data.access_token, username: username.trim(), role }; sessionStorage.setItem(KEY, JSON.stringify(next)); setSession(next); setPassword("");
    } catch (e) { setError(message(e)); } finally { setLoading(""); }
  }
  async function chat(e: FormEvent) {
    e.preventDefault(); const value = question.trim(); setError("");
    if (!value) return setError("Enter a question before sending."); if (!session) return;
    setLoading("chat"); setResult(null);
    try {
      const data = await apiRequest("/chat", { method: "POST", token: session.token, body: { question: value } }) as Omit<Result, "retrieval_type"> & { retrieval_type: "hybrid_rag" | "sql_rag" };
      setResult({ ...data, retrieval_type: data.retrieval_type === "sql_rag" ? "sql" : "document" });
    }
    catch (e) { if (e instanceof ApiError && e.status === 401) logout(); setError(message(e)); } finally { setLoading(""); }
  }
  if (!ready) return <main className="center"><p role="status">Restoring session…</p></main>;
  return <main className="shell">
    <header className="topbar"><a className="brand" href="#main"><b aria-hidden="true">+</b> MediBot</a><div className={`health ${health}`} role="status"><i />API {health}</div></header>
    {!session ? <section className="login-card" id="main" aria-labelledby="login-title">
      <p className="eyebrow">Staff portal</p><h1 id="login-title">Welcome to MediBot</h1><p className="intro">Sign in with your demo staff account to search information available to your role.</p>
      <form onSubmit={login} noValidate><label htmlFor="username">Username</label><input id="username" autoComplete="username" value={username} onChange={e => setUsername(e.target.value)} />
        <label htmlFor="password">Password</label><input id="password" type="password" autoComplete="current-password" value={password} onChange={e => setPassword(e.target.value)} />
        <label htmlFor="role">Role</label><select id="role" value={role} onChange={e => setRole(e.target.value as Role)}>{ROLES.map(r => <option key={r} value={r}>{r.replace("_", " ")}</option>)}</select>
        {error && <div className="alert" role="alert">{error}</div>}<button disabled={loading === "login"}>{loading === "login" ? "Signing in…" : "Sign in"}</button></form>
      <p className="privacy">Your token is kept only in this browser tab&apos;s session storage.</p>
    </section> : <div className="workspace" id="main">
      <aside className="sidebar" aria-label="Current access"><div><p className="eyebrow">Signed in as</p><h2>{session.username}</h2><span className="role-chip">{session.role.replace("_", " ")}</span></div>
        <div><h3>Accessible collections</h3>{loading === "collections" ? <p role="status">Loading access…</p> : <ul>{collections.map(c => <li key={c}>{c}</li>)}</ul>}</div><button className="secondary" onClick={logout}>Log out</button></aside>
      <section className="chat-panel" aria-labelledby="chat-title"><div className="chat-heading"><p className="eyebrow">Knowledge assistant</p><h1 id="chat-title">Ask MediBot</h1></div>
        <div className="answer-area" aria-live="polite">{!result && loading !== "chat" && !error && <div className="empty"><span>✦</span><h2>How can I help?</h2><p>Ask about documents, policies, equipment, or data available to your role.</p></div>}
          {loading === "chat" && <div className="loading" role="status"><i />Searching approved sources…</div>}
          {error && <div className="alert" role="alert"><strong>{error.includes("only") || error.includes("role") ? "Access refused" : "Something went wrong"}</strong><br />{error}</div>}
          {result && <article className="response"><span className={`type-label ${result.retrieval_type}`}>{result.retrieval_type === "sql" ? "SQL retrieval" : "Document retrieval"}</span><h2>Answer</h2><p className="answer">{result.answer}</p>
            {result.retrieval_type === "document" && result.sources.length > 0 && <section className="sources"><h3>Sources</h3><div className="source-grid">{result.sources.map((s, i) => <article className="source-card" key={`${s.label}-${i}`}><strong>{s.label}</strong><dl><div><dt>Document</dt><dd>{s.source_document}</dd></div><div><dt>Section</dt><dd>{s.section_title}</dd></div><div><dt>Collection</dt><dd>{s.collection}</dd></div></dl></article>)}</div></section>}
          </article>}</div>
        <form className="composer" onSubmit={chat}><label className="sr-only" htmlFor="question">Question</label><textarea id="question" rows={2} value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question…" disabled={loading === "chat"} /><button disabled={loading === "chat"}>Send <span aria-hidden="true">→</span></button></form>
      </section>
    </div>}
  </main>;
}
