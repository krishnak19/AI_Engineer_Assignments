import assert from "node:assert/strict";
import test from "node:test";
import { ApiError, apiRequest } from "./api.mjs";
test("adds bearer auth and a JSON body", async () => {
  let request;
  const fetcher = async (url, options) => { request = { url, options }; return new Response(JSON.stringify({ answer: "ok" }), { status: 200 }); };
  await apiRequest("/chat", { method: "POST", token: "demo-token", body: { question: "Hello" } }, fetcher);
  assert.equal(request.url, "http://localhost:8000/chat");
  assert.equal(request.options.headers.Authorization, "Bearer demo-token");
  assert.equal(request.options.body, JSON.stringify({ question: "Hello" }));
});
test("preserves API detail and status for an RBAC refusal", async () => {
  const fetcher = async () => new Response(JSON.stringify({ detail: "Role is not allowed." }), { status: 403 });
  await assert.rejects(() => apiRequest("/chat", {}, fetcher), (error) => { assert.ok(error instanceof ApiError); assert.equal(error.status, 403); assert.equal(error.message, "Role is not allowed."); return true; });
});
test("turns network failures into a readable API error", async () => {
  await assert.rejects(() => apiRequest("/health", {}, async () => { throw new Error("offline"); }), /API is unavailable/);
});
