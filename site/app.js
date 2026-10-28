// Neuron viewer logic (vanilla JS, no build step). Loads data/index.json, renders the pole list with filters,
// and shows one pole's detail page: label, stats, promoted tokens, top firings as token heatmaps, held-out
// detection examples, the steering dose-response with sampled generations, confirmation and mechanism results.
// Deep links: ?key=neuron:L31:N2047:+  or  ?layer=31&neuron=2047&sign=1
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
let INDEX = null;

// Token heatmap: background alpha = signed activation / signed peak, colour by pole.
function heat(ex, sign) {
  const acts = ex.acts.map((a) => a * sign);
  const peak = Math.max(1e-9, acts[ex.peak]);
  const rgb = sign > 0 ? "var(--pos)" : "var(--neg)";
  return ex.tokens.map((t, i) => {
    const a = Math.max(0, Math.min(1, acts[i] / peak));
    const cls = i === ex.peak ? "tok peak" : "tok";
    return `<span class="${cls}" title="${fmt(ex.acts[i], 3)}" style="background: rgba(${rgb}, ${a.toFixed(2)})">${esc(t)}</span>`;
  }).join("");
}

function summaryCards(idx) {
  const cards = [];
  if (idx.h1) {
    const n = idx.h1.neuron, d = idx.h1.direction, diff = idx.h1.difference;
    cards.push(`<div class="card"><h3>H1 · clean poles (held-out detection ≥ 0.8)</h3>
      <div class="big">${fmt(100 * n.clean_rate, 0)}% <span class="muted" style="font-size:1rem">neurons</span> vs ${fmt(100 * d.clean_rate, 0)}% <span class="muted" style="font-size:1rem">random directions</span></div>
      <div class="muted">Δ = ${fmt(100 * diff.clean_rate_diff, 1)} pts, 95% CI [${fmt(100 * diff.ci95[0], 1)}, ${fmt(100 * diff.ci95[1], 1)}], Fisher p = ${fmt(diff.fisher_one_sided_p, 4)}</div></div>`);
  }
  for (const [key, r] of Object.entries(idx.h2 || {})) {
    const c = r.conditions;
    cards.push(`<div class="card"><h3>H2 · steering <a href="?key=${encodeURIComponent(key)}">${esc(key)}</a></h3>
      <div>${esc(r.label)}</div>
      <div class="big">${fmt(c.clamp.hm)} <span class="muted" style="font-size:1rem">clamped vs ${fmt(c.unsteered.hm)} unsteered</span></div>
      <div class="muted">harmonic mean of concept and fluency (0–2) · p vs random neurons (Holm) = ${fmt(r.p_holm, 3)} · ${r.h2_supported ? '<span class="ok">supported</span>' : '<span class="no">not supported</span>'}</div></div>`);
  }
  if (idx.split_stability) {
    cards.push(`<div class="card"><h3>Statistic stability across splits</h3><div class="big">ρ = ${fmt(idx.split_stability.max)}</div>
      <div class="muted">Spearman correlation of each neuron's max activation, discovery vs held-out documents</div></div>`);
  }
  $("summary").innerHTML = cards.join("");
}

function renderList() {
  const q = $("search").value.trim().toLowerCase();
  const group = $("group").value;
  const minba = parseFloat($("minba").value);
  $("minba-v").textContent = minba.toFixed(2);
  const lm = q.match(/^l(\d+)\s*n(\d+)$/);
  const rows = INDEX.poles.filter((p) => {
    if (group === "steer" && !p.steering) return false;
    if (group === "candidate" && p.group !== "candidate") return false;
    if (group === "audit" && p.group !== "audit") return false;
    if (p.ba < minba) return false;
    if (lm) return p.layer === +lm[1] && p.index === +lm[2];
    return !q || p.label.toLowerCase().includes(q) || p.key.toLowerCase().includes(q);
  }).sort((a, b) => (b.confirmed - a.confirmed) || (b.ba - a.ba) || (a.layer - b.layer));
  $("list").innerHTML = rows.map((p) => `<li data-key="${esc(p.key)}"><div>${esc(p.label || "(no label)")}</div>
    <div class="k">${esc(p.key)} <span class="pill">acc ${fmt(p.ba)}</span>${p.confirmed ? '<span class="pill">confirmed</span>' : ""}</div></li>`).join("")
    || '<li class="muted">No poles match.</li>';
  for (const li of $("list").querySelectorAll("li[data-key]")) li.onclick = () => select(li.dataset.key, true);
}

function steeringBlock(d) {
  if (!d.screen) return "";
  const levels = Object.keys(d.screen.levels).map(Number).sort((a, b) => a - b);
  const rows = levels.map((c) => {
    const s = d.screen.levels[c];
    return `<tr><td>${c === 0 ? "unsteered" : c + "× max"}</td><td>${fmt(s.concept)}</td><td>${fmt(s.fluency)}</td><td>${fmt(s.hm)} <span class="bar" style="width:${40 * s.hm}px"></span></td><td>${fmt(s.nll)}</td></tr>`;
  }).join("");
  return `<h4>Turn it up: clamp sweep on 10 neutral prompts (exploratory screen)</h4>
    <table><tr><th>clamp</th><th>concept</th><th>fluency</th><th>harmonic mean</th><th>NLL</th></tr>${rows}</table>
    <div class="levels" id="levels">${levels.map((c) => `<button data-c="${c}">${c === 0 ? "off" : c + "×"}</button>`).join("")}</div>
    <div id="gens"></div>`;
}

function showGens(d, c) {
  for (const b of document.querySelectorAll("#levels button")) b.classList.toggle("on", +b.dataset.c === c);
  const gens = d.screen.generations.filter((g) => g.level === c);
  $("gens").innerHTML = gens.map((g) => `<div class="gen"><span class="prompt">${esc(g.prompt)}</span><b>${esc(g.text)}</b>
    <span class="muted"> · concept ${g.concept} · fluency ${g.fluency}</span></div>`).join("");
}

function confirmBlock(d) {
  if (!d.confirm) return "";
  const c = d.confirm.conditions;
  const order = ["unsteered", "clamp", "add", "diffmean", "prompting"];
  const null_ = d.confirm.random_null_hm;
  const rows = order.filter((k) => c[k]).map((k) => `<tr><td>${k}</td><td>${fmt(c[k].hm)} [${fmt(c[k].hm_ci95[0])}, ${fmt(c[k].hm_ci95[1])}]</td><td>${fmt(c[k].concept)}</td><td>${fmt(c[k].fluency)}</td><td>${fmt(c[k].nll)}</td><td>${fmt(c[k].distinct2)}</td></tr>`).join("");
  return `<h4>Pre-registered confirmation (50 held-out prompts × 2 seeds)</h4>
    <table><tr><th>condition</th><th>harmonic mean [95% CI]</th><th>concept</th><th>fluency</th><th>NLL</th><th>distinct-2</th></tr>${rows}
    <tr><td>20 random neurons, same clamp</td><td>mean ${fmt(null_.reduce((a, b) => a + b, 0) / null_.length)} · max ${fmt(Math.max(...null_))}</td><td colspan="4" class="muted">null distribution</td></tr></table>
    <p>Clamp − unsteered = ${fmt(d.confirm.clamp_minus_unsteered.diff)} [${fmt(d.confirm.clamp_minus_unsteered.ci95[0])}, ${fmt(d.confirm.clamp_minus_unsteered.ci95[1])}],
      empirical p vs random = ${fmt(d.confirm.p_vs_random_null, 3)}, Holm p = ${fmt(d.confirm.p_holm, 3)} →
      ${d.confirm.h2_supported ? '<b class="ok">H2 supported</b>' : '<b class="no">H2 not supported</b>'}</p>`;
}

function mechanismBlock(d) {
  if (!d.mechanism) return "";
  const m = d.mechanism, t = m.context_truncation_ratio;
  return `<h4>Mechanism</h4><table>
    <tr><td>Zero-ablation: drop in log p(next token) at ${m.n_positions} top firings</td><td>${fmt(m.h3.ablation_drop_nats, 4)} nats (random neurons max ${fmt(Math.max(...m.h3.random_null), 4)}, p = ${fmt(m.h3.p_empirical, 3)})</td></tr>
    <tr><td>Gate z &lt; 0 at firings / double-negative (z&lt;0, y&lt;0)</td><td>${fmt(100 * m.factor_split.frac_gate_negative, 0)}% / ${fmt(100 * m.factor_split.frac_double_negative, 0)}%</td></tr>
    <tr><td>Activation kept with only the last k tokens of context</td><td>${Object.entries(t).map(([k, v]) => `k=${k}: ${fmt(v)}`).join(" · ")}</td></tr>
    <tr><td>Patching its firing value into random contexts: Δ mean log p(promoted tokens)</td><td>${fmt(m.patching.delta_logp_promoted, 3)} (p vs random neurons = ${fmt(m.patching.p_empirical, 3)})</td></tr></table>`;
}

async function select(key, push) {
  for (const li of document.querySelectorAll("#list li")) li.classList.toggle("on", li.dataset.key === key);
  const row = INDEX.poles.find((p) => p.key === key);
  if (!row) return;
  if (push) history.replaceState(null, "", "?key=" + encodeURIComponent(key));
  const d = await (await fetch(`data/poles/${row.file}.json`)).json();
  const sign = d.sign;
  const stats = d.stats ? `<table><tr>${Object.keys(d.stats).map((k) => `<th>${k}</th>`).join("")}</tr><tr>${Object.values(d.stats).map((v) => `<td>${fmt(v, 3)}</td>`).join("")}</tr></table>` : "";
  const vocab = d.vocab ? `<div class="grid2"><div><h4>Writes up (promoted tokens)</h4><div class="chips">${d.vocab.promoted.map(([t]) => `<span>${esc(t)}</span>`).join("")}</div></div>
    <div><h4>Writes down (suppressed tokens)</h4><div class="chips">${d.vocab.suppressed.map(([t]) => `<span>${esc(t)}</span>`).join("")}</div></div></div>` : "";
  const det = d.examples.detect_top.concat(d.examples.detect_random);
  $("detail").innerHTML = `<h2>${esc(d.label || "(no label)")}</h2>
    <div class="muted">${esc(d.key)} · ${d.kind === "neuron" ? `layer ${d.layer}, neuron ${d.index}` : `layer ${d.layer}, random direction ${d.index}`} · ${sign > 0 ? "positive" : "negative"} pole · held-out detection accuracy <b>${fmt(d.ba)}</b></div>
    ${stats}${vocab}
    ${steeringBlock(d)}${confirmBlock(d)}${mechanismBlock(d)}
    <h4>Strongest firings (discovery documents; colour = activation / peak, outlined = peak token)</h4>
    ${d.examples.label.map((ex) => `<div class="snip">${heat(ex, sign)}</div>`).join("")}
    <h4>Held-out detection test: top firings (first ${d.examples.detect_top.length}) and random snippets</h4>
    ${det.map((ex, i) => `<div class="snip">${i < d.examples.detect_top.length ? "★ " : "· "}${heat(ex, sign)}</div>`).join("")}`;
  if (d.screen) {
    for (const b of document.querySelectorAll("#levels button")) b.onclick = () => showGens(d, +b.dataset.c);
    showGens(d, d.screen.best_level);
  }
}

async function main() {
  INDEX = await (await fetch("data/index.json")).json();
  $("model-name").textContent = INDEX.model;
  summaryCards(INDEX);
  for (const id of ["search", "group", "minba"]) $(id).addEventListener("input", renderList);
  const params = new URLSearchParams(location.search);
  let key = params.get("key");
  if (!key && params.get("layer")) key = `neuron:L${params.get("layer")}:N${params.get("neuron")}:${params.get("sign") === "-1" ? "-" : "+"}`;
  if (key && !INDEX.poles.find((p) => p.key === key)) key = null;
  if (!INDEX.poles.some((p) => p.steering)) $("group").value = "candidate";
  renderList();
  const first = key || (INDEX.poles.find((p) => p.confirmed) || INDEX.poles.find((p) => p.steering) || INDEX.poles[0] || {}).key;
  if (first) select(first, false);
}
main();
