// mini-infer site: every number comes from data/results.json (bench/export_site_data.py).
"use strict";

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (x, d = 0) => Number(x).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
const secs = (ms) => (ms >= 1000 ? `${fmt(ms / 1000, 1)} s` : `${fmt(ms, 0)} ms`);
const el = (id) => document.getElementById(id);

let DATA = null;
const charts = [];

function table(headers, rows) {
  const head = `<tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr>`;
  const body = rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("");
  return `<table>${head}${body}</table>`;
}

// ---------------------------------------------------------------- text + tables
function fillText(d) {
  const users = d.users.filter((u) => u.users > 1);
  const best = users.reduce((a, u) => (u.engine.throughput_tok_s / u.hf.throughput_tok_s > a.x
    ? { x: u.engine.throughput_tok_s / u.hf.throughput_tok_s, u: u.users } : a), { x: 0, u: 0 });
  const u32 = d.users.find((u) => u.users === 32);
  const ttftX = u32.hf.ttft_ms_p50 / u32.engine.ttft_ms_p50;
  const mem = d.memory;
  const capX = mem.max_concurrent.paged / mem.max_concurrent["reserve max context"];

  document.querySelector(".hero h1 .accent").textContent = `${fmt(best.x, 1)}× faster`;
  el("hero-stats").innerHTML = [
    [`${fmt(best.x, 2)}×`, `throughput vs Hugging Face at ${best.u} users`],
    [`${fmt(ttftX, 0)}×`, `faster first token at 32 users (${secs(u32.hf.ttft_ms_p50)} → ${secs(u32.engine.ttft_ms_p50)})`],
    [`${fmt(mem.waste_pct.paged, 1)}%`, `KV memory wasted, vs ${fmt(mem.waste_pct["reserve max context"], 1)}% reserving max length`],
    [`${fmt(capX, 1)}×`, `more requests fit in the same ${mem.budget_gb} GB of KV cache`],
  ].map(([v, l]) => `<div class="stat"><div class="value">${v}</div><div class="label">${l}</div></div>`).join("");

  const hw = d.hardware;
  const hwLine = `${hw.model} on one ${hw.gpu} · ${hw.workload} · ${hw.method}.`;
  el("hero-note").textContent = hwLine;
  el("hw-note").textContent = `${hwLine} A laptop RTX 3050 gave ${d.laptop.speedup}× at 32 users, so the result holds on a second GPU.`;

  el("t-throughput").innerHTML = table(["users", "Hugging Face", "mini-infer", "speedup"], d.users.map((u) =>
    [u.users, `${fmt(u.hf.throughput_tok_s, 1)} tok/s`, `${fmt(u.engine.throughput_tok_s, 1)} tok/s`,
      `<span class="good">${fmt(u.engine.throughput_tok_s / u.hf.throughput_tok_s, 2)}×</span>`]));
  el("t-ttft-users").innerHTML = table(["users", "HF p50", "mini-infer p50", "HF p99", "mini-infer p99"], d.users.map((u) =>
    [u.users, secs(u.hf.ttft_ms_p50), secs(u.engine.ttft_ms_p50), secs(u.hf.ttft_ms_p99), secs(u.engine.ttft_ms_p99)]));
  el("t-rate").innerHTML = table(["requests/s", "HF p50", "mini-infer p50", "HF p99", "mini-infer p99"], d.rates.map((r) =>
    [r.rate, secs(r.hf.ttft_ms_p50), secs(r.engine.ttft_ms_p50), secs(r.hf.ttft_ms_p99), secs(r.engine.ttft_ms_p99)]));

  const kv = d.kv_cache;
  el("n-kv").textContent = `With the cache every token costs about the same; without it each step re-reads the whole sequence. `
    + `${fmt(kv.cached_tok_s, 1)} vs ${fmt(kv.uncached_tok_s, 1)} tok/s: ${fmt(kv.speedup, 2)}× faster for a ${kv.prompt_len}-token prompt.`;
  el("cap-sub").textContent = `concurrent sequences in ${mem.budget_gb} GB of KV cache at ${fmt(mem.bytes_per_token / 1024, 0)} KiB per token`;
  el("n-capacity").textContent = `Median over 200 random mixes of real request lengths. Paging wastes at most one partly filled block per request.`;

  const [fp, i8] = d.quant;
  el("t-quant").innerHTML = table(["", "FP16", "INT8"], [
    ["weights", `${fmt(fp.weights_gb, 2)} GB`, `<span class="good">${fmt(i8.weights_gb, 2)} GB</span>`],
    ["WikiText-2 perplexity", fmt(fp.ppl, 3), `<span class="good">${fmt(i8.ppl, 3)}</span>`],
    ["1 sequence", `${fmt(fp.single_tok_s, 1)} tok/s`, `${fmt(i8.single_tok_s, 1)} tok/s`],
    ["32 sequences", `${fmt(fp.batched_tok_s, 0)} tok/s`, `${fmt(i8.batched_tok_s, 0)} tok/s`],
  ]);
  const sp = d.speculative;
  el("t-spec").innerHTML = table(["draft length k", "accepted", "tokens / target pass", "speed vs 1.5B alone"],
    sp.runs.map((r) => [r.k, `${fmt(r.acceptance_rate * 100, 0)}%`, fmt(r.tokens_per_round, 2), `${fmt(r.speedup, 2)}×`]));

  const ab = d.ablation;
  el("n-ablation").textContent = `Our forward pass alone matches Hugging Face's sequential speed (${fmt(ab[1].tok_s, 1)} vs ${fmt(ab[0].tok_s, 1)} tok/s). `
    + `Static batching gives ${fmt(ab[2].tok_s / ab[0].tok_s, 1)}×; paging plus continuous batching gives ${fmt(ab[3].tok_s / ab[0].tok_s, 1)}×.`;
}

// ---------------------------------------------------------------- charts
function baseOptions(yTitle, { logY = false, xTitle = "", logX = false } = {}) {
  const ink2 = css("--ink-2"), muted = css("--muted"), grid = css("--grid");
  return {
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { position: "top", align: "start", labels: { color: ink2, boxWidth: 12, boxHeight: 12, usePointStyle: true } },
      tooltip: { backgroundColor: css("--surface"), titleColor: css("--ink"), bodyColor: ink2, borderColor: grid, borderWidth: 1, padding: 10 },
    },
    scales: {
      x: { type: logX ? "logarithmic" : "category", title: { display: !!xTitle, text: xTitle, color: ink2 },
        grid: { display: false }, ticks: { color: muted }, border: { color: css("--axis") } },
      y: { type: logY ? "logarithmic" : "linear", beginAtZero: !logY, title: { display: true, text: yTitle, color: ink2 },
        grid: { color: grid }, ticks: { color: muted }, border: { display: false } },
    },
  };
}

function lineDataset(label, data, color) {
  return { label, data, borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 5,
    pointHoverRadius: 7, pointBorderColor: css("--surface"), pointBorderWidth: 2, tension: 0 };
}

function drawCharts(d) {
  charts.splice(0).forEach((c) => c.destroy());
  const ENG = css("--engine"), HF = css("--hf");
  const users = d.users.map((u) => u.users);

  const tp = baseOptions("tokens / second", { xTitle: "concurrent users" });
  tp.plugins.tooltip.callbacks = {
    label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y, 1)} tok/s`,
    footer: (items) => (items.length === 2 ? `speedup ${fmt(items[1].parsed.y / items[0].parsed.y, 2)}×` : ""),
  };
  charts.push(new Chart(el("c-throughput"), { type: "line", options: tp, data: { labels: users, datasets: [
    lineDataset("Hugging Face (best batch size)", d.users.map((u) => u.hf.throughput_tok_s), HF),
    lineDataset("mini-infer", d.users.map((u) => u.engine.throughput_tok_s), ENG)] } }));

  const tt = baseOptions("seconds (log)", { logY: true, xTitle: "concurrent users" });
  tt.plugins.tooltip.callbacks = { label: (c) => `${c.dataset.label}: ${secs(c.parsed.y * 1000)}` };
  charts.push(new Chart(el("c-ttft-users"), { type: "line", options: tt, data: { labels: users, datasets: [
    lineDataset("Hugging Face", d.users.map((u) => u.hf.ttft_ms_p50 / 1000), HF),
    lineDataset("mini-infer", d.users.map((u) => u.engine.ttft_ms_p50 / 1000), ENG)] } }));

  const rt = baseOptions("seconds (log)", { logY: true, xTitle: "arrival rate (requests / second)" });
  rt.plugins.tooltip.callbacks = { label: (c) => `${c.dataset.label}: ${secs(c.parsed.y * 1000)}` };
  charts.push(new Chart(el("c-rate"), { type: "line", options: rt, data: { labels: d.rates.map((r) => r.rate), datasets: [
    lineDataset("Hugging Face p99", d.rates.map((r) => r.hf.ttft_ms_p99 / 1000), HF),
    lineDataset("mini-infer p99", d.rates.map((r) => r.engine.ttft_ms_p99 / 1000), ENG)] } }));

  const kv = d.kv_cache;
  const steps = kv.cached_ms.slice(1).map((_, i) => i + 2);  // skip step 1 (prefill)
  const ko = baseOptions("ms per token", { xTitle: "generated token #" });
  ko.scales.x.ticks.maxTicksLimit = 9;
  ko.plugins.tooltip.callbacks = { title: (i) => `token ${i[0].label}`, label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y, 1)} ms` };
  const thin = (data, color, label) => ({ ...lineDataset(label, data, color), pointRadius: 0, pointHoverRadius: 4, borderWidth: 1.5 });
  charts.push(new Chart(el("c-kv"), { type: "line", options: ko, data: { labels: steps, datasets: [
    thin(kv.uncached_ms.slice(1), HF, "no cache (recompute)"), thin(kv.cached_ms.slice(1), ENG, "KV cache")] } }));

  const strategies = ["reserve max context", "pad to batch max", "paged"];
  const names = ["reserve max length", "pad to batch max (HF)", "paged, 16-token blocks"];
  const colors = [css("--muted"), HF, ENG];
  const bar = (yTitle, values, suffix, digits) => {
    const o = baseOptions(yTitle);
    o.indexAxis = "y";
    o.interaction = { mode: "nearest", intersect: true };
    o.plugins.legend.display = false;
    o.plugins.tooltip.callbacks = { label: (c) => `${fmt(c.parsed.x, digits)}${suffix}` };
    o.scales = { x: { beginAtZero: true, suggestedMax: Math.max(...values) * 1.22, grid: { color: css("--grid") },
      ticks: { color: css("--muted") }, border: { display: false }, title: { display: true, text: yTitle, color: css("--ink-2") } },
      y: { grid: { display: false }, ticks: { color: css("--ink-2") }, border: { color: css("--axis") } } };
    return { type: "bar", options: o, data: { labels: names, datasets: [{ data: values, backgroundColor: colors,
      borderRadius: 4, borderSkipped: "start", barPercentage: 0.6, borderColor: css("--surface"), borderWidth: 2 }] },
      plugins: [valueLabels(suffix, digits)] };
  };
  charts.push(new Chart(el("c-waste"), bar("% of reserved KV memory", strategies.map((s) => d.memory.waste_pct[s]), "%", 1)));
  charts.push(new Chart(el("c-capacity"), bar("sequences", strategies.map((s) => d.memory.max_concurrent[s]), "", 0)));

  const ab = d.ablation;
  const ao = baseOptions("tokens / second");
  ao.indexAxis = "y";
  ao.interaction = { mode: "nearest", intersect: true };
  ao.plugins.legend.display = false;
  ao.plugins.tooltip.callbacks = { label: (c) => `${fmt(c.parsed.x, 1)} tok/s (${fmt(c.parsed.x / ab[0].tok_s, 1)}× sequential)` };
  ao.scales = { x: { beginAtZero: true, suggestedMax: Math.max(...ab.map((a) => a.tok_s)) * 1.2, grid: { color: css("--grid") },
    ticks: { color: css("--muted") }, border: { display: false },
    title: { display: true, text: "tokens / second at 32 users", color: css("--ink-2") } },
    y: { grid: { display: false }, ticks: { color: css("--ink-2") }, border: { color: css("--axis") } } };
  charts.push(new Chart(el("c-ablation"), { type: "bar", options: ao, data: { labels: ab.map((a) => a.label), datasets: [{
    data: ab.map((a) => a.tok_s), backgroundColor: ab.map((a) => (a.system === "engine" ? ENG : HF)),
    borderRadius: 4, borderSkipped: "start", barPercentage: 0.6, borderColor: css("--surface"), borderWidth: 2 }] },
    plugins: [valueLabels(" tok/s", 0)] }));
}

// Direct value labels at the end of horizontal bars (text in ink, never the series color).
function valueLabels(suffix, digits) {
  return {
    id: "valueLabels",
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      ctx.save();
      ctx.fillStyle = css("--ink-2");
      ctx.font = "500 12px Inter, sans-serif";
      ctx.textBaseline = "middle";
      chart.getDatasetMeta(0).data.forEach((b, i) => {
        ctx.fillText(`${fmt(chart.data.datasets[0].data[i], digits)}${suffix}`, b.x + 6, b.y);
      });
      ctx.restore();
    },
  };
}

// ---------------------------------------------------------------- paged attention animation
// Up to 5 running sequences (like the engine's max_num_seqs), so each gets its own color.
const PG = { blocks: 32, blockSize: 4, maxLen: 32, maxSeqs: 5, tickMs: 650 };
const SEQ_COLORS = ["--seq-1", "--seq-2", "--seq-3", "--seq-4", "--seq-5"];

function makeRng(seed) {
  let s = seed;
  return () => ((s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296);
}

function paging() {
  let rng, state, timer = null, running = true;

  function reset() {
    rng = makeRng(7);
    state = { owner: new Array(PG.blocks).fill(null), free: [...Array(PG.blocks).keys()], seqs: [], waiting: [],
      nextId: 1, events: [] };
    render();
  }

  const need = (tokens) => Math.ceil(tokens / PG.blockSize);

  function spawn() {
    const prompt = 2 + Math.floor(rng() * 7);
    const target = prompt + 4 + Math.floor(rng() * 20);
    state.waiting.push({ id: state.nextId++, tokens: prompt, target, table: [], color: null });
  }

  function grow(seq, tokens) {
    while (seq.table.length < need(tokens)) {
      if (!state.free.length) return false;
      state.free.sort((a, b) => a - b);
      const b = state.free.shift();
      state.owner[b] = seq.id;
      seq.table.push(b);
    }
    seq.tokens = tokens;
    return true;
  }

  function release(seq) {
    seq.table.forEach((b) => { state.owner[b] = null; state.free.push(b); });
    seq.table = [];
  }

  function step() {
    state.events = [];
    // 1. every running sequence gets one more token; preempt the newest if the pool is full
    for (const seq of [...state.seqs]) {
      if (!state.seqs.includes(seq)) continue;
      while (!grow(seq, seq.tokens + 1)) {
        const victim = state.seqs[state.seqs.length - 1];
        release(victim);
        state.seqs.pop();
        state.waiting.unshift(victim);  // keeps its tokens: recomputed when re-admitted
        state.events.push(`S${victim.id} preempted`);
        if (victim === seq) break;
      }
    }
    // 2. retire finished sequences
    for (const seq of state.seqs.filter((s) => s.tokens >= s.target)) {
      release(seq);
      state.events.push(`S${seq.id} finished, ${need(seq.target)} blocks freed`);
    }
    state.seqs = state.seqs.filter((s) => s.tokens < s.target);
    // 3. arrivals, then admit while blocks are free
    if (rng() < 0.55) spawn();
    while (state.waiting.length && state.seqs.length < PG.maxSeqs && state.free.length >= need(state.waiting[0].tokens) + 1) {
      const seq = state.waiting.shift();
      const inUse = new Set(state.seqs.map((s) => s.color));
      seq.color = SEQ_COLORS.find((c) => !inUse.has(c));  // color follows the sequence while it runs
      grow(seq, seq.tokens);
      state.seqs.push(seq);
      state.events.push(`S${seq.id} admitted`);
    }
    render();
  }

  function render() {
    const byId = Object.fromEntries(state.seqs.map((s) => [s.id, s]));
    el("pg-grid").innerHTML = state.owner.map((id, b) => {
      if (id === null) return `<div class="pg-block free" title="block ${b}: free"><span class="id">${b}</span></div>`;
      const s = byId[id];
      const pos = s.table.indexOf(b);
      const used = Math.min(PG.blockSize, s.tokens - pos * PG.blockSize);
      const empty = 100 * (1 - used / PG.blockSize);
      return `<div class="pg-block" style="background:${css(s.color)}" title="block ${b}: S${id}, ${used}/${PG.blockSize} slots used">`
        + `<div class="fill" style="height:${empty}%"></div><span class="id">S${id}</span></div>`;
    }).join("");

    el("pg-tables").innerHTML = state.seqs.length
      ? state.seqs.map((s) => `<div><b style="color:${css(s.color)}">S${s.id}</b> ${s.tokens}/${s.target} tok → [${s.table.join(", ")}]</div>`).join("")
      : `<div>no running sequences</div>`;

    const used = state.seqs.reduce((a, s) => a + s.tokens, 0);
    const paged = state.seqs.reduce((a, s) => a + s.table.length, 0) * PG.blockSize;
    const contiguous = state.seqs.length * PG.maxLen;
    const cap = PG.blocks * PG.blockSize;
    const barHtml = (label, reserved) => {
      const u = Math.min(used, cap), w = Math.max(0, Math.min(reserved, cap) - u);
      const over = reserved > cap ? ` · needs ${reserved} slots, pool has ${cap}` : "";
      return `<div class="pg-bar">${label}: ${used} of ${reserved} reserved slots used${over}`
        + `<div class="track"><div class="used" style="width:${(100 * u) / cap}%"></div>`
        + `<div class="wasted" style="width:${(100 * w) / cap}%"></div></div></div>`;
    };
    el("pg-stats").innerHTML = barHtml("Paged", paged) + barHtml(`Reserve max length (${PG.maxLen} tokens each)`, contiguous)
      + `<div class="muted">${state.events.join(" · ") || "&nbsp;"}${state.waiting.length ? ` · ${state.waiting.length} waiting` : ""}</div>`;
  }

  function play() { timer = setInterval(step, PG.tickMs); running = true; el("pg-play").textContent = "Pause"; }
  function pause() { clearInterval(timer); running = false; el("pg-play").textContent = "Play"; }

  el("pg-legend").innerHTML = `<span><i style="background:var(--free)"></i>free block</span>`
    + `<span><i style="background:var(--seq-1)"></i>block owned by a sequence (pale top = unused slots)</span>`
    + `<span><i style="background:var(--engine)"></i>slots used</span>`
    + `<span><i style="background:var(--waste)"></i>reserved but wasted</span>`;
  el("pg-play").onclick = () => (running ? pause() : play());
  el("pg-step").onclick = () => { pause(); step(); };
  el("pg-reset").onclick = () => { reset(); };
  reset();
  for (let i = 0; i < 4; i++) spawn();
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { running = false; el("pg-play").textContent = "Play"; }
  else play();
  return { render };
}

// ---------------------------------------------------------------- boot
document.querySelectorAll("[data-copy]").forEach((b) => {
  b.onclick = async () => {
    try { await navigator.clipboard.writeText(el(b.dataset.copy).textContent); b.textContent = "Copied"; }
    catch { b.textContent = "Select and copy"; }
    setTimeout(() => (b.textContent = "Copy"), 1500);
  };
});

const pagingView = paging();

fetch("data/results.json")
  .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
  .then((d) => {
    DATA = d;
    fillText(d);
    if (window.Chart) drawCharts(d);
  })
  .catch((e) => {
    el("hero-stats").innerHTML = `<p class="muted">Could not load data/results.json (${e}). Serve this folder over HTTP.</p>`;
  });

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (DATA && window.Chart) drawCharts(DATA);
  pagingView.render();
});
