// Blog post chart: throughput vs users, drawn from the same data/results.json as the project page.
"use strict";

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (x, d = 0) => Number(x).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
let chart = null;
let DATA = null;

// Speedup labels above the engine's points (text in ink, never the series color).
const speedupLabels = {
  id: "speedupLabels",
  afterDatasetsDraw(c) {
    const hf = c.data.datasets[0].data, eng = c.data.datasets[1].data;
    const { ctx } = c;
    ctx.save();
    ctx.fillStyle = css("--ink");
    ctx.font = "500 12px Inter, sans-serif";
    ctx.textAlign = "center";
    c.getDatasetMeta(1).data.forEach((pt, i) => {
      if (i === 0) return;  // one user: nothing to batch
      ctx.fillText(`${fmt(eng[i] / hf[i], 2)}×`, pt.x, pt.y - 12);
    });
    ctx.restore();
  },
};

function draw(d) {
  if (chart) chart.destroy();
  const ink2 = css("--ink-2"), muted = css("--muted"), grid = css("--grid");
  const point = (color) => ({ borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 5, pointHoverRadius: 7,
    pointBorderColor: css("--surface"), pointBorderWidth: 2, tension: 0 });
  chart = new Chart(document.getElementById("c-post-throughput"), {
    type: "line",
    data: {
      labels: d.users.map((u) => u.users),
      datasets: [
        { label: "Hugging Face (best batch size)", data: d.users.map((u) => u.hf.throughput_tok_s), ...point(css("--muted")) },
        { label: "mini-infer", data: d.users.map((u) => u.engine.throughput_tok_s), ...point(css("--engine")) },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      layout: { padding: { top: 18, right: 28 } },  // room for the last speedup label
      plugins: {
        legend: { position: "top", align: "start", labels: { color: ink2, boxWidth: 12, boxHeight: 12, usePointStyle: true } },
        tooltip: {
          backgroundColor: css("--surface"), titleColor: css("--ink"), bodyColor: ink2, borderColor: grid, borderWidth: 1, padding: 10,
          callbacks: {
            title: (items) => `${items[0].label} concurrent users`,
            label: (c) => `${c.dataset.label}: ${fmt(c.parsed.y, 1)} tok/s`,
          },
        },
      },
      scales: {
        x: { title: { display: true, text: "concurrent users", color: ink2 }, grid: { display: false }, ticks: { color: muted },
          border: { color: css("--axis") } },
        y: { beginAtZero: true, suggestedMax: 450, title: { display: true, text: "tokens / second", color: ink2 },
          grid: { color: grid }, ticks: { color: muted }, border: { display: false } },
      },
    },
    plugins: [speedupLabels],
  });
}

fetch("../data/results.json")
  .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
  .then((d) => { DATA = d; if (window.Chart) draw(d); })
  .catch(() => { document.querySelector(".post-figure .chart-box").innerHTML = "<p class='note'>Chart data could not be loaded.</p>"; });

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (DATA && window.Chart) draw(DATA); });
