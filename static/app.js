(() => {
  "use strict";

  // Approximate Pantone Basic Color System hex values -- this ink system's
  // 14 colorants map to the standard basic-color set (Warm Red, Rubine Red,
  // Rhodamine Red, Reflex Blue, Process Blue, Purple, Violet, etc.), so real
  // approximations are used rather than arbitrary UI colors.
  const COLOR_MAP = {
    "UFO00061 transp. white": { label: "Transp. White", hex: "#ffffff", border: true },
    "UFO10031 yellow": { label: "Yellow", hex: "#fedd00" },
    "UFO20033 orange 021": { label: "Orange 021", hex: "#fe5000" },
    "UFO30001 warm red": { label: "Warm Red", hex: "#f9423a" },
    "UFO30002 rubine red": { label: "Rubine Red", hex: "#ce0058" },
    "UFO30003 rhodamine red": { label: "Rhodamine Red", hex: "#e10098" },
    "UFO30032 red 032": { label: "Red 032", hex: "#ef3340" },
    "UFO40011 purple": { label: "Purple", hex: "#632878" },
    "UFO40013 violet": { label: "Violet", hex: "#440099" },
    "UFO50021 reflex blue": { label: "Reflex Blue", hex: "#001489" },
    "UFO50022 process blue": { label: "Process Blue", hex: "#0085ca" },
    "UFO50072 blue 072": { label: "Blue 072", hex: "#14448b" },
    "UFO60051 green": { label: "Green", hex: "#00a651" },
    "UFO80071 black": { label: "Black", hex: "#2d2926" },
  };

  const els = {
    tabs: document.querySelectorAll(".tab"),
    tabContents: {
      sample: document.getElementById("tab-sample"),
      upload: document.getElementById("tab-upload"),
      paste: document.getElementById("tab-paste"),
    },
    loadSampleBtn: document.getElementById("load-sample-btn"),
    fileInput: document.getElementById("file-input"),
    fileDropLabel: document.getElementById("file-drop-label"),
    pasteLight: document.getElementById("paste-light"),
    pasteDark: document.getElementById("paste-dark"),
    pasteLoadBtn: document.getElementById("paste-load-btn"),
    predictBtn: document.getElementById("predict-btn"),
    inputStatus: document.getElementById("input-status"),
    chart: document.getElementById("spectrum-chart"),
    scaleOpts: document.querySelectorAll(".scale-opt"),
    resultsEmpty: document.getElementById("results-empty"),
    resultsContent: document.getElementById("results-content"),
    reviewBanner: document.getElementById("review-banner"),
    modeBanner: document.getElementById("mode-banner"),
    recipeList: document.getElementById("recipe-list"),
    sumCheck: document.getElementById("sum-check"),
    themeToggle: document.getElementById("theme-toggle"),
    resultsLoading: document.getElementById("results-loading"),
    loadStep1: document.getElementById("load-step-1"),
    loadStep2: document.getElementById("load-step-2"),
    loadElapsed: document.getElementById("load-elapsed-val"),
  };

  // ---------- shared state ----------
  // Declared BEFORE anything that reads them. applyTheme() below touches
  // currentSpectrum, and `let` bindings throw a ReferenceError if read above
  // their declaration -- which would abort this IIFE and silently leave every
  // event listener underneath unregistered.
  let wavelengths = [];
  let currentSpectrum = null;
  // 'fit' (axis follows the data) | 'full' (0-1) | 'log'. Fit is the default
  // because a dark-backing curve under 1% reflectance is invisible on 0-1.
  let chartScale = "fit";

  const cssVar = name =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ---------- theme ----------
  // Honour the OS by default; remember an explicit choice. The canvas reads its
  // colours from CSS custom properties, so a theme change must redraw it.
  const THEME_KEY = "vevo-theme";
  function applyTheme(mode) {
    if (mode) document.documentElement.setAttribute("data-theme", mode);
    else document.documentElement.removeAttribute("data-theme");
    if (currentSpectrum) drawChart(currentSpectrum);
  }
  function currentTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored) return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  applyTheme(localStorage.getItem(THEME_KEY));
  els.themeToggle.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
  });
  // Follow the OS while no explicit choice has been made.
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!localStorage.getItem(THEME_KEY)) applyTheme(null);
  });

  // ---------- chart axis scale ----------
  els.scaleOpts.forEach(btn => {
    btn.addEventListener("click", () => {
      chartScale = btn.dataset.scale;
      els.scaleOpts.forEach(b => b.classList.toggle("active", b === btn));
      if (currentSpectrum) drawChart(currentSpectrum);
    });
  });

  fetch("/api/meta").then(r => r.json()).then(meta => { wavelengths = meta.wavelengths; });

  // ---------- tabs ----------
  els.tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      els.tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      Object.entries(els.tabContents).forEach(([key, node]) => {
        node.hidden = key !== tab.dataset.tab;
      });
    });
  });

  // ---------- status helper ----------
  function setStatus(msg, kind) {
    els.inputStatus.textContent = msg;
    els.inputStatus.className = "status-text" + (kind ? " " + kind : "");
  }

  // Either backing may be supplied on its own. Both is still the best input,
  // so the status line says so rather than letting a reduced input pass
  // silently as if it were equivalent.
  function setSpectrum(spec, sourceLabel) {
    const hasLight = Array.isArray(spec.R_light);
    const hasDark = Array.isArray(spec.R_dark);
    if (!hasLight && !hasDark) {
      setStatus("JSON must contain an R_light array, an R_dark array, or both.", "error");
      return;
    }
    if (wavelengths.length) {
      for (const [key, present] of [["R_light", hasLight], ["R_dark", hasDark]]) {
        if (present && spec[key].length !== wavelengths.length) {
          setStatus(`${key} needs ${wavelengths.length} values ` +
                    `(380-730nm, 10nm steps), got ${spec[key].length}.`, "error");
          return;
        }
      }
    }

    currentSpectrum = {};
    if (hasLight) currentSpectrum.R_light = spec.R_light;
    if (hasDark) currentSpectrum.R_dark = spec.R_dark;
    els.predictBtn.disabled = false;

    if (hasLight && hasDark) {
      setStatus(`Loaded ${sourceLabel} — both backings.`, "ok");
    } else if (hasLight) {
      setStatus(`Loaded ${sourceLabel} — light backing only. ` +
                `Accuracy is slightly reduced; add a dark-backing reading if you have one.`, "warn");
    } else {
      setStatus(`Loaded ${sourceLabel} — dark backing only. ` +
                `Accuracy is substantially reduced; a light-backing reading is strongly preferred.`, "warn");
    }
    drawChart(currentSpectrum);
  }

  // ---------- sample ----------
  els.loadSampleBtn.addEventListener("click", async () => {
    setStatus("Loading sample...");
    try {
      const res = await fetch("/api/sample");
      const spec = await res.json();
      setSpectrum(spec, "sample spectrum");
    } catch (e) {
      setStatus("Could not load sample: " + e.message, "error");
    }
  });

  // ---------- upload ----------
  els.fileInput.addEventListener("change", () => {
    const file = els.fileInput.files[0];
    if (!file) return;
    els.fileDropLabel.textContent = file.name;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        // Same flexible parsing as the paste boxes. A JSON file names its own
        // backings; a bare list of numbers in a .csv/.txt cannot, so it is
        // taken as light backing -- the far more common single-backing case,
        // and the one the status line then states plainly.
        const { spec } = parseSpectrumInput(reader.result, "light");
        setSpectrum(spec, `"${file.name}"`);
      } catch (e) {
        setStatus(`Could not read "${file.name}": ${e.message}`, "error");
      }
    };
    reader.readAsText(file);
  });

  // ---------- flexible input parsing ----------
  // Instrument exports are rarely tidy JSON. Accept, in order:
  //   1. {"R_light": [...], "R_dark": [...]}  -- explicit, wins outright
  //   2. a bare JSON array                    -- backing comes from the picker
  //   3. raw numbers separated by commas, whitespace, newlines or semicolons
  // Returns {spec, label} or throws with a message aimed at the operator.
  function parseSpectrumInput(text, backing) {
    const raw = (text || "").trim();
    if (!raw) throw new Error("Nothing pasted yet.");

    // 1 & 2: valid JSON
    try {
      const parsed = JSON.parse(raw);
      if (parsed && !Array.isArray(parsed) && typeof parsed === "object") {
        if (!Array.isArray(parsed.R_light) && !Array.isArray(parsed.R_dark)) {
          throw new Error("JSON object must contain R_light, R_dark, or both.");
        }
        return { spec: parsed, label: "pasted JSON" };
      }
      if (Array.isArray(parsed)) {
        return {
          spec: { [backing === "dark" ? "R_dark" : "R_light"]: parsed.map(Number) },
          label: `pasted numbers (${backing} backing)`,
        };
      }
    } catch (e) {
      if (/must contain/.test(e.message)) throw e;   // our own message, keep it
      /* not JSON -- fall through to the loose number parser */
    }

    // 3: loose numbers
    const nums = raw.split(/[\s,;]+/).filter(Boolean).map(Number);
    if (nums.some(n => !Number.isFinite(n))) {
      throw new Error("Could not read that as numbers or JSON. Expected 36 reflectance values.");
    }
    if (!nums.length) throw new Error("No numbers found.");
    return {
      spec: { [backing === "dark" ? "R_dark" : "R_light"]: nums },
      label: `pasted numbers (${backing} backing)`,
    };
  }

  // ---------- paste ----------
  // Each box states its own backing, so nothing has to be inferred and no JSON
  // is required. A full JSON object dropped into either box still wins.
  els.pasteLoadBtn.addEventListener("click", () => {
    const boxes = [
      { el: els.pasteLight, backing: "light" },
      { el: els.pasteDark, backing: "dark" },
    ].filter(b => b.el && b.el.value.trim());

    if (!boxes.length) {
      setStatus("Paste numbers into the light box, the dark box, or both.", "error");
      return;
    }

    try {
      const spec = {};
      const filled = [];
      for (const { el, backing } of boxes) {
        const { spec: part } = parseSpectrumInput(el.value, backing);
        // A pasted JSON object may carry both backings on its own.
        Object.assign(spec, part);
        filled.push(backing);
      }
      const keys = Object.keys(spec);
      const desc = keys.length === 2 ? "both backings"
                 : keys[0] === "R_light" ? "light backing"
                 : "dark backing";
      setSpectrum(spec, `pasted numbers (${desc})`);
    } catch (e) {
      setStatus(e.message, "error");
    }
  });

  // ---------- chart ----------
  // Drawn at 2x the CSS size for crisp curves, and every colour is pulled from
  // CSS custom properties so the canvas follows the active theme rather than
  // holding its own hard-coded palette.
  function drawChart(spec) {
    const canvas = els.chart;
    const ctx = canvas.getContext("2d");
    const S = 2;                                   // drawing scale

    const anyCurve = spec.R_light || spec.R_dark;
    if (!anyCurve) return;

    const lightCol = cssVar("--curve-light") || "#0072ce";
    const darkCol = cssVar("--curve-dark") || "#1b2027";
    const series = [];
    if (spec.R_light) series.push({ v: spec.R_light, color: lightCol, name: "over light backing" });
    if (spec.R_dark) series.push({ v: spec.R_dark, color: darkCol, name: "over dark backing" });

    // A dark-backing spectrum routinely sits below 1% reflectance while the
    // light-backing one reaches 50-85%. Sharing one linear axis compresses the
    // dark curve to about a pixel of vertical variation -- it reads as a flat
    // line even though it is drawn correctly. Two y-scales in one frame would
    // be the other way out, and it is the classic way to mislead: the crossings
    // mean nothing. So "fit" gives each backing its own panel and its own
    // honest axis, which is the same comparison without the false geometry.
    const twoPanel = chartScale === "fit" && series.length === 2;
    const wantH = twoPanel ? 760 : 520;
    if (canvas.height !== wantH) canvas.height = wantH;   // note: this clears the canvas
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const wl = wavelengths.length ? wavelengths : anyCurve.map((_, i) => 380 + i * 10);
    // Panel captions sit above their plot, not inside it -- a curve that hugs
    // the top of its own fitted axis would otherwise run straight through them.
    const hasLabels = chartScale !== "full";
    const padR = 16 * S, padT = (hasLabels ? 28 : 16) * S, padB = 30 * S;

    // ---- ranges ----
    // 1/2/2.5/3/4/5/6/8/10 x 10^k: a denser ladder than powers of ten, so a
    // curve peaking at 0.51 gets a 0.6 ceiling rather than wasting half the panel.
    const niceCeil = v => {
      const e = Math.pow(10, Math.floor(Math.log10(v)));
      const f = v / e;
      const m = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 9, 10].find(n => f <= n * 1.0001) || 10;
      return m * e;
    };

    function fitRange(vals) {
      const mx = Math.max(...vals), mn = Math.min(...vals);
      // Zero baseline whenever the curve actually varies across its own height.
      // When it does not -- a dark curve spanning 0.0071 to 0.0074 is flat at
      // any zero-based scale -- pad around the data instead and say "zoomed",
      // so the shape is legible without the label implying it starts at zero.
      if (mx > 0 && (mx - mn) / mx >= 0.25) return { lo: 0, hi: niceCeil(mx), zeroed: true };
      const pad = (mx - mn) * 0.15 || mx * 0.05 || 1e-4;
      return { lo: Math.max(0, mn - pad), hi: mx + pad, zeroed: false };
    }

    const panels = twoPanel ? series.map(s => [s]) : [series];
    const specs = panels.map(ps => {
      const vals = [].concat(...ps.map(s => s.v));
      if (chartScale === "full") {
        return { lo: 0, hi: 1, zeroed: true, isLog: false, ticks: [0, 0.25, 0.5, 0.75, 1] };
      }
      if (chartScale === "log") {
        const lo = Math.pow(10, Math.floor(Math.log10(Math.max(Math.min(...vals), 1e-5))));
        let hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...vals, 1e-5))));
        if (hi <= lo) hi = lo * 10;          // a curve on a decade boundary would collapse the range
        const ticks = [];
        for (let d = lo; d <= hi * 1.0001; d *= 10) ticks.push(d);
        return { lo, hi, zeroed: false, isLog: true, ticks };
      }
      const r = fitRange(vals);
      const ticks = [0, 0.25, 0.5, 0.75, 1].map(f => r.lo + f * (r.hi - r.lo));
      return { ...r, isLog: false, ticks };
    });

    // Enough decimals to keep adjacent ticks distinct: a panel spanning 0.0005
    // needs five places, one spanning 0.6 needs two. A log axis is sized per
    // tick instead -- its ticks are decades apart, so one shared precision
    // would print both 0.001 and 0.01 as "0.00".
    const decimals = span => Math.min(6, Math.max(2, Math.ceil(-Math.log10(span || 1)) + 1));
    const logFmt = v =>
      v >= 0.1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v >= 0.001 ? v.toFixed(4) : v.toExponential(0);
    specs.forEach(sp => {
      const d = decimals(sp.hi - sp.lo);
      sp.fmt = sp.isLog ? logFmt : (v => v.toFixed(d));
    });

    const grid = cssVar("--grid") || "#e7eaed";
    const faint = cssVar("--ink-faint") || "#8b939d";
    const mono = `${11 * S}px ${cssVar("--font-mono") || "Consolas, monospace"}`;
    ctx.font = mono;

    // Size the label gutter to the widest tick actually drawn -- 0.00703 needs
    // more room than 0.25, and a fixed gutter would clip it.
    const padL = Math.max(...specs.flatMap(sp => sp.ticks.map(t => ctx.measureText(sp.fmt(t)).width))) + 20 * S;
    const plotW = W - padL - padR;
    const xAt = i => padL + (i / (wl.length - 1)) * plotW;

    const gap = (hasLabels ? 44 : 30) * S;
    const panelH = (H - padT - padB - gap * (panels.length - 1)) / panels.length;

    panels.forEach((ps, pi) => {
      const sp = specs[pi];
      const top = padT + pi * (panelH + gap);
      const bot = top + panelH;
      const lo = sp.isLog ? Math.log10(sp.lo) : sp.lo;
      const hi = sp.isLog ? Math.log10(sp.hi) : sp.hi;
      const yAt = v => {
        const t = sp.isLog ? Math.log10(Math.max(sp.lo, Math.min(sp.hi, v))) : v;
        return bot - Math.max(0, Math.min(1, (t - lo) / (hi - lo))) * panelH;
      };

      ctx.lineWidth = 1 * S;
      ctx.font = mono;
      ctx.textBaseline = "middle";
      sp.ticks.forEach(v => {
        const y = yAt(v);
        ctx.strokeStyle = grid;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(W - padR, y);
        ctx.stroke();
        ctx.fillStyle = faint;
        ctx.textAlign = "right";
        ctx.fillText(sp.fmt(v), padL - 9 * S, y);
      });

      // Name the panel and its scale. An axis that is zoomed or logarithmic must
      // say so on the panel itself -- a reader should never have to check a
      // toggle elsewhere to know whether a curve is small or merely magnified.
      const bits = [];
      if (twoPanel) bits.push(ps[0].name);
      if (sp.isLog) bits.push("log axis");
      else if (chartScale === "fit" && !sp.zeroed) bits.push(`zoomed ${sp.fmt(sp.lo)}–${sp.fmt(sp.hi)}`);
      else if (chartScale === "fit") bits.push(`0–${sp.fmt(sp.hi)}`);
      if (bits.length) {
        ctx.fillStyle = faint;
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(bits.join("  ·  "), padL, top - 7 * S);
        ctx.textBaseline = "middle";
      }

      // Wavelength ticks on every panel, labelled only under the last one --
      // the strip below the canvas is what those labels align to.
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      wl.forEach((nm, i) => {
        if (nm % 50 !== 0) return;
        ctx.strokeStyle = grid;
        ctx.beginPath();
        ctx.moveTo(xAt(i), bot);
        ctx.lineTo(xAt(i), bot + 5 * S);
        ctx.stroke();
        if (pi === panels.length - 1) {
          ctx.fillStyle = faint;
          ctx.fillText(String(nm), xAt(i), bot + 9 * S);
        }
      });

      ps.forEach(s => {
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        // Fill only under a zero-based linear axis: the area under a curve is
        // read as a quantity, and shading down to a floor that isn't zero
        // overstates it.
        if (sp.zeroed && !sp.isLog && s.color === lightCol) {
          const grad = ctx.createLinearGradient(0, top, 0, bot);
          grad.addColorStop(0, hexA(s.color, 0.22));
          grad.addColorStop(1, hexA(s.color, 0));
          ctx.beginPath();
          s.v.forEach((v, i) => (i ? ctx.lineTo(xAt(i), yAt(v)) : ctx.moveTo(xAt(i), yAt(v))));
          ctx.lineTo(xAt(s.v.length - 1), bot);
          ctx.lineTo(xAt(0), bot);
          ctx.closePath();
          ctx.fillStyle = grad;
          ctx.fill();
        }
        ctx.beginPath();
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2.4 * S;
        s.v.forEach((v, i) => (i ? ctx.lineTo(xAt(i), yAt(v)) : ctx.moveTo(xAt(i), yAt(v))));
        ctx.stroke();
      });
    });

    // Dim the legend entry for a backing that wasn't provided, so the chart
    // reads as "one curve supplied" rather than "a curve failed to render".
    document.querySelectorAll(".chart-legend span").forEach(el => {
      const isLight = el.textContent.toLowerCase().includes("light");
      const present = isLight ? !!spec.R_light : !!spec.R_dark;
      el.style.opacity = present ? "1" : "0.3";
      el.title = present ? "" : "not supplied for this sample";
    });
  }

  // #rrggbb -> rgba(), for the area fill under the light-backing curve.
  function hexA(hex, alpha) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec((hex || "").trim());
    if (!m) return `rgba(0,114,206,${alpha})`;
    const [r, g, b] = [1, 2, 3].map(i => parseInt(m[i], 16));
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // ---------- predict ----------
  // Stage 1 now runs on a GPU-accelerated foundation model (TabICL) that
  // does inference-time computation over the full training set on every
  // call -- typically ~60-90s, not instant. Without visible progress this
  // reads as a frozen page, so show elapsed time throughout the wait.
  const predictBtnLabel = els.predictBtn.querySelector(".btn-label");

  // The two stages are sequential and Stage 1 dominates the wall time, so the
  // panel marks Stage 1 active immediately and flips to Stage 2 once the
  // typical Stage 1 duration has passed. It reflects the real pipeline shape
  // rather than pretending to report true server-side progress.
  const STAGE_1_TYPICAL_MS = 60000;

  function showLoading(on) {
    if (on) {
      els.resultsEmpty.hidden = true;
      els.resultsContent.hidden = true;
      els.resultsLoading.hidden = false;
      els.loadStep1.className = "load-step is-active";
      els.loadStep2.className = "load-step";
      els.loadElapsed.textContent = "0s";
    } else {
      els.resultsLoading.hidden = true;
    }
  }

  els.predictBtn.addEventListener("click", async () => {
    if (!currentSpectrum) return;
    els.predictBtn.disabled = true;
    els.predictBtn.classList.add("is-loading");

    const startedAt = Date.now();
    showLoading(true);

    const tick = () => {
      const ms = Date.now() - startedAt;
      const secs = Math.round(ms / 1000);
      els.loadElapsed.textContent = secs + "s";
      predictBtnLabel.textContent = `Predicting… ${secs}s`;
      if (ms > STAGE_1_TYPICAL_MS) {
        els.loadStep1.className = "load-step is-done";
        els.loadStep2.className = "load-step is-active";
      }
    };
    tick();
    const timer = setInterval(tick, 1000);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentSpectrum),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      showLoading(false);
      renderResults(data);
      setStatus("Prediction complete.", "ok");
    } catch (e) {
      showLoading(false);
      // Restore whichever panel state the user came from, so a failure never
      // leaves the results column blank with no explanation.
      if (els.resultsContent.dataset.hasResult !== "1") els.resultsEmpty.hidden = false;
      setStatus("Prediction error: " + e.message, "error");
    } finally {
      clearInterval(timer);
      els.predictBtn.disabled = false;
      els.predictBtn.classList.remove("is-loading");
      predictBtnLabel.textContent = "Predict recipe";
    }
  });

  // Which model answered, and how much confidence it deserves. Shown first so
  // a reduced-input result can never be mistaken for a full-accuracy one.
  function renderMode(data) {
    const m = data.mode;
    if (!m) { els.modeBanner.hidden = true; return; }
    els.modeBanner.hidden = false;
    els.modeBanner.className = "mode-banner" + (m.is_degraded ? " degraded" : " full");
    // expected_exact_match is null when this mode has not been re-measured
    // against the deployed 0.1% threshold. Showing its old 2%-era figure beside
    // the light-only 88.2% told users that supplying LESS data scored BETTER,
    // because the two numbers answer different questions. Say nothing rather
    // than say something false; the advice text explains why.
    const hasPct = m.expected_exact_match != null;
    const pct = hasPct ? (m.expected_exact_match * 100).toFixed(1) : null;
    // cost_vs_both is null when this mode's accuracy was scored against a
    // different presence threshold than the both-backing figure. Rendering the
    // subtraction anyway would print "+1.0% vs both backings" -- telling the
    // user that supplying LESS data improves accuracy, which is false. Name the
    // labelling the number came from instead.
    let note = "";
    if (!hasPct) {
      note = `<span class="mb-note">accuracy pending re-measurement at the 0.1% threshold</span>`;
    } else if (m.is_degraded && m.cost_vs_both != null) {
      note = `<span class="mb-delta">${(m.cost_vs_both * 100).toFixed(1)}% vs both backings</span>`;
    } else if (m.is_degraded && m.measured_at_threshold != null) {
      const t = +(m.measured_at_threshold * 100).toFixed(3);
      note = `<span class="mb-note">measured at the ${t}% presence threshold &mdash; not comparable to the both-backing figure</span>`;
    }
    els.modeBanner.innerHTML =
      `<div class="mb-head"><strong>${escapeHtml(m.label)}</strong>` +
      (hasPct ? `<span class="mb-acc">${pct}% expected accuracy</span>` : "") +
      `${note}</div>` +
      `<div class="mb-advice">${escapeHtml(m.advice)}</div>`;
  }

  function renderResults(data) {
    els.resultsEmpty.hidden = true;
    els.resultsLoading.hidden = true;
    els.resultsContent.hidden = false;
    els.resultsContent.dataset.hasResult = "1";
    renderMode(data);

    // Review reasons are independent: a recipe may trip the Red 032 flag, the
    // colorant-count flag, both, or neither. review_flags is the full list;
    // needs_review is kept as a fallback so a cached older response still shows
    // the Red 032 warning rather than silently dropping it.
    const flags = Array.isArray(data.review_flags) ? data.review_flags : [];
    const legacyOnly = flags.length === 0 && !!data.needs_review;
    if (flags.length || legacyOnly) {
      els.reviewBanner.hidden = false;
      const icon =
        `<svg class="rb-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">` +
        `<path d="M12 9v4m0 4h.01M10.3 3.86L2.7 17.14A1.6 1.6 0 004.1 19.5h15.8a1.6 1.6 0 001.4-2.36L13.7 3.86a1.6 1.6 0 00-2.8 0z" ` +
        `stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
      // Red 032 keeps its richer wording, including the friendly colorant name.
      const red032 =
        `This recipe includes ${escapeHtml(COLOR_MAP[data.review_colorant]?.label || data.review_colorant)}, ` +
        `the model's most data-starved colorant (Stage 2 R&sup2;=0.47 on held-out test data). Verify against a physical ` +
        `drawdown before use &mdash; do not auto-approve.`;
      const items = legacyOnly
        ? [red032]
        : flags.map(f => (f.id === "red_032" ? red032 : escapeHtml(f.message)));
      const body = items.length > 1
        ? `<ul class="rb-list">${items.map(t => `<li>${t}</li>`).join("")}</ul>`
        : items[0];
      els.reviewBanner.innerHTML =
        icon + `<div><strong>Manual review required</strong>${body}</div>`;
    } else {
      els.reviewBanner.hidden = true;
    }

    const entries = Object.entries(data.recipe);
    const maxPct = Math.max(...entries.map(([, p]) => p), 1);
    els.recipeList.innerHTML = "";
    const fills = [];
    entries.forEach(([name, pct], i) => {
      const meta = COLOR_MAP[name] || { label: name, hex: "#999" };
      const li = document.createElement("li");
      li.className = "recipe-row";
      li.style.animationDelay = `${i * 45}ms`;
      li.innerHTML = `
        <span class="swatch" style="background:${meta.hex}${meta.border ? ";border-color:#ccc" : ""}"></span>
        <span class="recipe-main">
          <span class="recipe-name">${meta.label}</span>
          <span class="recipe-bar-track"><span class="recipe-bar-fill" style="background:${meta.hex}"></span></span>
        </span>
        <span class="recipe-pct">${pct.toFixed(2)}%</span>
      `;
      els.recipeList.appendChild(li);
      fills.push([li.querySelector(".recipe-bar-fill"), (pct / maxPct) * 100]);
    });
    // Bars start at width:0 (set in CSS) and are pushed to their target width
    // on the next frame so the CSS transition actually animates them in,
    // instead of rendering already at full width on creation.
    requestAnimationFrame(() => {
      fills.forEach(([el, pct]) => { el.style.width = pct + "%"; });
    });

    const sum = entries.reduce((s, [, p]) => s + p, 0);
    els.sumCheck.textContent = `Sum: ${sum.toFixed(2)}%`;

  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));
  }
})();
