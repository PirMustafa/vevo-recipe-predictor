"""Local web front end for the spectrum -> recipe prediction pipeline.

Usage:
    python app.py
Then open http://127.0.0.1:5000 in a browser.

Loads Stage 1 + Stage 2 models once at startup (not per-request -- the Stage
1 classifier alone is ~77MB, reloading it on every prediction would make the
UI feel sluggish for no reason).
"""
from __future__ import annotations

import pathlib

from flask import Flask, jsonify, render_template, request, send_from_directory

from predict import (
    RED_032,
    predict_recipes_batch,
    available_modes,
    load_artifacts,
    needs_manual_review,
    predict_recipe_flexible,
    review_flags,
    serving_report,
)
from src.backing_modes import MODE_BOTH, MODE_DARK, MODE_LIGHT, detect_mode
from src.batch_queue import PredictionQueue
from src.cxf_parser import COLORANT_NAMES, WAVELENGTHS

ROOT = pathlib.Path(__file__).parent
app = Flask(__name__, static_folder="static", template_folder="templates")

print("Loading Stage 1 + Stage 2 models...")
_ARTIFACTS = load_artifacts()
# The both-backing (production) models stay resident, since that is the normal
# path. Light-only and dark-only load lazily on first use -- together they are
# another ~241MB, and most sessions never touch them.
_ARTIFACTS_BY_MODE = {MODE_BOTH: _ARTIFACTS}
_MODES = available_modes()
# Name the artifacts, not just the mode. Two different models can both answer
# to "light" -- the pinned wolves_v1 retrain and the superseded 2%-trained one
# differ by 29 points of exact-match and are indistinguishable from the mode
# name alone. This log is the only record of what this process is serving.
print(f"Backing modes available: {', '.join(_MODES)}")
print(serving_report(_MODES))
# NOTE: the app deliberately does NOT compare results against the Pantone
# library. The comparison still exists in the codebase -- src/pantone.py backs
# train.py's target-delta training gate and the eval_pantone_* scripts -- it is
# just not part of what this tool reports. Removed 2026-08-20 by request.
# Requests are served through a micro-batching queue. TabICL's cost is almost
# entirely fixed (it re-reads all 2051 training rows per call): 1 spectrum
# takes ~40s and 297 take ~42s. Without batching, five users pressing Predict
# together wait 5 x 40s serially. With it they share ONE call and all finish in
# ~40s. The HTTP API is unchanged -- /api/predict still blocks and still
# returns a recipe -- so no client needs to know this exists.
_QUEUE = PredictionQueue(
    predict_batch=lambda mode, specs: predict_recipes_batch(
        mode, specs, artifacts_by_mode=_ARTIFACTS_BY_MODE),
    mode_of=lambda r_light, r_dark: detect_mode(r_light, r_dark),
)
print("Request batching enabled (window 1.5s, max 16 per call).")
print("Ready.")


def _asset_version() -> str:
    """A fingerprint that changes whenever app.js or style.css changes.

    Without this, a browser that has the page open across an update keeps
    running the JavaScript it already has -- the server serves the new file,
    the user sees the old behaviour, and the bug looks unfixed. Appending this
    to the asset URLs makes an updated file a different URL, so there is
    nothing stale to reuse. Cheap: two stat() calls per page load.
    """
    stamp = 0.0
    for name in ("app.js", "style.css"):
        path = ROOT / "static" / name
        if path.is_file():
            stamp = max(stamp, path.stat().st_mtime)
    return str(int(stamp))


@app.route("/")
def index():
    return render_template("index.html", v=_asset_version())


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(ROOT / "static", filename)


@app.route("/api/meta")
def meta():
    return jsonify({
        "wavelengths": WAVELENGTHS,
        "colorant_names": COLORANT_NAMES,
        "red032_name": RED_032,
    })


@app.route("/api/sample")
def sample():
    import json
    with open(ROOT / "sample_spectrum.json") as f:
        return jsonify(json.load(f))


def _clean_spectrum(values, name):
    """Validate one optional spectrum. Returns (list|None, error|None)."""
    if values is None:
        return None, None
    if not isinstance(values, list):
        return None, f"{name} must be an array of numbers"
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        return None, f"{name} must be an array of numbers"
    if len(values) != len(WAVELENGTHS):
        return None, (f"{name} needs {len(WAVELENGTHS)} values "
                      f"(380-730nm, 10nm steps), got {len(values)}")
    if any(v < 0 or v > 1 for v in values):
        return None, f"{name} values must be between 0 and 1"
    return values, None


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True, silent=True) or {}

    # Either spectrum may be omitted; the router picks the matching model.
    # Supplying both is the normal case and behaves exactly as it always has.
    r_light, err = _clean_spectrum(body.get("R_light"), "R_light")
    if err:
        return jsonify({"error": err}), 400
    r_dark, err = _clean_spectrum(body.get("R_dark"), "R_dark")
    if err:
        return jsonify({"error": err}), 400

    if r_light is None and r_dark is None:
        return jsonify({
            "error": "Supply R_light, R_dark, or both. Both gives the best accuracy."
        }), 400

    try:
        out = _QUEUE.wait(_QUEUE.submit(r_light, r_dark))
        if "error" in out:
            return jsonify({"error": out["error"]}), 500
        recipe, mode_info = out["recipe"], out["mode"]
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # Two independent review reasons, kept separate so the UI can show either
    # one or both. needs_review / review_colorant are retained unchanged for
    # any client already reading them; `review_flags` is the fuller answer.
    flags = review_flags(recipe)
    return jsonify({
        "recipe": recipe,
        "needs_review": needs_manual_review(recipe),
        "review_colorant": RED_032,
        "review_flags": flags,
        "mode": mode_info,
    })


if __name__ == "__main__":
    import os

    # Bind address must be configurable, and must default to loopback.
    #
    # Flask's default binds to 127.0.0.1, which is correct on a workstation and
    # WRONG inside a container: the process starts, logs "Running on
    # http://127.0.0.1:5000", passes a naive health check, and then refuses
    # every connection from outside the container. That failure looks like a
    # networking or security-group problem and is not.
    #
    # The default stays 127.0.0.1 so running this locally is unchanged and the
    # app is never accidentally exposed. The container sets HOST=0.0.0.0
    # explicitly, and is itself reached only through an SSM tunnel -- there is
    # no public listener and no authentication in this app, so binding it to a
    # routable interface without that tunnel would publish the client's
    # formulation data to anyone who found the address.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    if host != "127.0.0.1":
        print(f"NOTE: binding to {host}:{port}. This app has NO authentication; "
              f"it must sit behind a tunnel or a private network, never on a "
              f"public interface.")
    app.run(debug=False, host=host, port=port)
