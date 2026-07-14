"""
app.py — Flask Application Entry Point
=========================================
Main Flask server for the Pancreatic Cancer AI Detection system.

Routes:
  GET  /              — Landing page with upload form
  POST /predict       — Upload & run AI inference pipeline
  GET  /result/<id>   — Display analysis results dashboard
  GET  /report/<id>   — View PDF report (inline)
  GET  /download/<id> — Download PDF report
  POST /api/predict   — JSON API for inference
  GET  /api/metrics/<id> — JSON API for metrics

The application loads ML models once at startup for fast inference.
"""

import os
import uuid
import time
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, jsonify, session
)
from werkzeug.utils import secure_filename

import config
from utils.preprocessing import allowed_file, validate_image
from utils.inference import run_full_inference, load_models
from utils.report import generate_pdf_report, generate_clinical_summary


# ===========================================================================
# Flask App Factory
# ===========================================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER

# In-memory results store (for demo; use database in production)
results_store = {}


# ===========================================================================
# Model Preloading — Load once at startup
# ===========================================================================

print("=" * 60)
print("  PancreasAI — Pancreatic Cancer Detection System")
print("=" * 60)
print(f"  Device:    {config.DEVICE}")
print(f"  Demo Mode: {config.DEMO_MODE}")
print(f"  Models:    {config.MODEL_DIR}")
print("=" * 60)

# Load models into memory (or initialize demo mode)
load_models()
print("[OK] Application ready.\n")


# ===========================================================================
# Route: Landing Page
# ===========================================================================

@app.route("/")
def index():
    """
    Render the landing page with the upload form.
    """
    return render_template("index.html")


# ===========================================================================
# Route: Upload & Predict
# ===========================================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle file upload and run the full AI inference pipeline.
    
    Steps:
      1. Validate the uploaded file
      2. Save to uploads directory
      3. Run preprocessing + inference + metrics + visualization
      4. Generate PDF report
      5. Redirect to results dashboard
    """
    # --- Check if file was uploaded ---
    if "file" not in request.files:
        flash("No file selected. Please upload a CT scan image.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected. Please choose a file.", "error")
        return redirect(url_for("index"))

    # --- Validate file extension ---
    if not allowed_file(file.filename):
        flash("Invalid file format. Supported: PNG, JPG, JPEG.", "error")
        return redirect(url_for("index"))

    # --- Sanitize filename and save ---
    original_filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    ext = original_filename.rsplit(".", 1)[1].lower()
    safe_filename = f"{file_id}.{ext}"
    filepath = os.path.join(config.UPLOAD_FOLDER, safe_filename)

    file.save(filepath)

    # --- Validate image integrity ---
    is_valid, message = validate_image(filepath)
    if not is_valid:
        os.remove(filepath)  # Clean up invalid file
        flash(f"Invalid image: {message}", "error")
        return redirect(url_for("index"))

    try:
        # --- Run full inference pipeline ---
        results = run_full_inference(filepath, file_id)

        # --- Generate PDF report ---
        report_path = generate_pdf_report(results, file_id=file_id)
        if report_path:
            results["report_path"] = report_path

        # --- Store results ---
        results_store[file_id] = results

        # Redirect to results page
        return redirect(url_for("result", file_id=file_id))

    except Exception as e:
        flash(f"Analysis failed: {str(e)}", "error")
        return redirect(url_for("index"))


# ===========================================================================
# Route: Results Dashboard
# ===========================================================================

@app.route("/result/<file_id>")
def result(file_id):
    """
    Display the analysis results dashboard for a given file ID.
    """
    results = results_store.get(file_id)

    if results is None:
        flash("Results not found. Please upload a new image.", "error")
        return redirect(url_for("index"))

    return render_template("result.html", results=results)


# ===========================================================================
# Route: View Report
# ===========================================================================

@app.route("/report/<file_id>")
def report(file_id):
    """
    Display the PDF report inline in the browser.
    """
    results = results_store.get(file_id)
    if results is None or "report_path" not in results:
        flash("Report not found.", "error")
        return redirect(url_for("index"))

    report_path = results["report_path"]
    if not os.path.exists(report_path):
        flash("Report file not found on server.", "error")
        return redirect(url_for("index"))

    return send_file(report_path, mimetype="application/pdf")


# ===========================================================================
# Route: Download Report
# ===========================================================================

@app.route("/download/<file_id>")
def download(file_id):
    """
    Download the PDF report as an attachment.
    """
    results = results_store.get(file_id)
    if results is None or "report_path" not in results:
        flash("Report not found.", "error")
        return redirect(url_for("index"))

    report_path = results["report_path"]
    if not os.path.exists(report_path):
        flash("Report file not found on server.", "error")
        return redirect(url_for("index"))

    return send_file(
        report_path,
        as_attachment=True,
        download_name=f"PancreasAI_Report_{file_id[:8]}.pdf",
        mimetype="application/pdf",
    )


# ===========================================================================
# API Route: JSON Prediction
# ===========================================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint for programmatic access.
    
    Request: POST with 'file' in form-data.
    Response: JSON with prediction results.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    # Save file
    original_filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    ext = original_filename.rsplit(".", 1)[1].lower()
    filepath = os.path.join(config.UPLOAD_FOLDER, f"{file_id}.{ext}")
    file.save(filepath)

    # Validate
    is_valid, message = validate_image(filepath)
    if not is_valid:
        os.remove(filepath)
        return jsonify({"error": message}), 400

    try:
        # Run inference
        results = run_full_inference(filepath, file_id)

        # Build API response
        api_response = {
            "prediction": results["prediction"],
            "confidence": results["confidence"],
            "cancer_probability": results["cancer_probability"],
            "tumor_area": results["tumor_area"],
            "tumor_volume": results["tumor_volume"],
            "tumor_location": results["tumor_location"],
            "risk_level": results["risk_level"],
            "stage_suggestion": results["stage_suggestion"],
            "dice_score": results["metrics"].get("dice_score", 0),
            "iou": results["metrics"].get("iou", 0),
            "precision": results["metrics"].get("precision", 0),
            "recall": results["metrics"].get("recall", 0),
            "f1_score": results["metrics"].get("f1_score", 0),
            "inference_time": results["inference_time"],
            "heatmap": results.get("heatmap", ""),
            "mask": results.get("tumor_mask", ""),
            "overlay": results.get("overlay", ""),
            "file_id": file_id,
            "demo_mode": results.get("demo_mode", False),
        }

        # Store results for potential follow-up
        results_store[file_id] = results

        return jsonify(api_response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================================================================
# API Route: Get Metrics
# ===========================================================================

@app.route("/api/metrics/<file_id>")
def api_metrics(file_id):
    """
    Return evaluation metrics for a completed analysis.
    """
    results = results_store.get(file_id)
    if results is None:
        return jsonify({"error": "Results not found"}), 404

    return jsonify({
        "file_id": file_id,
        "metrics": results.get("metrics", {}),
        "inference_time": results.get("inference_time", 0),
    }), 200


# ===========================================================================
# Error Handlers
# ===========================================================================

@app.errorhandler(413)
def file_too_large(e):
    """Handle file size exceeding MAX_CONTENT_LENGTH."""
    flash("File too large. Maximum upload size is 16 MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    flash("Page not found.", "error")
    return redirect(url_for("index"))


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    flash("An internal error occurred. Please try again.", "error")
    return redirect(url_for("index"))


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=config.DEBUG,
    )
