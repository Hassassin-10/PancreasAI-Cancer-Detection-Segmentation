"""
report.py — PDF Report Generation Module
===========================================
Generates hospital-style PDF reports using ReportLab.
Includes patient info, prediction results, tumor metrics,
segmentation images, evaluation metrics, and clinical summary.
"""

import os
import io
import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch, mm
    from reportlab.lib.colors import HexColor, black, white, gray
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable, PageBreak
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_clinical_summary(results):
    """
    Generate a natural language clinical summary from the analysis results.
    
    This summary mimics how a radiologist might describe AI-assisted findings.

    Args:
        results (dict): Complete inference results dictionary.

    Returns:
        str: Clinical summary text.
    """
    if results.get("is_valid") == False:
        summary = (
            f"AI-ASSISTED ANALYSIS SUMMARY\n\n"
            f"WARNING: LOW-CONFIDENCE PREDICTION DETECTED\n"
            f"Validation Status: {results.get('validation_msg', 'Anatomical features out of range')}\n\n"
            f"The safety validation layer rejected this prediction because the segmented structures "
            f"did not satisfy clinical criteria (e.g. area size bounds or multiple disconnected segments). "
            f"Therefore, the segmentation masks and overlays have been suppressed for safety.\n\n"
            f"RECOMMENDATION: Verify that the uploaded image is a valid abdominal CT scan slice containing the "
            f"pancreas. Manual radiologist correlation is strongly advised."
        )
        return summary

    prediction = results.get("prediction", "Unknown")
    confidence = results.get("confidence", 0)
    tumor_area = results.get("tumor_area", 0)
    tumor_volume = results.get("tumor_volume", 0)
    tumor_location = results.get("tumor_location", "Not Detected")
    risk_level = results.get("risk_level", "Unknown")
    stage = results.get("stage_suggestion", "Unknown")

    if results.get("prediction_label") == "Cancer":
        summary = (
            f"AI-ASSISTED ANALYSIS SUMMARY\n\n"
            f"The deep learning model has identified findings consistent with "
            f"pancreatic malignancy with a confidence of {confidence}%. "
            f"The detected lesion is located in the {tumor_location} region, "
            f"with an estimated area of {tumor_area} cm² and an approximate "
            f"volume of {tumor_volume} cc.\n\n"
            f"STAGING: {stage}\n"
            f"RISK ASSESSMENT: {risk_level}\n\n"
            f"The segmentation analysis reveals a focal lesion that warrants "
            f"further clinical evaluation. The Grad-CAM attention map highlights "
            f"the regions most influential in the model's decision.\n\n"
            f"RECOMMENDATION: Correlation with clinical history, laboratory "
            f"markers (CA 19-9), and contrast-enhanced CT/MRI is strongly "
            f"recommended. Referral to a hepatopancreatobiliary specialist "
            f"is advised for further workup."
        )
    else:
        summary = (
            f"AI-ASSISTED ANALYSIS SUMMARY\n\n"
            f"The deep learning model has classified this scan as NORMAL "
            f"with a confidence of {confidence}%. No significant pancreatic "
            f"lesion was detected in the analyzed image.\n\n"
            f"RISK ASSESSMENT: {risk_level}\n\n"
            f"The segmentation analysis did not identify any focal mass lesion "
            f"in the pancreatic parenchyma. The pancreatic morphology appears "
            f"within normal limits based on the AI analysis.\n\n"
            f"RECOMMENDATION: If clinical suspicion remains high, correlation "
            f"with clinical presentation and laboratory markers is recommended. "
            f"Consider follow-up imaging if symptoms persist."
        )

    return summary


def generate_pdf_report(results, save_dir=None, file_id=""):
    """
    Generate a professional hospital-style PDF report.

    Args:
        results (dict): Complete inference results dictionary.
        save_dir (str): Directory to save the PDF.
        file_id (str): Unique file identifier.

    Returns:
        str: Path to the generated PDF file.
    """
    if not REPORTLAB_AVAILABLE:
        return None

    if save_dir is None:
        save_dir = config.REPORT_FOLDER

    os.makedirs(save_dir, exist_ok=True)

    pdf_path = os.path.join(save_dir, f"{file_id}_report.pdf")

    # --- Create PDF document ---
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    # --- Define styles ---
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        textColor=HexColor("#1a56db"),
        spaceAfter=6,
        alignment=TA_CENTER,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=HexColor("#1a56db"),
        spaceBefore=12,
        spaceAfter=6,
        borderWidth=1,
        borderColor=HexColor("#1a56db"),
        borderPadding=4,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=4,
    )

    small_style = ParagraphStyle(
        "SmallText",
        parent=styles["Normal"],
        fontSize=8,
        textColor=gray,
        alignment=TA_CENTER,
    )

    # --- Build document elements ---
    elements = []

    # ---- Header ----
    elements.append(Paragraph(config.REPORT_TITLE, title_style))
    elements.append(Paragraph(config.INSTITUTION_NAME, ParagraphStyle(
        "Institution", parent=styles["Normal"], fontSize=11,
        alignment=TA_CENTER, textColor=HexColor("#4b5563")
    )))
    elements.append(Spacer(1, 4 * mm))
    elements.append(HRFlowable(width="100%", thickness=2,
                               color=HexColor("#1a56db")))
    elements.append(Spacer(1, 4 * mm))

    # ---- Patient Info ----
    now = datetime.datetime.now()
    patient_data = [
        ["Patient ID:", file_id[:12].upper(),
         "Date:", now.strftime("%Y-%m-%d")],
        ["Analysis ID:", file_id[:8],
         "Time:", now.strftime("%H:%M:%S")],
        ["Report Type:", "AI-Assisted Screening",
         "Mode:", "Demo" if results.get("demo_mode") else "Clinical"],
    ]

    patient_table = Table(patient_data, colWidths=[80, 140, 60, 140])
    patient_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#374151")),
        ("TEXTCOLOR", (2, 0), (2, -1), HexColor("#374151")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 6 * mm))

    # ---- Prediction Result ----
    elements.append(Paragraph("PREDICTION RESULT", heading_style))

    prediction = results.get("prediction", "Unknown")
    confidence = results.get("confidence", 0)
    cancer_prob = results.get("cancer_probability", 0)

    pred_color = "#dc2626" if "Cancer" in prediction else "#16a34a"
    elements.append(Paragraph(
        f'<font size="14" color="{pred_color}"><b>{prediction}</b></font>',
        ParagraphStyle("Pred", parent=body_style, alignment=TA_CENTER, spaceAfter=8)
    ))

    pred_data = [
        ["Metric", "Value"],
        ["Cancer Probability", f"{cancer_prob}%"],
        ["Confidence", f"{confidence}%"],
        ["Risk Level", results.get("risk_level", "N/A")],
    ]
    pred_table = Table(pred_data, colWidths=[200, 200])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a56db")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#d1d5db")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f3f4f6")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(pred_table)
    elements.append(Spacer(1, 4 * mm))

    # ---- Tumor Metrics ----
    elements.append(Paragraph("TUMOR ANALYSIS", heading_style))

    tumor_data = [
        ["Parameter", "Value"],
        ["Tumor Area", f"{results.get('tumor_area', 0)} cm²"],
        ["Estimated Volume", f"{results.get('tumor_volume', 0)} cc"],
        ["Location", results.get("tumor_location", "N/A")],
        ["Stage Suggestion", results.get("stage_suggestion", "N/A")],
        ["Inference Time", f"{results.get('inference_time', 0)} seconds"],
        ["GPU Status", results.get("gpu_status", "N/A")],
    ]
    tumor_table = Table(tumor_data, colWidths=[200, 200])
    tumor_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a56db")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#d1d5db")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f3f4f6")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(tumor_table)
    elements.append(Spacer(1, 4 * mm))

    # ---- Evaluation Metrics ----
    elements.append(Paragraph("EVALUATION METRICS", heading_style))

    metrics = results.get("metrics", {})
    metrics_data = [["Metric", "Score"]]
    metric_names = {
        "dice_score": "Dice Coefficient (DSC)",
        "iou": "Intersection over Union (IoU)",
        "hausdorff_95": "Hausdorff Distance 95% (HD95)",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "accuracy": "Accuracy",
        "asd": "Avg. Surface Distance (ASD)",
        "volume_similarity": "Volume Similarity (VS)",
    }
    for key, display_name in metric_names.items():
        value = metrics.get(key, "N/A")
        if isinstance(value, float):
            value = f"{value:.4f}"
        metrics_data.append([display_name, str(value)])

    metrics_table = Table(metrics_data, colWidths=[260, 140])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a56db")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#d1d5db")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f3f4f6")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 4 * mm))

    # ---- Images ----
    elements.append(Paragraph("VISUALIZATION", heading_style))

    # Try to embed heatmap overlay and segmentation overlay
    for img_key, label in [("heatmap_overlay", "Grad-CAM Heatmap"),
                           ("overlay", "Segmentation Overlay")]:
        img_rel_path = results.get(img_key, "")
        img_abs_path = os.path.join(config.BASE_DIR, img_rel_path)
        if os.path.exists(img_abs_path):
            try:
                elements.append(Paragraph(f"<b>{label}</b>", body_style))
                elements.append(RLImage(img_abs_path, width=160 * mm, height=120 * mm,
                                         kind="proportional"))
                elements.append(Spacer(1, 4 * mm))
            except Exception:
                pass

    # ---- Clinical Summary ----
    elements.append(PageBreak())
    elements.append(Paragraph("CLINICAL SUMMARY", heading_style))

    summary = generate_clinical_summary(results)
    for line in summary.split("\n"):
        if line.strip():
            elements.append(Paragraph(line.strip(), body_style))
        else:
            elements.append(Spacer(1, 2 * mm))

    elements.append(Spacer(1, 6 * mm))

    # ---- Doctor Notes Section ----
    elements.append(Paragraph("PHYSICIAN NOTES", heading_style))
    elements.append(Spacer(1, 3 * mm))
    for _ in range(5):
        elements.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#d1d5db")))
        elements.append(Spacer(1, 8 * mm))

    # ---- Disclaimer ----
    elements.append(Spacer(1, 8 * mm))
    elements.append(HRFlowable(width="100%", thickness=1, color=HexColor("#dc2626")))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "<b>DISCLAIMER:</b> This report was generated by an AI system for research "
        "and educational purposes only. It is NOT a certified medical diagnosis. "
        "All findings must be reviewed and validated by a qualified healthcare "
        "professional before any clinical decisions are made.",
        ParagraphStyle("Disclaimer", parent=body_style, fontSize=8,
                       textColor=HexColor("#dc2626"))
    ))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        f"Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"by {config.INSTITUTION_NAME}",
        small_style
    ))

    # --- Build PDF ---
    doc.build(elements)

    return pdf_path
