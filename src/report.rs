//! PDF Report Generator — Electronic Nose Coffee Classification System
//! Generates a structured 3-page PDF report from model training results.

use printpdf::*;
use printpdf::path::{PaintMode, WindingOrder};
use std::fs::File;
use std::io::BufWriter;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::ml::evaluation::EvaluationResults;
use crate::power_monitor::PowerSummary;

// ── A4 page geometry (mm) ─────────────────────────────────────────────────────
const PW: f32 = 210.0;
const PH: f32 = 297.0;
const ML: f32 = 15.0;
const MR: f32 = 15.0;
const CW: f32 = PW - ML - MR; // 180 mm

// ── Colour helpers ────────────────────────────────────────────────────────────
fn c_green() -> Color       { Color::Rgb(Rgb::new(0.133, 0.545, 0.133, None)) }
fn c_dark_green() -> Color  { Color::Rgb(Rgb::new(0.07,  0.37,  0.07,  None)) }
fn c_white() -> Color       { Color::Rgb(Rgb::new(1.0,   1.0,   1.0,   None)) }
fn c_black() -> Color       { Color::Rgb(Rgb::new(0.0,   0.0,   0.0,   None)) }
fn c_dark_gray() -> Color   { Color::Rgb(Rgb::new(0.35,  0.35,  0.35,  None)) }
fn c_mid_gray() -> Color    { Color::Rgb(Rgb::new(0.65,  0.65,  0.65,  None)) }
fn c_light_green() -> Color { Color::Rgb(Rgb::new(0.88,  0.96,  0.88,  None)) }
fn c_blue() -> Color        { Color::Rgb(Rgb::new(0.0,   0.30,  0.70,  None)) }
fn c_orange() -> Color      { Color::Rgb(Rgb::new(0.55,  0.22,  0.0,   None)) }
fn c_red() -> Color         { Color::Rgb(Rgb::new(0.75,  0.0,   0.0,   None)) }
fn c_light_red() -> Color   { Color::Rgb(Rgb::new(0.98,  0.88,  0.88,  None)) }

// ── Report data types ─────────────────────────────────────────────────────────

/// All model types supported for PDF generation.
pub enum TrainingReport {
    RandomForest {
        train_eval:         EvaluationResults,
        val_eval:           EvaluationResults,
        train_accuracy:     f32,
        val_accuracy:       f32,
        n_trees:            usize,
        training_secs:      f64,
        accuracy_curve:     Vec<(usize, f32, f32)>,
        power:              Option<PowerSummary>,
    },
    MLP {
        train_eval:     EvaluationResults,
        val_eval:       EvaluationResults,
        train_accuracy: f32,
        val_accuracy:   f32,
        train_loss:     f32,
        val_loss:       f32,
        n_epochs:       usize,
        training_secs:  f64,
        accuracy_curve: Vec<(usize, f32, f32)>,
        loss_curve:     Vec<(usize, f32, f32)>,
        power:          Option<PowerSummary>,
    },
    SVM {
        train_eval:         EvaluationResults,
        val_eval:           EvaluationResults,
        train_accuracy:     f32,
        val_accuracy:       f32,
        n_epochs:           usize,
        n_support_vectors:  usize,
        training_secs:      f64,
        final_loss:         f32,
        accuracy_curve:     Vec<(usize, f32, f32)>,
        power:              Option<PowerSummary>,
    },
    LSTM {
        train_eval:     EvaluationResults,
        val_eval:       EvaluationResults,
        train_accuracy: f32,
        val_accuracy:   f32,
        train_loss:     f32,
        val_loss:       f32,
        n_epochs:       usize,
        training_secs:  f64,
        accuracy_curve: Vec<(usize, f32, f32)>,
        loss_curve:     Vec<(usize, f32, f32)>,
        power:          Option<PowerSummary>,
    },
    CNN {
        train_eval:     EvaluationResults,
        val_eval:       EvaluationResults,
        train_accuracy: f32,
        val_accuracy:   f32,
        train_loss:     f32,
        val_loss:       f32,
        n_epochs:       usize,
        training_secs:  f64,
        accuracy_curve: Vec<(usize, f32, f32)>,
        loss_curve:     Vec<(usize, f32, f32)>,
        power:          Option<PowerSummary>,
    },
}

impl TrainingReport {
    fn model_name(&self) -> &str {
        match self {
            TrainingReport::RandomForest { .. } => "Random Forest",
            TrainingReport::MLP { .. }          => "Multi-Layer Perceptron (MLP)",
            TrainingReport::SVM { .. }          => "Support Vector Machine (SVM)",
            TrainingReport::LSTM { .. }         => "Long Short-Term Memory (LSTM)",
            TrainingReport::CNN { .. }          => "Convolutional Neural Network (1D-CNN)",
        }
    }
    fn model_short(&self) -> &str {
        match self {
            TrainingReport::RandomForest { .. } => "rf",
            TrainingReport::MLP { .. }          => "mlp",
            TrainingReport::SVM { .. }          => "svm",
            TrainingReport::LSTM { .. }         => "lstm",
            TrainingReport::CNN { .. }          => "cnn",
        }
    }
    fn train_accuracy(&self) -> f32 {
        match self {
            TrainingReport::RandomForest { train_accuracy, .. } |
            TrainingReport::MLP          { train_accuracy, .. } |
            TrainingReport::SVM          { train_accuracy, .. } |
            TrainingReport::LSTM         { train_accuracy, .. } |
            TrainingReport::CNN          { train_accuracy, .. } => *train_accuracy,
        }
    }
    fn val_accuracy(&self) -> f32 {
        match self {
            TrainingReport::RandomForest { val_accuracy, .. } |
            TrainingReport::MLP          { val_accuracy, .. } |
            TrainingReport::SVM          { val_accuracy, .. } |
            TrainingReport::LSTM         { val_accuracy, .. } |
            TrainingReport::CNN          { val_accuracy, .. } => *val_accuracy,
        }
    }
    fn training_secs(&self) -> f32 {
        match self {
            TrainingReport::RandomForest { training_secs, .. } |
            TrainingReport::MLP          { training_secs, .. } |
            TrainingReport::SVM          { training_secs, .. } |
            TrainingReport::LSTM         { training_secs, .. } |
            TrainingReport::CNN          { training_secs, .. } => *training_secs as f32,
        }
    }
    fn train_eval(&self) -> &EvaluationResults {
        match self {
            TrainingReport::RandomForest { train_eval, .. } |
            TrainingReport::MLP          { train_eval, .. } |
            TrainingReport::SVM          { train_eval, .. } |
            TrainingReport::LSTM         { train_eval, .. } |
            TrainingReport::CNN          { train_eval, .. } => train_eval,
        }
    }
    fn val_eval(&self) -> &EvaluationResults {
        match self {
            TrainingReport::RandomForest { val_eval, .. } |
            TrainingReport::MLP          { val_eval, .. } |
            TrainingReport::SVM          { val_eval, .. } |
            TrainingReport::LSTM         { val_eval, .. } |
            TrainingReport::CNN          { val_eval, .. } => val_eval,
        }
    }
    fn power(&self) -> Option<&PowerSummary> {
        match self {
            TrainingReport::RandomForest { power, .. } |
            TrainingReport::MLP          { power, .. } |
            TrainingReport::SVM          { power, .. } |
            TrainingReport::LSTM         { power, .. } |
            TrainingReport::CNN          { power, .. } => power.as_ref(),
        }
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Approximate rendered text width in mm (Helvetica, ~55% char width ratio).
fn aw(text: &str, size: f32) -> f32 {
    text.chars().count() as f32 * size * 0.55 * 0.3528
}

/// Unix timestamp to "YYYY-MM-DD HH:MM:SS"
fn unix_to_date(ts: u64) -> String {
    let h = (ts % 86400) / 3600;
    let m = (ts % 3600) / 60;
    let s = ts % 60;
    let mut days = ts / 86400;
    let mut year = 1970u32;
    loop {
        let dy: u64 = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 366 } else { 365 };
        if days < dy { break; }
        days -= dy;
        year += 1;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let dim: [u64; 12] = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u32;
    let mut day = days + 1;
    for &d in &dim {
        if day <= d { break; }
        day -= d;
        month += 1;
    }
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", year, month, day, h, m, s)
}

/// Select up to `n` evenly-spaced indices (always includes first and last).
#[allow(dead_code)]
fn sample_indices(len: usize, n: usize) -> Vec<usize> {
    if len == 0 { return vec![]; }
    if len <= n  { return (0..len).collect(); }
    let mut v = vec![0usize];
    for i in 1..n - 1 { v.push(i * (len - 1) / (n - 1)); }
    v.push(len - 1);
    v.dedup();
    v
}

// ── Low-level drawing primitives ──────────────────────────────────────────────

fn rect_points(x: f32, y: f32, w: f32, h: f32) -> Vec<(Point, bool)> {
    vec![
        (Point::new(Mm(x),     Mm(y)),     false),
        (Point::new(Mm(x + w), Mm(y)),     false),
        (Point::new(Mm(x + w), Mm(y - h)), false),
        (Point::new(Mm(x),     Mm(y - h)), false),
    ]
}

/// Draw a filled rectangle.
fn fill_rect(l: &PdfLayerReference, x: f32, y: f32, w: f32, h: f32, color: Color) {
    l.set_fill_color(color);
    l.add_polygon(Polygon {
        rings: vec![rect_points(x, y, w, h)],
        mode: PaintMode::Fill,
        winding_order: WindingOrder::NonZero,
    });
}

/// Draw a stroked rectangle (border only).
fn stroke_rect(l: &PdfLayerReference, x: f32, y: f32, w: f32, h: f32,
               color: Color, thick: f32) {
    l.set_outline_color(color);
    l.set_outline_thickness(thick);
    l.add_polygon(Polygon {
        rings: vec![rect_points(x, y, w, h)],
        mode: PaintMode::Stroke,
        winding_order: WindingOrder::NonZero,
    });
}

/// Draw a horizontal line.
fn hline(l: &PdfLayerReference, x1: f32, x2: f32, y: f32, color: Color, thick: f32) {
    l.set_outline_color(color);
    l.set_outline_thickness(thick);
    l.add_line(Line {
        points: vec![
            (Point::new(Mm(x1), Mm(y)), false),
            (Point::new(Mm(x2), Mm(y)), false),
        ],
        is_closed: false,
    });
}

/// Draw a vertical line.
fn vline(l: &PdfLayerReference, x: f32, y1: f32, y2: f32, color: Color, thick: f32) {
    l.set_outline_color(color);
    l.set_outline_thickness(thick);
    l.add_line(Line {
        points: vec![
            (Point::new(Mm(x), Mm(y1)), false),
            (Point::new(Mm(x), Mm(y2)), false),
        ],
        is_closed: false,
    });
}

/// Draw text at absolute position with explicit colour.
fn txt(l: &PdfLayerReference, s: &str, size: f32, x: f32, y: f32,
       font: &IndirectFontRef, color: Color) {
    l.set_fill_color(color);
    l.use_text(s, size, Mm(x), Mm(y), font);
}

/// Draw text horizontally centred on the page.
fn txt_center(l: &PdfLayerReference, s: &str, size: f32, y: f32,
              font: &IndirectFontRef, color: Color) {
    let x = ((PW - aw(s, size)) / 2.0).max(ML);
    txt(l, s, size, x, y, font, color);
}

/// Draw text right-aligned to `rx`.
fn txt_right(l: &PdfLayerReference, s: &str, size: f32, rx: f32, y: f32,
             font: &IndirectFontRef, color: Color) {
    let x = (rx - aw(s, size)).max(ML);
    txt(l, s, size, x, y, font, color);
}

// ── Composite components ──────────────────────────────────────────────────────

fn draw_footer(l: &PdfLayerReference, f: &IndirectFontRef, page: u32, total: u32) {
    hline(l, ML, PW - MR, 14.0, c_mid_gray(), 0.3);
    txt(l, "Electronic Nose Coffee Classification System", 7.0, ML, 10.0, f, c_dark_gray());
    txt_right(l, &format!("Page {} / {}", page, total), 7.0, PW - MR, 10.0, f, c_dark_gray());
}

fn page_subheader(l: &PdfLayerReference, fb: &IndirectFontRef, title: &str) {
    fill_rect(l, 0.0, PH, PW, 18.0, c_dark_green());
    txt_center(l, title, 11.0, PH - 12.0, fb, c_white());
}

/// Section title with green underline. Returns Y below.
fn section_title(l: &PdfLayerReference, fb: &IndirectFontRef, title: &str, y: f32) -> f32 {
    txt(l, title, 10.5, ML, y, fb, c_dark_green());
    let y2 = y - 4.0;
    hline(l, ML, PW - MR, y2, c_green(), 0.6);
    y2 - 3.0
}

/// Coloured metric summary box.
fn metric_box(l: &PdfLayerReference,
              f: &IndirectFontRef, fb: &IndirectFontRef,
              x: f32, y: f32, w: f32, h: f32,
              label: &str, value: &str, color: Color) {
    fill_rect(l, x, y, w, h, color);
    let lx = (x + (w - aw(label, 7.5)) / 2.0).max(x + 1.0);
    txt(l, label, 7.5, lx, y - 7.0, f, c_white());
    let vx = (x + (w - aw(value, 15.0)) / 2.0).max(x + 1.0);
    txt(l, value, 15.0, vx, y - h + 6.0, fb, c_white());
}

/// Per-class metrics table. Returns Y below.
fn eval_table(l: &PdfLayerReference,
              f: &IndirectFontRef, fb: &IndirectFontRef,
              eval: &EvaluationResults, label: &str, y: f32) -> f32 {
    let heading = format!("{} - Accuracy: {:.2}%", label, eval.accuracy * 100.0);
    let y0 = section_title(l, fb, &heading, y);
    let cw = [48.0f32, 66.0, 66.0];
    let tw: f32 = cw.iter().sum();
    let rh = 7.5;

    fill_rect(l, ML, y0, tw, rh, c_dark_green());
    let hdrs = ["Metric", "High Grade (Arabica)", "Low Grade (Robusta)"];
    let mut hx = ML;
    for (&w, h) in cw.iter().zip(hdrs.iter()) {
        txt(l, h, 8.5, hx + 2.0, y0 - rh + 2.5, fb, c_white());
        hx += w;
    }

    let rows: [(&str, String, String); 4] = [
        ("Precision",
         format!("{:.4}", eval.high_grade_metrics.precision),
         format!("{:.4}", eval.low_grade_metrics.precision)),
        ("Recall",
         format!("{:.4}", eval.high_grade_metrics.recall),
         format!("{:.4}", eval.low_grade_metrics.recall)),
        ("F1-Score",
         format!("{:.4}", eval.high_grade_metrics.f1_score),
         format!("{:.4}", eval.low_grade_metrics.f1_score)),
        ("Support (n samples)",
         format!("{}", eval.high_grade_metrics.support),
         format!("{}", eval.low_grade_metrics.support)),
    ];

    let mut ry = y0 - rh;
    for (i, (metric, high, low)) in rows.iter().enumerate() {
        let bg = if i % 2 == 0 { c_light_green() } else { c_white() };
        fill_rect(l, ML, ry, tw, rh, bg);
        let mut rx = ML;
        txt(l, metric, 8.5, rx + 2.0, ry - rh + 2.5, fb, c_dark_green()); rx += cw[0];
        txt(l, high,   8.5, rx + 2.0, ry - rh + 2.5, f,  c_black());      rx += cw[1];
        txt(l, low,    8.5, rx + 2.0, ry - rh + 2.5, f,  c_black());
        let _ = rx;
        ry -= rh;
    }

    stroke_rect(l, ML, y0, tw, rh * 5.0, c_dark_gray(), 0.3);
    let mut vx = ML;
    for &w in &cw[..cw.len() - 1] {
        vx += w;
        vline(l, vx, y0, y0 - rh * 5.0, c_dark_gray(), 0.3);
    }
    for i in 1..5 { hline(l, ML, ML + tw, y0 - rh * i as f32, c_dark_gray(), 0.3); }

    ry - 4.0
}

/// 2x2 confusion matrix. Returns Y below.
fn confusion_matrix_grid(l: &PdfLayerReference,
                         f: &IndirectFontRef, fb: &IndirectFontRef,
                         matrix: [[usize; 2]; 2], y: f32) -> f32 {
    let y0 = section_title(l, fb, "Confusion Matrix (Validation Data)", y);
    let cw = 40.0f32;
    let ch = 14.0f32;
    let ox = ML + 5.0;

    fill_rect(l, ox + cw,       y0, cw, ch, c_dark_green());
    fill_rect(l, ox + cw * 2.0, y0, cw, ch, c_dark_green());
    txt(l, "Predicted: High", 7.5, ox + cw + 2.0,       y0 - ch + 3.5, fb, c_white());
    txt(l, "Predicted: Low",  7.5, ox + cw * 2.0 + 2.0, y0 - ch + 3.5, fb, c_white());

    fill_rect(l, ox, y0 - ch, cw, ch, c_dark_green());
    txt(l, "Actual: High", 7.5, ox + 2.0, y0 - ch * 2.0 + 3.5, fb, c_white());
    fill_rect(l, ox + cw, y0 - ch, cw, ch, c_light_green());
    let tp = format!("{}", matrix[0][0]);
    txt(l, &tp, 14.0, ox + cw + (cw - aw(&tp, 14.0)) / 2.0, y0 - ch * 1.5 - 2.0, fb, c_dark_green());
    fill_rect(l, ox + cw * 2.0, y0 - ch, cw, ch, c_light_red());
    let fn_v = format!("{}", matrix[0][1]);
    txt(l, &fn_v, 14.0, ox + cw * 2.0 + (cw - aw(&fn_v, 14.0)) / 2.0, y0 - ch * 1.5 - 2.0, fb, c_red());

    fill_rect(l, ox, y0 - ch * 2.0, cw, ch, c_dark_green());
    txt(l, "Actual: Low", 7.5, ox + 2.0, y0 - ch * 3.0 + 3.5, fb, c_white());
    fill_rect(l, ox + cw, y0 - ch * 2.0, cw, ch, c_light_red());
    let fp_v = format!("{}", matrix[1][0]);
    txt(l, &fp_v, 14.0, ox + cw + (cw - aw(&fp_v, 14.0)) / 2.0, y0 - ch * 2.5 - 2.0, fb, c_red());
    fill_rect(l, ox + cw * 2.0, y0 - ch * 2.0, cw, ch, c_light_green());
    let tn = format!("{}", matrix[1][1]);
    txt(l, &tn, 14.0, ox + cw * 2.0 + (cw - aw(&tn, 14.0)) / 2.0, y0 - ch * 2.5 - 2.0, fb, c_dark_green());

    stroke_rect(l, ox, y0, cw * 3.0, ch * 3.0, c_dark_gray(), 0.4);
    vline(l, ox + cw,       y0, y0 - ch * 3.0, c_dark_gray(), 0.3);
    vline(l, ox + cw * 2.0, y0, y0 - ch * 3.0, c_dark_gray(), 0.3);
    hline(l, ox, ox + cw * 3.0, y0 - ch,       c_dark_gray(), 0.3);
    hline(l, ox, ox + cw * 3.0, y0 - ch * 2.0, c_dark_gray(), 0.3);

    let ly = y0 - ch * 3.0 - 5.0;
    txt(l, "TP = True Positive  (High predicted as High)",  7.5, ML,        ly,       f, c_dark_gray());
    txt(l, "TN = True Negative  (Low predicted as Low)",    7.5, ML + 92.0, ly,       f, c_dark_gray());
    txt(l, "FP = False Positive (Low predicted as High)",   7.5, ML,        ly - 5.0, f, c_dark_gray());
    txt(l, "FN = False Negative (High predicted as Low)",   7.5, ML + 92.0, ly - 5.0, f, c_dark_gray());

    ly - 9.0
}

/// Generic ranked table with optional bar-chart column (bar_col = column index + max value).
fn ranked_table(l: &PdfLayerReference,
                f: &IndirectFontRef, fb: &IndirectFontRef,
                headers: &[&str], widths: &[f32],
                rows: &[Vec<String>],
                bar_col: Option<(usize, f32)>,
                y: f32) -> f32 {
    let tw: f32 = widths.iter().sum();
    let rh = 7.0;

    fill_rect(l, ML, y, tw, rh, c_dark_green());
    let mut hx = ML;
    for (&w, h) in widths.iter().zip(headers.iter()) {
        txt(l, h, 8.5, hx + 2.0, y - rh + 2.5, fb, c_white());
        hx += w;
    }

    let max_bar = bar_col.map(|(_, mv)| mv).unwrap_or(1.0f32);
    let bar_idx = bar_col.map(|(i, _)| i);
    let mut ry = y - rh;

    for (i, row) in rows.iter().enumerate() {
        let bg = if i % 2 == 0 { c_light_green() } else { c_white() };
        fill_rect(l, ML, ry, tw, rh, bg);
        let mut rx = ML;
        for (ci, (&w, cell)) in widths.iter().zip(row.iter()).enumerate() {
            if bar_idx == Some(ci) {
                if let Ok(val) = cell.parse::<f32>() {
                    let bar_px = (w - 28.0).max(5.0);
                    let bw = if max_bar > 0.0 { bar_px * (val / max_bar) } else { 0.0 };
                    fill_rect(l, rx + 2.0, ry - 1.5, bw, rh - 3.0, c_green());
                    txt(l, &format!("{:.4}", val), 6.5, rx + bw + 4.0, ry - rh + 2.5, f, c_dark_gray());
                }
            } else {
                txt(l, cell, 8.5, rx + 2.0, ry - rh + 2.5, f, c_black());
            }
            rx += w;
        }
        let _ = rx;
        ry -= rh;
    }

    stroke_rect(l, ML, y, tw, rh * (rows.len() + 1) as f32, c_dark_gray(), 0.3);
    let mut vx = ML;
    for &w in &widths[..widths.len() - 1] {
        vx += w;
        vline(l, vx, y, y - rh * (rows.len() + 1) as f32, c_dark_gray(), 0.3);
    }
    for i in 1..=rows.len() {
        hline(l, ML, ML + tw, y - rh * i as f32, c_dark_gray(), 0.3);
    }
    ry - 4.0
}

/// 2-column key-value info table. Returns Y below.
fn kv_table(l: &PdfLayerReference,
            f: &IndirectFontRef, fb: &IndirectFontRef,
            rows: &[(&str, String)], y: f32) -> f32 {
    let rh = 7.5;
    let c0 = 72.0;
    let tw = CW;
    for (i, (k, v)) in rows.iter().enumerate() {
        let ry = y - i as f32 * rh;
        let bg = if i % 2 == 0 { c_light_green() } else { c_white() };
        fill_rect(l, ML, ry, tw, rh, bg);
        txt(l, k, 8.5, ML + 2.0, ry - rh + 2.5, fb, c_dark_green());
        txt(l, v, 8.5, ML + c0,  ry - rh + 2.5, f,  c_black());
    }
    stroke_rect(l, ML, y, tw, rh * rows.len() as f32, c_dark_gray(), 0.3);
    vline(l, ML + c0, y, y - rh * rows.len() as f32, c_dark_gray(), 0.3);
    for i in 1..rows.len() {
        hline(l, ML, ML + tw, y - rh * i as f32, c_dark_gray(), 0.3);
    }
    y - rh * rows.len() as f32 - 4.0
}

/// Accuracy checkpoint table (10% interval sampling). Returns Y below.
fn accuracy_checkpoint_table(l: &PdfLayerReference,
                             f: &IndirectFontRef, fb: &IndirectFontRef,
                             x_label: &str, curve: &[(usize, f32, f32)], y: f32) -> f32 {
    if curve.is_empty() { return y; }
    let y0 = section_title(l, fb, "Accuracy Curve (Checkpoint)", y);
    let widths = [28.0f32, 76.0, 76.0];
    let hdrs = [x_label, "Training Accuracy (%)", "Validation Accuracy (%)"];
    
    // Limit to max 100 epochs
    let _max_epoch = curve.iter().map(|(x, _, _)| *x).max().unwrap_or(1);
    let filtered_curve: Vec<_> = curve.iter()
        .filter(|(x, _, _)| *x <= 100)
        .collect();
    
    if filtered_curve.is_empty() { return y0 - 5.0; }
    
    // Take every 5 epochs (checkpoint sampling)
    let rows: Vec<Vec<String>> = filtered_curve.iter()
        .filter(|(epoch, _, _)| epoch % 5 == 1 || *epoch == 100)
        .map(|(x, ta, va)| {
            vec![format!("{}", x), format!("{:.2}", ta * 100.0), format!("{:.2}", va * 100.0)]
        })
        .collect();
    
    ranked_table(l, f, fb, &hdrs, &widths, &rows, None, y0)
}

/// Accuracy vs X curve table. Returns Y below.
#[allow(dead_code)]
fn accuracy_curve_table(l: &PdfLayerReference,
                        f: &IndirectFontRef, fb: &IndirectFontRef,
                        x_label: &str, curve: &[(usize, f32, f32)], y: f32) -> f32 {
    if curve.is_empty() { return y; }
    let y0 = section_title(l, fb, "Accuracy Curve (All Epochs)", y);
    let widths = [28.0f32, 76.0, 76.0];
    let hdrs = [x_label, "Training Accuracy (%)", "Validation Accuracy (%)"];
    // Limit to first 100 epochs
    let rows: Vec<Vec<String>> = curve.iter()
        .filter(|(x, _, _)| *x <= 100)
        .map(|(x, ta, va)| {
            vec![format!("{}", x), format!("{:.2}", ta * 100.0), format!("{:.2}", va * 100.0)]
        }).collect();
    ranked_table(l, f, fb, &hdrs, &widths, &rows, None, y0)
}

/// Loss checkpoint curve table (10% interval sampling). Returns Y below.
fn loss_checkpoint_table(l: &PdfLayerReference,
                         f: &IndirectFontRef, fb: &IndirectFontRef,
                         curve: &[(usize, f32, f32)], y: f32) -> f32 {
    if curve.is_empty() { return y; }
    let y0 = section_title(l, fb, "Loss Curve (Checkpoint)", y);
    let widths = [28.0f32, 76.0, 76.0];
    let hdrs = ["Epoch", "Training Loss", "Validation Loss"];
    
    // Limit to max 100 epochs
    let filtered_curve: Vec<_> = curve.iter()
        .filter(|(ep, _, _)| *ep <= 100)
        .collect();
    
    if filtered_curve.is_empty() { return y0 - 5.0; }
    
    // Take every 5 epochs (checkpoint sampling)
    let rows: Vec<Vec<String>> = filtered_curve.iter()
        .filter(|(epoch, _, _)| epoch % 5 == 1 || *epoch == 100)
        .map(|(ep, tl, vl)| {
            vec![format!("{}", ep), format!("{:.6}", tl), format!("{:.6}", vl)]
        })
        .collect();
    
    ranked_table(l, f, fb, &hdrs, &widths, &rows, None, y0)
}

/// Loss curve table. Returns Y below.
#[allow(dead_code)]
fn loss_curve_table(l: &PdfLayerReference,
                    f: &IndirectFontRef, fb: &IndirectFontRef,
                    curve: &[(usize, f32, f32)], y: f32) -> f32 {
    if curve.is_empty() { return y; }
    let y0 = section_title(l, fb, "Loss Curve (All Epochs)", y);
    let widths = [28.0f32, 76.0, 76.0];
    let hdrs = ["Epoch", "Training Loss", "Validation Loss"];
    // Limit to first 100 epochs
    let rows: Vec<Vec<String>> = curve.iter()
        .filter(|(ep, _, _)| *ep <= 100)
        .map(|(ep, tl, vl)| {
            vec![format!("{}", ep), format!("{:.6}", tl), format!("{:.6}", vl)]
        }).collect();
    ranked_table(l, f, fb, &hdrs, &widths, &rows, None, y0)
}

// ── Page builders ─────────────────────────────────────────────────────────────

/// Power consumption section. Returns Y below.
fn draw_power_section(l: &PdfLayerReference,
                      f: &IndirectFontRef, fb: &IndirectFontRef,
                      ps: &PowerSummary, y: f32) -> f32 {
    let y0 = section_title(l, fb, "POWER CONSUMPTION DURING TRAINING", y);
    let y0 = y0 - 2.0;

    // 4-row x 4-column layout: [label | value | label | value]
    let cw = [62.0f32, 28.0, 62.0, 28.0];
    let tw: f32 = cw.iter().sum(); // = 180 mm = CW
    let rh = 7.5f32;

    let energy_wh = ps.energy_joules / 3600.0;
    let dur_str = if ps.duration_secs < 60.0 {
        format!("{:.1} s", ps.duration_secs)
    } else {
        format!("{:.1} min", ps.duration_secs / 60.0)
    };

    let rows: [(&str, String, &str, String); 4] = [
        ("Avg Total Power",    format!("{:.2} W",   ps.avg_total_w),
         "Avg CPU Temp",       format!("{:.1} C",  ps.avg_cpu_temp)),
        ("Peak Total Power",   format!("{:.2} W",   ps.peak_total_w),
         "Peak CPU Temp",      format!("{:.1} C",  ps.peak_cpu_temp)),
        ("Avg CPU+GPU Power",  format!("{:.2} W",   ps.avg_cpu_gpu_w),
         "Energy Used",        format!("{:.1} J  ({:.4} Wh)", ps.energy_joules, energy_wh)),
        ("Peak CPU+GPU Power", format!("{:.2} W",   ps.peak_cpu_gpu_w),
         "Training Duration",  dur_str),
    ];

    // Header bar
    fill_rect(l, ML, y0, tw, rh, c_orange());
    let hdrs = ["Power Metric", "Value", "Thermal / Energy", "Value"];
    let mut hx = ML;
    for (&w, h) in cw.iter().zip(hdrs.iter()) {
        txt(l, h, 8.5, hx + 2.0, y0 - rh + 2.5, fb, c_white());
        hx += w;
    }

    // Data rows
    let mut ry = y0 - rh;
    for (i, (l1, v1, l2, v2)) in rows.iter().enumerate() {
        let bg = if i % 2 == 0 { c_light_green() } else { c_white() };
        fill_rect(l, ML, ry, tw, rh, bg);
        let mut rx = ML;
        txt(l, l1, 8.5, rx + 2.0, ry - rh + 2.5, fb, c_dark_green()); rx += cw[0];
        txt(l, v1, 8.5, rx + 2.0, ry - rh + 2.5, f,  c_black());      rx += cw[1];
        txt(l, l2, 8.5, rx + 2.0, ry - rh + 2.5, fb, c_dark_green()); rx += cw[2];
        txt(l, v2, 8.5, rx + 2.0, ry - rh + 2.5, f,  c_black());
        let _ = rx;
        ry -= rh;
    }

    // Grid lines
    stroke_rect(l, ML, y0, tw, rh * 5.0, c_dark_gray(), 0.3);
    let mut vx = ML;
    for &w in &cw[..3] {
        vx += w;
        vline(l, vx, y0, y0 - rh * 5.0, c_dark_gray(), 0.3);
    }
    for i in 1..5 {
        hline(l, ML, ML + tw, y0 - rh * i as f32, c_dark_gray(), 0.3);
    }

    ry - 4.0
}


fn draw_page1(l: &PdfLayerReference,
              f: &IndirectFontRef, fb: &IndirectFontRef,
              report: &TrainingReport, ts: u64) {
    fill_rect(l, 0.0, PH, PW, 5.0, c_dark_green());
    fill_rect(l, 0.0, PH - 5.0, PW, 58.0, c_green());
    txt_center(l, "MODEL TRAINING REPORT", 20.0, PH - 20.0, fb, c_white());
    txt_center(l, "Electronic Nose Coffee Classification System", 10.0, PH - 31.0, f, c_white());
    txt_center(l, &format!("Model: {}", report.model_name()), 13.0, PH - 43.0, fb, c_white());
    txt_center(l, &format!("Date: {}", unix_to_date(ts)), 8.5, PH - 54.0, f, c_white());

    let mut y = PH - 72.0;
    y = section_title(l, fb, "TRAINING SUMMARY", y);
    y -= 3.0;

    let bw = (CW - 10.0) / 3.0;
    let bh = 26.0;
    let secs = report.training_secs();
    let time_ms = secs * 1000.0;
    let time_str = format!("{:.2} ms", time_ms);
    metric_box(l, f, fb, ML,                    y, bw, bh, "Training Accuracy",
               &format!("{:.1}%", report.train_accuracy() * 100.0), c_dark_green());
    metric_box(l, f, fb, ML + bw + 5.0,         y, bw, bh, "Validation Accuracy",
               &format!("{:.1}%", report.val_accuracy() * 100.0), c_blue());
    metric_box(l, f, fb, ML + (bw + 5.0) * 2.0, y, bw, bh, "Training Time",
               &time_str, c_orange());
    y = y - bh - 8.0;

    y = section_title(l, fb, "MODEL CONFIGURATION", y);
    y -= 2.0;

    let cfg: Vec<(&str, String)> = match report {
        TrainingReport::RandomForest { n_trees, .. } => vec![
            ("Model Type",          "Random Forest".to_string()),
            ("Number of Trees",     format!("{}", n_trees)),
            ("Data Split",          "80% Training / 20% Validation".to_string()),
            ("Extracted Features",  "mean, std, min, max, range, median".to_string()),
            ("Feature Dimensions",  "6 statistics x 8 sensors = 48 features".to_string()),
        ],
        TrainingReport::MLP { n_epochs, .. } => vec![
            ("Model Type",    "Multi-Layer Perceptron (MLP)".to_string()),
            ("Epochs",        format!("{}", n_epochs)),
            ("Data Split",    "80% Training / 20% Validation".to_string()),
            ("Architecture",  "Input -> Hidden Layers -> Sigmoid -> Output".to_string()),
            ("Input Shape",   "8 sensors x 300 timesteps (flattened -> 2400)".to_string()),
        ],
        TrainingReport::SVM { n_epochs, n_support_vectors, final_loss, .. } => vec![
            ("Model Type",          "Support Vector Machine (SVM)".to_string()),
            ("Epochs",              format!("{}", n_epochs)),
            ("Support Vectors",     format!("{}", n_support_vectors)),
            ("Final Training Loss", format!("{:.5}", final_loss)),
            ("Data Split",          "80% Training / 20% Validation".to_string()),
        ],
        TrainingReport::LSTM { n_epochs, .. } => vec![
            ("Model Type",      "Long Short-Term Memory (LSTM)".to_string()),
            ("Epochs",          format!("{}", n_epochs)),
            ("Data Split",      "80% Training / 20% Validation".to_string()),
            ("Input Shape",     "8 sensors x 300 timesteps".to_string()),
            ("Architecture",    "Recurrent Neural Network (LSTM)".to_string()),
        ],
        TrainingReport::CNN { n_epochs, .. } => vec![
            ("Model Type",      "1D Convolutional Neural Network (1D-CNN)".to_string()),
            ("Epochs",          format!("{}", n_epochs)),
            ("Data Split",      "80% Training / 20% Validation".to_string()),
            ("Input Shape",     "8 sensors x 300 timesteps".to_string()),
            ("Architecture",    "1D-CNN + Backpropagation (SGD, Cross Entropy)".to_string()),
        ],
    };

    let rh = 7.5;
    let c0 = 68.0;
    for (i, (k, v)) in cfg.iter().enumerate() {
        let ry = y - i as f32 * rh;
        let bg = if i % 2 == 0 { c_light_green() } else { c_white() };
        fill_rect(l, ML, ry, CW, rh, bg);
        txt(l, k, 8.5, ML + 2.0, ry - rh + 2.5, fb, c_dark_green());
        txt(l, v, 8.5, ML + c0,  ry - rh + 2.5, f,  c_black());
    }
    stroke_rect(l, ML, y, CW, rh * cfg.len() as f32, c_dark_gray(), 0.3);
    vline(l, ML + c0, y, y - rh * cfg.len() as f32, c_dark_gray(), 0.3);
    for i in 1..cfg.len() { hline(l, ML, ML + CW, y - rh * i as f32, c_dark_gray(), 0.3); }

    let y2 = y - rh * cfg.len() as f32 - 8.0;
    let y3 = section_title(l, fb, "GAS SENSORS USED (8 Sensors)", y2);
    let sensors = ["TGS2600", "MQ135", "MQ3", "MQ6", "MQ7", "TGS2602", "TGS2611", "TGS2620"];
    for (i, s) in sensors.iter().enumerate() {
        let bx = ML + (i % 4) as f32 * 46.0;
        let by = y3 - 2.0 - (i / 4) as f32 * 10.0;
        fill_rect(l, bx, by, 43.0, 8.0, c_dark_green());
        txt(l, s, 8.5, bx + 3.0, by - 5.5, fb, c_white());
    }

    // Power section - rendered below the sensors if data is available
    let y_after_sensors = y3 - 28.0; // 2 sensor rows (10mm each) + 8mm gap
    if let Some(ps) = report.power() {
        draw_power_section(l, f, fb, ps, y_after_sensors);
    }

    draw_footer(l, f, 1, 3);
}

fn draw_page2(l: &PdfLayerReference,
              f: &IndirectFontRef, fb: &IndirectFontRef,
              report: &TrainingReport) {
    page_subheader(l, fb, "MODEL EVALUATION RESULTS");
    let mut y = PH - 25.0;
    y = eval_table(l, f, fb, report.train_eval(), "Training Data", y);
    y -= 8.0;
    y = eval_table(l, f, fb, report.val_eval(), "Validation Data", y);
    y -= 8.0;
    confusion_matrix_grid(l, f, fb, report.val_eval().confusion_matrix, y);
    draw_footer(l, f, 2, 3);
}

fn draw_page3(l: &PdfLayerReference,
              f: &IndirectFontRef, fb: &IndirectFontRef,
              report: &TrainingReport) {
    page_subheader(l, fb, "MODEL DETAILS & TRAINING CURVES");
    let mut y = PH - 25.0;

    match report {
        TrainingReport::RandomForest { n_trees, train_accuracy, val_accuracy, training_secs,
                                       accuracy_curve, .. } => {
            y = section_title(l, fb, &format!("Random Forest Summary -- {} Trees", n_trees), y);
            y -= 2.0;
            let training_ms = training_secs * 1000.0;
            let summary = [
                ("Training Accuracy",   format!("{:.6}", train_accuracy)),
                ("Validation Accuracy", format!("{:.6}", val_accuracy)),
                ("Total Trees",         format!("{}", n_trees)),
                ("Training Time",       format!("{:.2} ms", training_ms)),
            ];
            y = kv_table(l, f, fb, &summary, y);
            y -= 6.0;
            
            accuracy_checkpoint_table(l, f, fb, "N Trees", accuracy_curve, y);
        }

        TrainingReport::MLP { n_epochs, train_loss, val_loss, training_secs: _,
                              accuracy_curve, loss_curve, .. } => {
            y = section_title(l, fb, &format!("MLP Summary -- {} Epochs", n_epochs), y);
            y -= 2.0;
            let summary = [
                ("Train Loss",      format!("{:.6}", train_loss)),
                ("Validation Loss", format!("{:.6}", val_loss)),
                ("Total Epochs",    format!("{}", n_epochs)),
            ];
            y = kv_table(l, f, fb, &summary, y);
            y -= 6.0;
            y = accuracy_checkpoint_table(l, f, fb, "Epoch", accuracy_curve, y);
            y -= 6.0;
            loss_checkpoint_table(l, f, fb, loss_curve, y);
        }

        TrainingReport::SVM { n_epochs, n_support_vectors, final_loss, training_secs: _,
                              accuracy_curve, .. } => {
            y = section_title(l, fb, &format!("SVM Summary -- {} Epochs", n_epochs), y);
            y -= 2.0;
            let summary = [
                ("Train Loss",      format!("{:.6}", final_loss)),
                ("Support Vectors", format!("{}", n_support_vectors)),
                ("Total Epochs",    format!("{}", n_epochs)),
            ];
            y = kv_table(l, f, fb, &summary, y);
            y -= 6.0;

            accuracy_checkpoint_table(l, f, fb, "Epoch", accuracy_curve, y);
        }

        TrainingReport::LSTM { n_epochs, train_loss, val_loss, training_secs: _,
                               accuracy_curve, loss_curve, .. } => {
            y = section_title(l, fb, &format!("LSTM Summary -- {} Epochs", n_epochs), y);
            y -= 2.0;
            let summary = [
                ("Train Loss",      format!("{:.6}", train_loss)),
                ("Validation Loss", format!("{:.6}", val_loss)),
                ("Total Epochs",    format!("{}", n_epochs)),
            ];
            y = kv_table(l, f, fb, &summary, y);
            y -= 6.0;
            y = accuracy_checkpoint_table(l, f, fb, "Epoch", accuracy_curve, y);
            y -= 6.0;
            loss_checkpoint_table(l, f, fb, loss_curve, y);
        }

        TrainingReport::CNN { n_epochs, train_loss, val_loss, training_secs: _,
                              accuracy_curve, loss_curve, .. } => {
            y = section_title(l, fb, &format!("1D-CNN Summary -- {} Epochs", n_epochs), y);
            y -= 2.0;
            let summary = [
                ("Train Loss",      format!("{:.6}", train_loss)),
                ("Validation Loss", format!("{:.6}", val_loss)),
                ("Total Epochs",    format!("{}", n_epochs)),
            ];
            y = kv_table(l, f, fb, &summary, y);
            y -= 6.0;
            y = accuracy_checkpoint_table(l, f, fb, "Epoch", accuracy_curve, y);
            y -= 6.0;
            loss_checkpoint_table(l, f, fb, loss_curve, y);
        }
    }

    draw_footer(l, f, 3, 3);
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Generate a structured 3-page A4 PDF training report.
/// Returns the output file path on success, or an error message on failure.
pub fn generate_training_pdf(report: &TrainingReport, output_dir: &str)
    -> Result<String, String>
{
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = format!("{}/{}_report_{}.pdf", output_dir, report.model_short(), ts);

    let (doc, p1, l1) = PdfDocument::new(
        format!("{} Training Report", report.model_name()),
        Mm(PW), Mm(PH), "Layer 1",
    );
    let f  = doc.add_builtin_font(BuiltinFont::Helvetica)    .map_err(|e| e.to_string())?;
    let fb = doc.add_builtin_font(BuiltinFont::HelveticaBold) .map_err(|e| e.to_string())?;

    draw_page1(&doc.get_page(p1).get_layer(l1), &f, &fb, report, ts);

    let (p2, l2) = doc.add_page(Mm(PW), Mm(PH), "Layer 1");
    draw_page2(&doc.get_page(p2).get_layer(l2), &f, &fb, report);

    let (p3, l3) = doc.add_page(Mm(PW), Mm(PH), "Layer 1");
    draw_page3(&doc.get_page(p3).get_layer(l3), &f, &fb, report);

    std::fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    doc.save(&mut BufWriter::new(
        File::create(&path).map_err(|e| e.to_string())?
    )).map_err(|e| e.to_string())?;

    Ok(path)
}
