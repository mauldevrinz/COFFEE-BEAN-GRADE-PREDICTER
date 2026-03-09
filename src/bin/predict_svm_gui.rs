//! SVM Coffee Grade Predictor GUI
//! Prediction interface using pre-trained SVM model (svm_model.json)

use coffee_classifier::ml::{CoffeeDataLoader, NormalizationStats, CoffeeSVM};
use eframe::egui;
use egui::{Color32, RichText};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════

#[derive(Debug, Clone)]
enum PredictionState {
    Idle,
    Loading(Instant),
    Success(PredictionResult),
    Error(String),
}

#[derive(Debug, Clone)]
struct PredictionResult {
    predicted_class: String,
    high_grade_prob: f32,
    low_grade_prob: f32,
    confidence: f32,
    prediction_time: Duration,
}

// ═══════════════════════════════════════════════════════
// MAIN GUI APPLICATION
// ═══════════════════════════════════════════════════════

struct SVMPredictorApp {
    file_path: String,
    state: PredictionState,
    norm_stats: Option<NormalizationStats>,
    model: Option<CoffeeSVM>,
    rx: Option<Receiver<Result<PredictionResult, String>>>,
    model_status: String,
}

impl Default for SVMPredictorApp {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            state: PredictionState::Idle,
            norm_stats: None,
            model: None,
            rx: None,
            model_status: String::new(),
        }
    }
}

impl SVMPredictorApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::light());
        setup_fonts(&cc.egui_ctx);

        let mut app = Self::default();
        let mut status_parts = Vec::new();

        if let Ok(stats) = load_normalization_stats("models/normalization_stats.json") {
            app.norm_stats = Some(stats);
            println!("✅ Normalization stats loaded");
            status_parts.push("✅ Norm stats");
        } else {
            println!("⚠️ Failed to load normalization stats");
            status_parts.push("❌ Norm stats missing");
        }

        if let Ok(model) = CoffeeSVM::load("models/svm_model.json") {
            println!("✅ SVM model loaded");
            status_parts.push("✅ SVM model");
            app.model = Some(model);
        } else {
            println!("⚠️ Failed to load SVM model");
            status_parts.push("❌ SVM model missing");
        }

        app.model_status = status_parts.join("  |  ");
        app
    }

    fn predict(&mut self) {
        if self.file_path.is_empty() {
            self.state = PredictionState::Error("Please select a CSV file first!".to_string());
            return;
        }

        let path = PathBuf::from(&self.file_path);
        if !path.exists() {
            self.state = PredictionState::Error(format!("File not found: {}", self.file_path));
            return;
        }

        if self.norm_stats.is_none() || self.model.is_none() {
            self.state =
                PredictionState::Error("Model or normalization stats not loaded!".to_string());
            return;
        }

        let start_time = Instant::now();
        self.state = PredictionState::Loading(start_time);

        let file_path = self.file_path.clone();
        let norm_stats = self.norm_stats.clone().unwrap();
        let model = self.model.clone().unwrap();

        let (tx, rx) = channel();
        self.rx = Some(rx);

        thread::spawn(move || {
            let result = run_prediction(&file_path, &norm_stats, &model, start_time);
            let _ = tx.send(result);
        });
    }
}

impl eframe::App for SVMPredictorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(rx) = &self.rx {
            if let Ok(result) = rx.try_recv() {
                match result {
                    Ok(pred) => self.state = PredictionState::Success(pred),
                    Err(e) => self.state = PredictionState::Error(e),
                }
                self.rx = None;
            }
        }

        // Header panel
        egui::TopBottomPanel::top("svm_predict_header")
            .exact_height(70.0)
            .frame(egui::Frame::none().fill(Color32::from_rgb(0, 120, 200)))
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(12.0);
                    ui.label(
                        RichText::new("☕  SVM Coffee Grade Predictor")
                            .size(22.0)
                            .strong()
                            .color(Color32::WHITE),
                    );
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("Predict coffee quality: High Grade or Low Grade")
                            .size(12.0)
                            .color(Color32::from_rgba_unmultiplied(255, 255, 255, 180)),
                    );
                });
            });

        // Status bar
        egui::TopBottomPanel::top("svm_predict_status")
            .exact_height(28.0)
            .frame(egui::Frame::none().fill(Color32::from_rgb(240, 245, 255)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(12.0);
                    ui.label(
                        RichText::new(&self.model_status)
                            .size(11.0)
                            .color(Color32::DARK_GRAY),
                    );
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(20.0);

            // ── File Picker ─────────────────────────────────────
            egui::Frame::none()
                .fill(Color32::from_rgb(245, 248, 255))
                .rounding(egui::Rounding::same(8.0))
                .inner_margin(egui::Margin::same(16.0))
                .stroke(egui::Stroke::new(1.0, Color32::from_rgb(180, 210, 240)))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new("📂  Select CSV File")
                            .size(14.0)
                            .strong()
                            .color(Color32::from_rgb(0, 120, 200)),
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        let te = egui::TextEdit::singleline(&mut self.file_path)
                            .desired_width(ui.available_width() - 120.0)
                            .hint_text("Select sensor CSV file...");
                        ui.add(te);
                        ui.add_space(8.0);
                        if ui
                            .add(
                                egui::Button::new(RichText::new("📁 Browse").size(13.0))
                                    .fill(Color32::from_rgb(0, 120, 200))
                                    .min_size(egui::vec2(90.0, 30.0)),
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("CSV", &["csv"])
                                .pick_file()
                            {
                                self.file_path = path.display().to_string();
                                self.state = PredictionState::Idle;
                            }
                        }
                    });
                });

            ui.add_space(16.0);

            // ── Predict Button ───────────────────────────────────
            ui.vertical_centered(|ui| {
                let btn_size = egui::vec2(200.0, 46.0);
                let (rect, response) = ui.allocate_exact_size(btn_size, egui::Sense::click());
                let is_loading = matches!(self.state, PredictionState::Loading(_));
                let is_hover = response.hovered();

                let bg = if is_loading {
                    Color32::from_rgb(100, 100, 100)
                } else if is_hover {
                    Color32::from_rgb(0, 100, 180)
                } else {
                    Color32::from_rgb(0, 120, 200)
                };

                ui.painter().rect_filled(rect, 10.0, bg);

                let label = if is_loading {
                    "⏳  Predicting...".to_string()
                } else {
                    "🔍  Predict".to_string()
                };

                let galley = ui.painter().layout_no_wrap(
                    label,
                    egui::FontId::new(15.0, egui::FontFamily::Name("PoppinsBold".into())),
                    Color32::WHITE,
                );
                let text_pos = rect.center() - galley.size() / 2.0;
                ui.painter().galley(text_pos, galley, Color32::WHITE);

                if response.clicked() && !is_loading {
                    self.predict();
                }
            });

            ui.add_space(24.0);
            ui.separator();
            ui.add_space(16.0);

            // ── Results ─────────────────────────────────────────
            match &self.state.clone() {
                PredictionState::Idle => {
                    ui.vertical_centered(|ui| {
                        ui.add_space(40.0);
                        ui.label(
                            RichText::new("📊  Select a CSV file and click Predict")
                                .size(15.0)
                                .color(Color32::GRAY),
                        );
                        ui.add_space(8.0);
                        ui.label(
                            RichText::new("Model: SVM  •  Classes: High Grade / Low Grade")
                                .size(12.0)
                                .color(Color32::from_rgb(150, 170, 200)),
                        );
                    });
                }
                PredictionState::Loading(start_time) => {
                    ui.vertical_centered(|ui| {
                        ui.add_space(40.0);
                        ui.spinner();
                        ui.add_space(10.0);
                        let elapsed = start_time.elapsed();
                        let s = elapsed.as_secs();
                        let ms = elapsed.subsec_millis();
                        ui.label(
                            RichText::new(format!("Analyzing... {}.{:03} seconds", s, ms))
                                .size(15.0)
                                .color(Color32::from_rgb(0, 120, 200)),
                        );
                    });
                    ctx.request_repaint();
                }
                PredictionState::Error(msg) => {
                    ui.vertical_centered(|ui| {
                        ui.add_space(30.0);
                        ui.label(
                            RichText::new(format!("❌  Error: {}", msg))
                                .size(14.0)
                                .color(Color32::RED),
                        );
                    });
                }
                PredictionState::Success(result) => {
                    display_results(ui, result);
                }
            }
        });

        if matches!(self.state, PredictionState::Loading(_)) {
            ctx.request_repaint();
        }
    }
}

// ═══════════════════════════════════════════════════════
// RESULTS DISPLAY
// ═══════════════════════════════════════════════════════

fn display_results(ui: &mut egui::Ui, result: &PredictionResult) {
    // Prediction time
    ui.horizontal(|ui| {
        ui.label(RichText::new("⏱️  Prediction Time:").size(13.0).strong());
        let s = result.prediction_time.as_secs();
        let ms = result.prediction_time.subsec_millis();
        let time_text = if s > 0 {
            format!("{}.{:03} seconds", s, ms)
        } else {
            format!("{} ms", ms)
        };
        ui.label(
            RichText::new(time_text)
                .size(13.0)
                .color(Color32::from_rgb(0, 150, 255)),
        );
    });

    ui.add_space(12.0);

    // Main result box
    let is_high = result.predicted_class == "High Grade";
    let (box_color, text_color, icon) = if is_high {
        (
            Color32::from_rgb(220, 235, 255),
            Color32::from_rgb(0, 80, 180),
            "🏆",
        )
    } else {
        (
            Color32::from_rgb(255, 225, 225),
            Color32::from_rgb(180, 0, 0),
            "📉",
        )
    };

    egui::Frame::none()
        .fill(box_color)
        .rounding(egui::Rounding::same(12.0))
        .inner_margin(egui::Margin::same(20.0))
        .stroke(egui::Stroke::new(2.0, text_color))
        .show(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.label(
                    RichText::new(format!("{}  {}", icon, result.predicted_class))
                        .size(34.0)
                        .strong()
                        .color(text_color),
                );
                ui.add_space(6.0);
                ui.label(
                    RichText::new(format!(
                        "Confidence: {:.1}%",
                        result.confidence * 100.0
                    ))
                    .size(15.0)
                    .color(text_color),
                );
            });
        });

    ui.add_space(20.0);

    // Confidence bars
    ui.label(
        RichText::new("📊  Confidence Score")
            .size(14.0)
            .strong()
            .color(Color32::DARK_GRAY),
    );
    ui.add_space(10.0);

    // High Grade bar
    ui.horizontal(|ui| {
        ui.label(RichText::new("High Grade:").size(13.0).strong());
        ui.add_space(4.0);
        let p = result.high_grade_prob;
        let bar_color = if p > 0.5 {
            Color32::from_rgb(0, 100, 220)
        } else {
            Color32::from_rgb(120, 170, 220)
        };
        ui.add(
            egui::ProgressBar::new(p)
                .fill(bar_color)
                .desired_width(280.0),
        );
        ui.label(
            RichText::new(format!("{:.1}%", p * 100.0))
                .size(13.0)
                .strong(),
        );
    });

    ui.add_space(6.0);

    // Low Grade bar
    ui.horizontal(|ui| {
        ui.label(RichText::new("Low Grade:").size(13.0).strong());
        ui.add_space(4.0);
        let p = result.low_grade_prob;
        let bar_color = if p > 0.5 {
            Color32::from_rgb(200, 0, 0)
        } else {
            Color32::from_rgb(200, 150, 150)
        };
        ui.add(
            egui::ProgressBar::new(p)
                .fill(bar_color)
                .desired_width(280.0),
        );
        ui.label(
            RichText::new(format!("{:.1}%", p * 100.0))
                .size(13.0)
                .strong(),
        );
    });

    ui.add_space(16.0);
}

// ═══════════════════════════════════════════════════════
// PREDICTION LOGIC
// ═══════════════════════════════════════════════════════

fn run_prediction(
    file_path: &str,
    norm_stats: &NormalizationStats,
    model: &CoffeeSVM,
    start_time: Instant,
) -> Result<PredictionResult, String> {
    let loader = CoffeeDataLoader::new("data/raw");

    let data = loader
        .load_single_file(&PathBuf::from(file_path), norm_stats)
        .map_err(|e| format!("Failed to load CSV: {}", e))?;

    // data shape from load_single_file: [1, timesteps, sensors]
    // models expect: [1, channels=sensors, timesteps] → permute axes
    let data = data.permuted_axes([0, 2, 1]);

    // Returns Vec<(class, p_high, p_low)>
    let results = model.predict_with_confidence(&data);
    let (class, p_high, p_low) = results
        .into_iter()
        .next()
        .ok_or_else(|| "No prediction result".to_string())?;

    let confidence = p_high.max(p_low);
    let predicted_class = if class == 0 {
        "High Grade".to_string()
    } else {
        "Low Grade".to_string()
    };

    Ok(PredictionResult {
        predicted_class,
        high_grade_prob: p_high,
        low_grade_prob: p_low,
        confidence,
        prediction_time: start_time.elapsed(),
    })
}

fn load_normalization_stats(path: &str) -> Result<NormalizationStats, String> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read normalization stats: {}", e))?;
    serde_json::from_str(&json).map_err(|e| format!("Failed to parse normalization stats: {}", e))
}

// ═══════════════════════════════════════════════════════
// FONT SETUP
// ═══════════════════════════════════════════════════════

fn setup_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    if let Ok(font_data) = std::fs::read("assets/fonts/Poppins-Regular.ttf") {
        fonts.font_data.insert("Poppins".to_owned(), egui::FontData::from_owned(font_data));
        fonts.families.get_mut(&egui::FontFamily::Proportional).unwrap()
            .insert(0, "Poppins".to_owned());
        fonts.families.insert(egui::FontFamily::Name("Poppins".into()), vec!["Poppins".to_owned()]);
    }

    if let Ok(font_data) = std::fs::read("assets/fonts/Poppins-Bold.ttf") {
        fonts.font_data.insert("PoppinsBold".to_owned(), egui::FontData::from_owned(font_data));
        fonts.families.get_mut(&egui::FontFamily::Proportional).unwrap()
            .insert(0, "PoppinsBold".to_owned());
        fonts.families.insert(egui::FontFamily::Name("PoppinsBold".into()), vec!["PoppinsBold".to_owned()]);
    }

    ctx.set_fonts(fonts);
}

// ═══════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([620.0, 580.0])
            .with_title("SVM Coffee Grade Predictor"),
        ..Default::default()
    };

    eframe::run_native(
        "SVM Coffee Predictor",
        options,
        Box::new(|cc| Ok(Box::new(SVMPredictorApp::new(cc)))),
    )
}
