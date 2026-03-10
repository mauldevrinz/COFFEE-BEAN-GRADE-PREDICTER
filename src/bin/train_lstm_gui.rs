//! LSTM Training GUI
//! File: src/bin/train_lstm_gui.rs
//! Layout konsisten dengan train_rf_gui.rs dan train_svm_gui.rs

use coffee_classifier::ml::*;
use coffee_classifier::ml::evaluation::EvaluationResults;
use coffee_classifier::power_monitor::{PowerMonitor, PowerSummary};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Legend, Corner};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use std::time::{Duration, Instant};
use colored::*;
use std::path::PathBuf;
use ndarray::Array1;

// ─────────────────────────────────────────────
// Folder tree node
// ─────────────────────────────────────────────

#[derive(Clone, Debug)]
struct FolderNode {
    name: String,
    files: Vec<String>,
    expanded: bool,
}

// ─────────────────────────────────────────────
// Result data dari training thread
// ─────────────────────────────────────────────

#[derive(Clone, Debug)]
struct LSTMResult {
    train_eval: EvaluationResults,
    val_eval: EvaluationResults,
    train_accuracy: f32,
    val_accuracy: f32,
    train_loss: f32,
    val_loss: f32,
    n_epochs: usize,
    training_secs: f64,
    /// (epoch_checkpoint, train_acc, val_acc)
    accuracy_curve: Vec<(usize, f32, f32)>,
    /// (epoch_checkpoint, train_loss, val_loss)
    loss_curve: Vec<(usize, f32, f32)>,
}

// ─────────────────────────────────────────────
// Main GUI struct
// ─────────────────────────────────────────────

use image as image_crate;
use coffee_classifier::report::{TrainingReport, generate_training_pdf};

struct LSTMTrainingGUI {
    is_training: bool,
    training_complete: bool,
    start_time: Option<Instant>,
    final_time: Option<f64>,
    result_receiver: Option<Receiver<LSTMResult>>,
    status_receiver: Option<Receiver<String>>,
    lstm_result: Option<LSTMResult>,
    status_log: Arc<Mutex<Vec<String>>>,
    epochs_done: Arc<AtomicUsize>,
    power_monitor: Option<PowerMonitor>,
    training_start_sample: usize,
    screenshot_requested: bool,
    screenshot_saved: Option<String>,
    pdf_saved: Option<String>,
    chart_rects: Vec<egui::Rect>,
    total_epochs: usize,

    high_grade_folders: Vec<FolderNode>,
    low_grade_folders: Vec<FolderNode>,
    high_grade_expanded: bool,
    low_grade_expanded: bool,
    last_refresh: Instant,
    refresh_interval: Duration,
}

impl Default for LSTMTrainingGUI {
    fn default() -> Self {
        let mut gui = Self {
            is_training: false,
            training_complete: false,
            start_time: None,
            final_time: None,
            result_receiver: None,
            status_receiver: None,
            lstm_result: None,
            status_log: Arc::new(Mutex::new(Vec::new())),
            epochs_done: Arc::new(AtomicUsize::new(0)),
            power_monitor: Some(PowerMonitor::start()),
            training_start_sample: 0,
            screenshot_requested: false,
            screenshot_saved: None,
            pdf_saved: None,
            chart_rects: Vec::new(),
            total_epochs: 50,
            high_grade_folders: Vec::new(),
            low_grade_folders: Vec::new(),
            high_grade_expanded: false,
            low_grade_expanded: false,
            last_refresh: Instant::now(),
            refresh_interval: Duration::from_secs(3),
        };
        gui.load_data_files();
        gui
    }
}

impl LSTMTrainingGUI {
    fn new() -> Self { Self::default() }

    fn auto_refresh_data(&mut self) {
        if self.last_refresh.elapsed() >= self.refresh_interval {
            self.load_data_files();
            self.last_refresh = Instant::now();
        }
    }

    fn load_data_files(&mut self) {
        let base = PathBuf::from("data/raw");
        self.high_grade_folders = Self::load_folder_nodes(&base.join("high_grade"));
        self.low_grade_folders  = Self::load_folder_nodes(&base.join("low_grade"));
    }

    fn load_folder_nodes(base: &PathBuf) -> Vec<FolderNode> {
        let mut nodes = Vec::new();
        let Ok(entries) = std::fs::read_dir(base) else { return nodes };
        let mut dirs: Vec<_> = entries.flatten().collect();
        dirs.sort_by_key(|e| e.file_name());
        for entry in dirs {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                let mut files: Vec<String> = std::fs::read_dir(&path)
                    .into_iter().flatten().flatten()
                    .filter(|e| e.path().extension().map(|x| x == "csv").unwrap_or(false))
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .collect();
                files.sort();
                nodes.push(FolderNode { name, files, expanded: false });
            }
        }
        nodes
    }

    fn open_file_explorer() {
        let path = PathBuf::from("data/raw");
        #[cfg(target_os = "linux")]
        { let _ = std::process::Command::new("xdg-open").arg(&path).spawn(); }
        #[cfg(target_os = "windows")]
        { let _ = std::process::Command::new("explorer").arg(&path).spawn(); }
        #[cfg(target_os = "macos")]
        { let _ = std::process::Command::new("open").arg(&path).spawn(); }
    }

    fn push_status(log: &Arc<Mutex<Vec<String>>>, msg: &str) {
        if let Ok(mut v) = log.lock() {
            println!("{}", msg);
            v.push(msg.to_string());
        }
    }

    fn start_training(&mut self) {
        if self.is_training { return; }

        self.is_training = true;
        self.training_complete = false;
        self.start_time = Some(Instant::now());
        self.final_time = None;
        self.lstm_result = None;
        self.epochs_done.store(0, Ordering::Relaxed);
        let needs_new_monitor = self.power_monitor.as_ref().map(|pm| pm.is_stopped()).unwrap_or(true);
        if needs_new_monitor {
            self.power_monitor = Some(PowerMonitor::start());
            self.training_start_sample = 0;
        } else {
            self.training_start_sample = self.power_monitor.as_ref().map(|pm| pm.sample_count()).unwrap_or(0);
        }
        self.total_epochs = 50; // harus match lstm_config.n_epochs

        if let Ok(mut log) = self.status_log.lock() { log.clear(); }

        let (result_tx, result_rx) = channel();
        let (status_tx, status_rx) = channel();
        self.result_receiver = Some(result_rx);
        self.status_receiver = Some(status_rx);
        let log = Arc::clone(&self.status_log);
        let epochs_done = Arc::clone(&self.epochs_done);

        thread::spawn(move || {
            let push = |msg: &str| {
                let _ = status_tx.send(msg.to_string());
                Self::push_status(&log, msg);
            };

            push(&format!(
                "\n{}",
                "╔═══════════════════════════════════════════════════╗".bold().magenta()
            ));
            push(&format!(
                "{}",
                "║   Coffee Arabica - LSTM Classifier               ║".bold().magenta()
            ));
            push(&format!(
                "{}",
                "╚═══════════════════════════════════════════════════╝".bold().magenta()
            ));

            let total_start = Instant::now();

            // ── Phase 1: Load Data ──
            push(&format!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".magenta()));
            push(&format!("{}", " Phase 1/4: Loading Dataset".bold().magenta()));

            let data_config = DataLoaderConfig {
                train_ratio: 0.8,
                val_ratio: 0.2,
                test_ratio: 0.0,
                selected_samples: None,
                min_timesteps: 300,
                max_timesteps: 300,
            };

            let loader = CoffeeDataLoader::with_config("data/raw", data_config);
            let (train_dataset, val_dataset, _) = match loader.load_all_data() {
                Ok(d) => d,
                Err(e) => { push(&format!("❌ Gagal load data: {}", e)); return; }
            };

            push(&format!("   Train samples: {}", train_dataset.labels.len()));
            push(&format!("   Val   samples: {}", val_dataset.labels.len()));
            let t_high = train_dataset.labels.iter().filter(|&&x| x == 0).count();
            let t_low  = train_dataset.labels.iter().filter(|&&x| x == 1).count();
            push(&format!("   Train: {} High, {} Low", t_high, t_low));
            let v_high = val_dataset.labels.iter().filter(|&&x| x == 0).count();
            let v_low  = val_dataset.labels.iter().filter(|&&x| x == 1).count();
            push(&format!("   Val:   {} High, {} Low", v_high, v_low));

            // ── Phase 2: Preprocess ──
            push(&format!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".magenta()));
            push(&format!("{}", " Phase 2/4: Preprocessing".bold().magenta()));

            let preprocess_config = PreprocessingConfig {
                remove_outliers: false,
                outlier_threshold: 3.0,
                handle_missing: true,
                missing_strategy: MissingStrategy::Interpolate,
                apply_smoothing: false,
                smoothing_window: 5,
                clip_values: false,
                clip_min: -10.0,
                clip_max: 10.0,
            };

            let preprocessor = DataPreprocessor::new(preprocess_config);
            let train_processed = match preprocessor.preprocess_batch(&train_dataset.samples) {
                Ok(d) => d,
                Err(e) => { push(&format!("❌ Preprocessing error: {}", e)); return; }
            };
            let val_processed = match preprocessor.preprocess_batch(&val_dataset.samples) {
                Ok(d) => d,
                Err(e) => { push(&format!("❌ Preprocessing error: {}", e)); return; }
            };

            // LSTM: (samples, channels, timesteps) — permuted_axes([0,2,1])
            let train_lstm = train_processed.permuted_axes([0, 2, 1]);
            let val_lstm   = val_processed.permuted_axes([0, 2, 1]);
            push("   ✅ Preprocessing selesai");

            // ── Phase 3: Train LSTM ──
            push(&format!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".magenta()));
            push(&format!("{}", " Phase 3/4: Training LSTM".bold().magenta()));

            let lstm_config = LSTMConfig {
                hidden_size: 64,
                n_epochs: 50,
                learning_rate: 0.005,
                batch_size: 8,
                stride: 3,
                clip_grad: 5.0,
            };
            let eff_t = (300 + lstm_config.stride - 1) / lstm_config.stride;
            push(&format!(
                "   Config: hidden={}, lr={}, epochs={}, batch={}, stride={} ({}→{} timesteps)",
                lstm_config.hidden_size, lstm_config.learning_rate,
                lstm_config.n_epochs, lstm_config.batch_size,
                lstm_config.stride, 300, eff_t
            ));

            let train_start = Instant::now();
            let mut lstm = CoffeeLSTM::new(lstm_config);
            let (fit_result, accuracy_curve, loss_curve) = match lstm.fit_with_curve(
                &train_lstm,
                &train_dataset.labels,
                &val_lstm,
                &val_dataset.labels,
                |done, total| {
                    epochs_done.store(done, Ordering::Relaxed);
                    if done % 5 == 0 || done == total {
                        println!("   🔁 Epoch: {}/{}", done, total);
                    }
                }
            ) {
                Ok(r) => r,
                Err(e) => { push(&format!("❌ Training error: {}", e)); return; }
            };
            let training_secs = train_start.elapsed().as_secs_f64();

            push(&format!("   ✅ {} epochs selesai dalam {:.2}s",
                fit_result.n_epochs_trained, training_secs));
            push(&format!("   Train accuracy: {:.2}%", fit_result.train_accuracy * 100.0));
            push(&format!("   Train loss:     {:.4}", fit_result.train_loss));

            // ── Phase 4: Evaluate & Save ──
            push(&format!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".magenta()));
            push(&format!("{}", " Phase 4/4: Evaluasi & Simpan Model".bold().magenta()));

            // Train eval
            let train_probs = lstm.predict(&train_lstm);
            let mut train_preds = Array1::<i64>::zeros(train_probs.nrows());
            for i in 0..train_probs.nrows() {
                train_preds[i] = if train_probs[[i, 0]] >= train_probs[[i, 1]] { 0 } else { 1 };
            }
            let train_eval = Evaluator::evaluate(&train_preds, &train_dataset.labels);
            Evaluator::print_results(&train_eval, "Train");

            // Val eval
            let val_probs = lstm.predict(&val_lstm);
            let mut val_preds = Array1::<i64>::zeros(val_probs.nrows());
            for i in 0..val_probs.nrows() {
                val_preds[i] = if val_probs[[i, 0]] >= val_probs[[i, 1]] { 0 } else { 1 };
            }
            let val_eval = Evaluator::evaluate(&val_preds, &val_dataset.labels);
            Evaluator::print_results(&val_eval, "Validation");

            let val_correct = val_preds.iter().zip(val_dataset.labels.iter())
                .filter(|(p, l)| p == l).count();
            let val_accuracy = val_correct as f32 / val_dataset.labels.len() as f32;

            // Val loss (ambil dari kurva terakhir)
            let val_loss = loss_curve.last().map(|(_, _, vl)| *vl).unwrap_or(0.0);

            // Save model
            std::fs::create_dir_all("models").ok();
            lstm.save("models/lstm_model.json").ok();
            push("   ✅ Model disimpan: models/lstm_model.json");

            let total_time = total_start.elapsed();
            push(&format!(
                "\n✅ Training selesai! Total: {:.2}s ({:.2}min)",
                total_time.as_secs_f64(),
                total_time.as_secs_f64() / 60.0
            ));

            let _ = result_tx.send(LSTMResult {
                train_eval,
                val_eval,
                train_accuracy: fit_result.train_accuracy,
                val_accuracy,
                train_loss: fit_result.train_loss,
                val_loss,
                n_epochs: fit_result.n_epochs_trained,
                training_secs,
                accuracy_curve,
                loss_curve,
            });
        });
    }

    fn update_state(&mut self) {
        if let Some(ref rx) = self.status_receiver {
            while rx.try_recv().is_ok() {}
        }
        if let Some(ref rx) = self.result_receiver {
            if let Ok(result) = rx.try_recv() {
                self.lstm_result = Some(result);
                self.is_training = false;
                self.training_complete = true;
                if let Some(ref pm) = self.power_monitor { pm.stop(); }
                if let Some(start) = self.start_time {
                    self.final_time = Some(start.elapsed().as_secs_f64());
                }
            }
        }
    }

    fn draw_tree_view(&mut self, ui: &mut egui::Ui, is_high_grade: bool) {
        let (folders, expanded) = if is_high_grade {
            (&mut self.high_grade_folders, &mut self.high_grade_expanded)
        } else {
            (&mut self.low_grade_folders, &mut self.low_grade_expanded)
        };
        let total_folders = folders.len();
        let root_name = if is_high_grade { "high_grade" } else { "low_grade" };
        let arrow = if *expanded { "▼" } else { "►" };
        let folder_icon = if *expanded { "📂" } else { "📁" };
        ui.horizontal(|ui| {
            if ui.button(format!("{} {} {} ({} folders)", arrow, folder_icon, root_name, total_folders)).clicked() {
                *expanded = !*expanded;
            }
        });
        if *expanded {
            for folder in folders.iter_mut() {
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    let fa = if folder.expanded { "▼" } else { "►" };
                    let fi = if folder.expanded { "📂" } else { "📁" };
                    if ui.button(format!("{} {} {} ({} files)", fa, fi, folder.name, folder.files.len())).clicked() {
                        folder.expanded = !folder.expanded;
                    }
                });
                if folder.expanded {
                    for file in &folder.files {
                        ui.horizontal(|ui| {
                            ui.add_space(40.0);
                            ui.label(egui::RichText::new(format!("📄 {}", file))
                                .size(12.0).color(egui::Color32::from_rgb(80, 80, 80))
                                .family(egui::FontFamily::Monospace));
                        });
                    }
                }
            }
        }
    }

    fn draw_accuracy_plot(&self, ui: &mut egui::Ui, result: &LSTMResult) -> egui::Rect {
        if result.accuracy_curve.is_empty() { return egui::Rect::NOTHING; }

        ui.label(egui::RichText::new("Accuracy vs Epoch")
            .size(14.0).color(egui::Color32::BLACK)
            .family(egui::FontFamily::Name("PoppinsBold".into())));
        ui.add_space(6.0);

        let train_pts: PlotPoints = result.accuracy_curve.iter()
            .map(|(ep, ta, _)| [*ep as f64, (*ta * 100.0) as f64]).collect();
        let val_pts: PlotPoints = result.accuracy_curve.iter()
            .map(|(ep, _, va)| [*ep as f64, (*va * 100.0) as f64]).collect();

        let plot_resp = Plot::new("lstm_acc_curve")
            .legend(Legend::default().position(Corner::RightBottom))
            .show_axes([true, true])
            .show_grid([true, true])
            .allow_drag(false).allow_zoom(false).allow_scroll(false)
            .x_axis_label("Epoch").y_axis_label("Accuracy (%)")
            .height(190.0).width(560.0)
            .include_x(0.0).include_x(result.n_epochs as f64)
            .include_y(0.0).include_y(100.0)
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new(train_pts)
                    .color(egui::Color32::from_rgb(255, 140, 0)).width(2.5).name("Train Acc"));
                plot_ui.line(Line::new(val_pts)
                    .color(egui::Color32::from_rgb(150, 50, 220)).width(2.5).name("Val Acc"));
            });
        plot_resp.response.rect
    }

    fn draw_loss_plot(&self, ui: &mut egui::Ui, result: &LSTMResult) -> egui::Rect {
        if result.loss_curve.is_empty() { return egui::Rect::NOTHING; }

        ui.label(egui::RichText::new("Loss vs Epoch")
            .size(14.0).color(egui::Color32::BLACK)
            .family(egui::FontFamily::Name("PoppinsBold".into())));
        ui.add_space(6.0);

        let train_pts: PlotPoints = result.loss_curve.iter()
            .map(|(ep, tl, _)| [*ep as f64, *tl as f64]).collect();
        let val_pts: PlotPoints = result.loss_curve.iter()
            .map(|(ep, _, vl)| [*ep as f64, *vl as f64]).collect();

        let plot_resp = Plot::new("lstm_loss_curve")
            .legend(Legend::default().position(Corner::RightTop))
            .show_axes([true, true])
            .show_grid([true, true])
            .allow_drag(false).allow_zoom(false).allow_scroll(false)
            .x_axis_label("Epoch").y_axis_label("Loss")
            .height(190.0).width(560.0)
            .include_x(0.0).include_x(result.n_epochs as f64)
            .include_y(0.0)
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new(train_pts)
                    .color(egui::Color32::from_rgb(255, 80, 80)).width(2.5).name("Train Loss"));
                plot_ui.line(Line::new(val_pts)
                    .color(egui::Color32::from_rgb(150, 50, 220)).width(2.5).name("Val Loss"));
            });
        plot_resp.response.rect
    }

    fn draw_eval_tables(&self, ui: &mut egui::Ui, id_prefix: &str, eval: &EvaluationResults) {
        let table_width = 590.0;
        ui.vertical(|ui| {
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(240, 230, 250))
                .inner_margin(10.0).rounding(5.0)
                .show(ui, |ui| {
                    ui.set_width(table_width);
                    ui.label(egui::RichText::new("Per-Class Metrics:")
                        .color(egui::Color32::BLACK).size(14.0)
                        .family(egui::FontFamily::Name("PoppinsBold".into())));
                    ui.add_space(5.0);
                    egui::Grid::new(format!("{}_lstm_metrics", id_prefix))
                        .striped(false).spacing([20.0, 5.0])
                        .show(ui, |ui| {
                            for header in ["Metric", "High Grade", "Low Grade"] {
                                ui.label(egui::RichText::new(header).color(egui::Color32::BLACK)
                                    .strong().family(egui::FontFamily::Name("Poppins".into())));
                            }
                            ui.end_row();
                            let rows = [
                                ("Precision", eval.high_grade_metrics.precision, eval.low_grade_metrics.precision),
                                ("Recall",    eval.high_grade_metrics.recall,    eval.low_grade_metrics.recall),
                                ("F1-Score",  eval.high_grade_metrics.f1_score,  eval.low_grade_metrics.f1_score),
                            ];
                            for (name, high, low) in &rows {
                                ui.label(egui::RichText::new(*name).color(egui::Color32::BLACK)
                                    .family(egui::FontFamily::Name("Poppins".into())));
                                ui.label(egui::RichText::new(format!("{:.3}", high)).color(egui::Color32::BLACK)
                                    .family(egui::FontFamily::Name("Poppins".into())));
                                ui.label(egui::RichText::new(format!("{:.3}", low)).color(egui::Color32::BLACK)
                                    .family(egui::FontFamily::Name("Poppins".into())));
                                ui.end_row();
                            }
                            ui.label(egui::RichText::new("Support").color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{}", eval.high_grade_metrics.support))
                                .color(egui::Color32::BLACK).family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{}", eval.low_grade_metrics.support))
                                .color(egui::Color32::BLACK).family(egui::FontFamily::Name("Poppins".into())));
                            ui.end_row();
                        });
                });

            ui.add_space(5.0);

            egui::Frame::none()
                .fill(egui::Color32::from_rgb(240, 230, 250))
                .inner_margin(10.0).rounding(5.0)
                .show(ui, |ui| {
                    ui.set_width(table_width);
                    ui.label(egui::RichText::new("Confusion Matrix:")
                        .color(egui::Color32::BLACK).size(14.0)
                        .family(egui::FontFamily::Name("PoppinsBold".into())));
                    ui.add_space(5.0);
                    egui::Grid::new(format!("{}_lstm_cm", id_prefix))
                        .striped(false).spacing([20.0, 5.0])
                        .show(ui, |ui| {
                            for header in ["", "Pred: High", "Pred: Low"] {
                                ui.label(egui::RichText::new(header).color(egui::Color32::BLACK)
                                    .strong().family(egui::FontFamily::Name("Poppins".into())));
                            }
                            ui.end_row();
                            for (row_label, row_idx) in [("True: High", 0usize), ("True: Low", 1usize)] {
                                ui.label(egui::RichText::new(row_label).color(egui::Color32::BLACK)
                                    .family(egui::FontFamily::Name("Poppins".into())));
                                for col_idx in 0..2 {
                                    let val = eval.confusion_matrix[row_idx][col_idx];
                                    let color = if row_idx == col_idx {
                                        egui::Color32::from_rgb(120, 30, 200)
                                    } else {
                                        egui::Color32::from_rgb(180, 50, 50)
                                    };
                                    ui.label(egui::RichText::new(format!("{}", val))
                                        .color(color).strong()
                                        .family(egui::FontFamily::Name("Poppins".into())));
                                }
                                ui.end_row();
                            }
                        });
                    ui.add_space(5.0);
                    ui.label(egui::RichText::new(format!("Accuracy: {:.2}%", eval.accuracy * 100.0))
                        .size(14.0).color(egui::Color32::from_rgb(120, 30, 200))
                        .family(egui::FontFamily::Name("PoppinsBold".into())));
                });
        });
    }

    fn save_pdf(&mut self) {
        if let Some(ref result) = self.lstm_result {
            let report = TrainingReport::LSTM {
                train_eval:     result.train_eval.clone(),
                val_eval:       result.val_eval.clone(),
                train_accuracy: result.train_accuracy,
                val_accuracy:   result.val_accuracy,
                train_loss:     result.train_loss,
                val_loss:       result.val_loss,
                n_epochs:       result.n_epochs,
                training_secs:  result.training_secs,
                accuracy_curve: result.accuracy_curve.clone(),
                loss_curve:     result.loss_curve.clone(),
                power:          self.power_monitor.as_ref()
                                    .and_then(|pm| pm.summary_from(self.training_start_sample)),
            };
            match generate_training_pdf(&report, "testing_results") {
                Ok(path) => self.pdf_saved = Some(format!("PDF: {}", path)),
                Err(e)   => self.pdf_saved = Some(format!("PDF Error: {}", e)),
            }
        }
    }
}

impl eframe::App for LSTMTrainingGUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_state();
        if self.is_training {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        self.auto_refresh_data();

        // ── Handle screenshot result ──
        let ppp = ctx.pixels_per_point();
        ctx.input(|i| {
            for event in &i.events {
                if let egui::Event::Screenshot { image, .. } = event {
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs()).unwrap_or(0);
                    let path = format!("testing_results/lstm_charts_{}.png", ts);
                    let [w, h] = image.size;
                    let _ = std::fs::create_dir_all("testing_results");
                    let save_result = if !self.chart_rects.is_empty() {
                        let bbox = self.chart_rects.iter().fold(egui::Rect::NOTHING, |acc, r| acc.union(*r));
                        let x0 = (bbox.min.x * ppp).floor() as usize;
                        let y0 = (bbox.min.y * ppp).floor() as usize;
                        let x1 = ((bbox.max.x * ppp).ceil() as usize).min(w);
                        let y1 = ((bbox.max.y * ppp).ceil() as usize).min(h);
                        let x0 = x0.min(w); let y0 = y0.min(h);
                        let cw = x1.saturating_sub(x0);
                        let ch = y1.saturating_sub(y0);
                        if cw > 0 && ch > 0 {
                            let mut cropped = Vec::with_capacity(cw * ch * 4);
                            for row in y0..y1 {
                                for col in x0..x1 {
                                    let p = image.pixels[row * w + col];
                                    cropped.extend_from_slice(&[p.r(), p.g(), p.b(), p.a()]);
                                }
                            }
                            image_crate::save_buffer(&path, &cropped, cw as u32, ch as u32, image_crate::ColorType::Rgba8)
                        } else {
                            let raw: Vec<u8> = image.pixels.iter().flat_map(|p| [p.r(), p.g(), p.b(), p.a()]).collect();
                            image_crate::save_buffer(&path, &raw, w as u32, h as u32, image_crate::ColorType::Rgba8)
                        }
                    } else {
                        let raw: Vec<u8> = image.pixels.iter().flat_map(|p| [p.r(), p.g(), p.b(), p.a()]).collect();
                        image_crate::save_buffer(&path, &raw, w as u32, h as u32, image_crate::ColorType::Rgba8)
                    };
                    match save_result {
                        Ok(_) => self.screenshot_saved = Some(format!("Saved: {}", path)),
                        Err(e) => self.screenshot_saved = Some(format!("Error: {}", e)),
                    }
                }
            }
        });
        if self.screenshot_requested {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
            self.screenshot_requested = false;
        }

        let header_color = egui::Color32::from_rgb(100, 20, 180);
        let bg_color = egui::Color32::from_rgb(248, 245, 255);

        // ── Header ──
        egui::TopBottomPanel::top("lstm_header")
            .frame(egui::Frame::none()
                .fill(header_color)
                .inner_margin(egui::Margin { left: 20.0, right: 20.0, top: 12.0, bottom: 12.0 }))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("🔁 LSTM Training")
                        .size(22.0).color(egui::Color32::WHITE)
                        .family(egui::FontFamily::Name("PoppinsBold".into())));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(egui::RichText::new("Long Short-Term Memory · Coffee Quality Classifier")
                            .size(13.0).color(egui::Color32::from_rgb(210, 185, 240))
                            .family(egui::FontFamily::Name("Poppins".into())));
                    });
                });
            });

        // ── Bottom bar ──
        egui::TopBottomPanel::bottom("lstm_bottom")
            .frame(egui::Frame::none()
                .fill(bg_color)
                .inner_margin(egui::Margin { left: 20.0, right: 20.0, top: 12.0, bottom: 12.0 }))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let btn_size = egui::vec2(220.0, 40.0);
                    let (rect, response) = ui.allocate_exact_size(btn_size, egui::Sense::click());

                    let (fill_color, icon, text) = if self.is_training {
                        (egui::Color32::from_rgb(255, 140, 0), "⏳", "TRAINING...")
                    } else {
                        (header_color, "▶", "START TRAINING")
                    };
                    let fill_color = if response.hovered() && !self.is_training {
                        egui::Color32::from_rgb(120, 40, 200)
                    } else { fill_color };

                    ui.painter().rect_filled(rect, 4.0, fill_color);
                    let icon_g = ui.painter().layout_no_wrap(
                        icon.to_string(),
                        egui::FontId::proportional(16.0),
                        egui::Color32::WHITE,
                    );
                    let text_g = ui.painter().layout_no_wrap(
                        format!(" {}", text),
                        egui::FontId::new(15.0, egui::FontFamily::Name("PoppinsBold".into())),
                        egui::Color32::WHITE,
                    );
                    let total_w = icon_g.size().x + text_g.size().x;
                    let sx = rect.center().x - total_w / 2.0;
                    let icon_pos = egui::pos2(sx, rect.center().y - icon_g.size().y / 2.0);
                    let text_pos = egui::pos2(sx + icon_g.size().x, rect.center().y - text_g.size().y / 2.0);
                    ui.painter().galley(icon_pos, icon_g, egui::Color32::WHITE);
                    ui.painter().galley(text_pos, text_g, egui::Color32::WHITE);

                    if response.clicked() && !self.is_training {
                        self.start_training();
                    }

                    if self.training_complete {
                        ui.add_space(15.0);
                        let dl_size = egui::vec2(170.0, 40.0);
                        let (dl_rect, dl_resp) = ui.allocate_exact_size(dl_size, egui::Sense::click());
                        let dl_fill = if dl_resp.hovered() {
                            egui::Color32::from_rgb(0, 110, 200)
                        } else {
                            egui::Color32::from_rgb(0, 90, 170)
                        };
                        ui.painter().rect_filled(dl_rect, 4.0, dl_fill);
                        let dl_t = ui.painter().layout_no_wrap(
                            "📷 Download".to_string(),
                            egui::FontId::new(15.0, egui::FontFamily::Name("PoppinsBold".into())),
                            egui::Color32::WHITE,
                        );
                        let dl_tp = egui::pos2(
                            dl_rect.center().x - dl_t.size().x / 2.0,
                            dl_rect.center().y - dl_t.size().y / 2.0,
                        );
                        ui.painter().galley(dl_tp, dl_t, egui::Color32::WHITE);
                        if dl_resp.clicked() {
                            self.screenshot_requested = true;
                        }
                        // ── PDF Report button ──
                        ui.add_space(10.0);
                        let pdf_size = egui::vec2(170.0, 40.0);
                        let (pdf_rect, pdf_resp) = ui.allocate_exact_size(pdf_size, egui::Sense::click());
                        let pdf_fill = if pdf_resp.hovered() {
                            egui::Color32::from_rgb(140, 20, 20)
                        } else {
                            egui::Color32::from_rgb(110, 0, 0)
                        };
                        ui.painter().rect_filled(pdf_rect, 4.0, pdf_fill);
                        let pdf_t = ui.painter().layout_no_wrap(
                            "📄 PDF Report".to_string(),
                            egui::FontId::new(15.0, egui::FontFamily::Name("PoppinsBold".into())),
                            egui::Color32::WHITE,
                        );
                        let pdf_tp = egui::pos2(
                            pdf_rect.center().x - pdf_t.size().x / 2.0,
                            pdf_rect.center().y - pdf_t.size().y / 2.0,
                        );
                        ui.painter().galley(pdf_tp, pdf_t, egui::Color32::WHITE);
                        if pdf_resp.clicked() {
                            self.save_pdf();
                        }
                    }

                    ui.add_space(15.0);
                    let pred_size = egui::vec2(150.0, 40.0);
                    let (pred_rect, pred_resp) = ui.allocate_exact_size(pred_size, egui::Sense::click());
                    let pred_fill = if pred_resp.hovered() {
                        egui::Color32::from_rgb(220, 120, 0)
                    } else {
                        egui::Color32::from_rgb(200, 100, 0)
                    };
                    ui.painter().rect_filled(pred_rect, 4.0, pred_fill);
                    let pred_t = ui.painter().layout_no_wrap(
                        "🔍 Predict".to_string(),
                        egui::FontId::new(15.0, egui::FontFamily::Name("PoppinsBold".into())),
                        egui::Color32::WHITE,
                    );
                    let pred_tp = egui::pos2(
                        pred_rect.center().x - pred_t.size().x / 2.0,
                        pred_rect.center().y - pred_t.size().y / 2.0,
                    );
                    ui.painter().galley(pred_tp, pred_t, egui::Color32::WHITE);
                    if pred_resp.clicked() {
                        std::thread::spawn(|| {
                            use std::process::Command;
                            let display = std::env::var("DISPLAY").unwrap_or(":0".to_string());
                            let bin_name = "predict_lstm_gui";
                            let possible_paths = [
                                format!("./target/release/{}", bin_name),
                                format!("target/release/{}", bin_name),
                                format!("./target/debug/{}", bin_name),
                                format!("target/debug/{}", bin_name),
                            ];
                            let binary_path = possible_paths.iter()
                                .find(|p| std::path::Path::new(p.as_str()).exists())
                                .cloned()
                                .or_else(|| {
                                    std::env::current_exe().ok()
                                        .and_then(|e| e.parent().map(|d| d.join(bin_name).to_string_lossy().into_owned()))
                                });
                            match binary_path {
                                Some(path) => {
                                    if let Err(e) = Command::new(&path)
                                        .env("DISPLAY", display)
                                        .env("LIBGL_ALWAYS_SOFTWARE", "1")
                                        .env("GDK_BACKEND", "x11")
                                        .spawn()
                                    { eprintln!("[X] Failed to launch predict GUI: {}", e); }
                                }
                                None => eprintln!("[X] predict GUI binary not found: {}", bin_name),
                            }
                        });
                    }

                    ui.add_space(15.0);
                    if !self.training_complete {
                        let manage_size = egui::vec2(150.0, 40.0);
                        let (mrect, mresp) = ui.allocate_exact_size(manage_size, egui::Sense::click());
                        let mc = if mresp.hovered() {
                            egui::Color32::from_rgb(120, 20, 20)
                        } else {
                            egui::Color32::from_rgb(100, 0, 0)
                        };
                        ui.painter().rect_filled(mrect, 4.0, mc);
                        let mt = ui.painter().layout_no_wrap(
                            "Manage Data".to_string(),
                            egui::FontId::new(15.0, egui::FontFamily::Name("PoppinsBold".into())),
                            egui::Color32::WHITE,
                        );
                        let mtp = egui::pos2(
                            mrect.center().x - mt.size().x / 2.0,
                            mrect.center().y - mt.size().y / 2.0,
                        );
                        ui.painter().galley(mtp, mt, egui::Color32::WHITE);
                        if mresp.clicked() { Self::open_file_explorer(); }
                    }

                    if self.training_complete {
                        if let Some(ft) = self.final_time {
                            let mins = (ft / 60.0).floor() as i32;
                            let secs = (ft % 60.0).floor() as i32;
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if let Some(ref msg) = self.pdf_saved {
                                    ui.label(egui::RichText::new(msg.as_str())
                                        .size(10.0)
                                        .color(egui::Color32::from_rgb(120, 0, 0))
                                        .family(egui::FontFamily::Name("Poppins".into())));
                                    ui.add_space(10.0);
                                }
                                if let Some(ref msg) = self.screenshot_saved {
                                    ui.label(egui::RichText::new(msg.as_str())
                                        .size(11.0)
                                        .color(egui::Color32::from_rgb(0, 120, 0))
                                        .family(egui::FontFamily::Name("Poppins".into())));
                                    ui.add_space(10.0);
                                }
                                ui.label(egui::RichText::new(
                                    format!("{}m {}s ({:.2}min)", mins, secs, ft / 60.0))
                                    .size(16.0).color(egui::Color32::BLACK)
                                    .family(egui::FontFamily::Name("PoppinsBold".into())));
                            });
                        }
                    }
                });
            });

        // ── Central Panel ──
        egui::CentralPanel::default()
            .frame(egui::Frame::none()
                .fill(bg_color)
                .inner_margin(egui::Margin { left: 15.0, right: 15.0, top: 10.0, bottom: 10.0 }))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let avail_w = ui.available_width();

                    // ── Data Panel ──
                    {
                        let im = 12.0;
                        let gap = 20.0;
                        let col_w = (avail_w - gap) / 2.0;
                        let content_w = col_w - im * 2.0;
                        ui.horizontal(|ui| {
                            egui::Frame::none()
                                .fill(egui::Color32::WHITE)
                                .inner_margin(im).rounding(5.0)
                                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 200, 200)))
                                .show(ui, |ui| {
                                    ui.set_width(content_w);
                                    ui.label(egui::RichText::new("📂 Training Data")
                                        .size(14.0).color(egui::Color32::BLACK)
                                        .family(egui::FontFamily::Name("PoppinsBold".into())));
                                    ui.add_space(6.0);
                                    self.draw_tree_view(ui, true);
                                });
                            ui.add_space(gap);
                            egui::Frame::none()
                                .fill(egui::Color32::WHITE)
                                .inner_margin(im).rounding(5.0)
                                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 200, 200)))
                                .show(ui, |ui| {
                                    ui.set_width(content_w);
                                    ui.label(egui::RichText::new("📂 Validation Data")
                                        .size(14.0).color(egui::Color32::BLACK)
                                        .family(egui::FontFamily::Name("PoppinsBold".into())));
                                    ui.add_space(6.0);
                                    self.draw_tree_view(ui, false);
                                });
                        });
                    }
                    ui.add_space(10.0);

                    // ── Status / Progress ──
                    if self.is_training {
                        egui::Frame::none()
                            .fill(egui::Color32::WHITE)
                            .inner_margin(15.0).rounding(5.0)
                            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 200, 200)))
                            .show(ui, |ui| {
                                ui.set_min_width(avail_w);

                                let elapsed = self.start_time.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("⏳ Training LSTM, harap tunggu...")
                                        .size(16.0).color(header_color)
                                        .family(egui::FontFamily::Name("PoppinsBold".into())));
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        ui.label(egui::RichText::new(format!("⏱ {:.1}s", elapsed))
                                            .size(14.0).color(egui::Color32::from_rgb(100, 100, 100))
                                            .family(egui::FontFamily::Name("Poppins".into())));
                                    });
                                });

                                ui.add_space(8.0);

                                let done = self.epochs_done.load(Ordering::Relaxed);
                                let total = self.total_epochs.max(1);
                                let progress = done as f32 / total as f32;
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("🔁 Epochs:")
                                        .size(13.0).color(egui::Color32::from_rgb(60, 60, 60))
                                        .family(egui::FontFamily::Name("Poppins".into())));
                                    let bar_w = (ui.available_width() - 80.0).max(100.0);
                                    ui.add(
                                        egui::ProgressBar::new(progress)
                                            .desired_width(bar_w)
                                            .show_percentage()
                                            .animate(true)
                                    );
                                    ui.label(egui::RichText::new(format!("{}/{}", done, total))
                                        .size(13.0).color(egui::Color32::from_rgb(60, 60, 60))
                                        .family(egui::FontFamily::Name("Poppins".into())));
                                });

                                ui.add_space(8.0);

                                let log = if let Ok(v) = self.status_log.lock() { v.clone() } else { vec![] };
                                egui::ScrollArea::vertical()
                                    .max_height(150.0).stick_to_bottom(true)
                                    .show(ui, |ui| {
                                        for line in &log {
                                            ui.label(egui::RichText::new(line)
                                                .size(12.0).color(egui::Color32::from_rgb(60, 60, 60))
                                                .family(egui::FontFamily::Monospace));
                                        }
                                    });

                                // ── Power ──
                                if let Some(ref pm) = self.power_monitor {
                                    if let Some(s) = pm.current_sample() {
                                        ui.add_space(6.0);
                                        ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new("⚡ Power:")
                                                .size(13.0)
                                                .color(egui::Color32::from_rgb(60, 60, 60))
                                                .family(egui::FontFamily::Name("Poppins".into())));
                                            let total_j = pm.accumulated_joules_from(self.training_start_sample);
                                            ui.label(egui::RichText::new(format!(
                                                "System {:.1} W  |  CPU+GPU {:.1} W  |  🌡 CPU {:.1}°C  GPU {:.1}°C  |  ⚡ Total: {:.1} J ({:.4} Wh)",
                                                s.vdd_in_mw as f32 / 1000.0,
                                                s.vdd_cpu_gpu_mw as f32 / 1000.0,
                                                s.cpu_temp_c,
                                                s.gpu_temp_c,
                                                total_j,
                                                total_j / 3600.0,
                                            ))
                                            .size(13.0)
                                            .color(egui::Color32::from_rgb(120, 30, 200))
                                            .family(egui::FontFamily::Name("Poppins".into())));
                                        });
                                    }
                                }
                            });
                        ui.add_space(15.0);
                    }

                    // ── Results ──
                    if let Some(ref result) = self.lstm_result.clone() {
                        // Summary bar
                        egui::Frame::none()
                            .fill(header_color)
                            .inner_margin(10.0).rounding(5.0)
                            .show(ui, |ui| {
                                ui.set_min_width(avail_w);
                                let summary = format!(
                                    "🔁 {} Epochs  |  ⏱ {:.1}s  |  Train Loss: {:.4}  |  Val Loss: {:.4}  |  Train: {:.1}%  |  Val: {:.1}%  |  models/lstm_model.json",
                                    result.n_epochs, result.training_secs,
                                    result.train_loss, result.val_loss,
                                    result.train_accuracy * 100.0,
                                    result.val_accuracy * 100.0,
                                );
                                ui.label(egui::RichText::new(summary)
                                    .size(13.0).color(egui::Color32::WHITE)
                                    .family(egui::FontFamily::Name("PoppinsBold".into())));
                            });
                        ui.add_space(10.0);

                        // ── Power Summary ──
                        if let Some(ref pm) = self.power_monitor {
                            if let Some(ps) = pm.summary_from(self.training_start_sample) {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(245, 240, 255))
                                    .inner_margin(10.0)
                                    .rounding(5.0)
                                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 185, 230)))
                                    .show(ui, |ui| {
                                        ui.set_min_width(avail_w);
                                        ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new("⚡ Power Summary")
                                                .size(13.0)
                                                .color(egui::Color32::from_rgb(100, 20, 180))
                                                .family(egui::FontFamily::Name("PoppinsBold".into())));
                                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                ui.label(egui::RichText::new(format!("{} samples", ps.sample_count))
                                                    .size(11.0)
                                                    .color(egui::Color32::from_rgb(150, 150, 150))
                                                    .family(egui::FontFamily::Name("Poppins".into())));
                                            });
                                        });
                                        ui.add_space(5.0);
                                        ui.columns(4, |cols| {
                                            cols[0].vertical_centered(|ui| {
                                                ui.label(egui::RichText::new("Avg System").size(11.0).color(egui::Color32::from_rgb(130,130,130)).family(egui::FontFamily::Name("Poppins".into())));
                                                ui.label(egui::RichText::new(format!("{:.1} W", ps.avg_total_w)).size(15.0).color(egui::Color32::from_rgb(100,20,180)).family(egui::FontFamily::Name("PoppinsBold".into())));
                                                ui.label(egui::RichText::new(format!("Peak {:.1} W", ps.peak_total_w)).size(11.0).color(egui::Color32::from_rgb(130,130,130)).family(egui::FontFamily::Name("Poppins".into())));
                                            });
                                            cols[1].vertical_centered(|ui| {
                                                ui.label(egui::RichText::new("Avg CPU+GPU").size(11.0).color(egui::Color32::from_rgb(130,130,130)).family(egui::FontFamily::Name("Poppins".into())));
                                                ui.label(egui::RichText::new(format!("{:.1} W", ps.avg_cpu_gpu_w)).size(15.0).color(egui::Color32::from_rgb(100,20,180)).family(egui::FontFamily::Name("PoppinsBold".into())));
                                                ui.label(egui::RichText::new(format!("Peak {:.1} W", ps.peak_cpu_gpu_w)).size(11.0).color(egui::Color32::from_rgb(130,130,130)).family(egui::FontFamily::Name("Poppins".into())));
                                            });
                                            cols[2].vertical_centered(|ui| {
                                                ui.label(egui::RichText::new("CPU/GPU Temp").size(11.0).color(egui::Color32::from_rgb(130,130,130)).family(egui::FontFamily::Name("Poppins".into())));
                                                ui.label(egui::RichText::new(format!("{:.1}°C / {:.1}°C", ps.avg_cpu_temp, ps.avg_gpu_temp)).size(15.0).color(egui::Color32::from_rgb(100,20,180)).family(egui::FontFamily::Name("PoppinsBold".into())));
                                                ui.label(egui::RichText::new(format!("Peak {:.1}°C / {:.1}°C", ps.peak_cpu_temp, ps.peak_gpu_temp)).size(11.0).color(egui::Color32::from_rgb(130,130,130)).family(egui::FontFamily::Name("Poppins".into())));
                                            });
                                            cols[3].vertical_centered(|ui| {
                                                ui.label(egui::RichText::new("🔋 Total Energy").size(11.0).color(egui::Color32::from_rgb(100,20,180)).family(egui::FontFamily::Name("PoppinsBold".into())));
                                                ui.label(egui::RichText::new(format!("{:.1} J", ps.energy_joules)).size(18.0).color(egui::Color32::from_rgb(100,20,180)).family(egui::FontFamily::Name("PoppinsBold".into())));
                                                ui.label(egui::RichText::new(format!("{:.4} Wh", ps.energy_joules / 3600.0)).size(12.0).color(egui::Color32::from_rgb(100,20,180)).family(egui::FontFamily::Name("Poppins".into())));
                                            });
                                        });
                                    });
                                ui.add_space(10.0);
                            }
                        }

                        // ── Row 1: Accuracy + Loss plots ──
                        let mut frame_rects: Vec<egui::Rect> = Vec::new();
                        ui.columns(2, |cols| {
                            egui::Frame::none()
                                .fill(egui::Color32::WHITE)
                                .inner_margin(15.0).rounding(5.0)
                                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 200, 200)))
                                .show(&mut cols[0], |ui| {
                                    ui.set_min_width(ui.available_width());
                                    frame_rects.push(self.draw_accuracy_plot(ui, result));
                                });
                            egui::Frame::none()
                                .fill(egui::Color32::WHITE)
                                .inner_margin(15.0).rounding(5.0)
                                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 200, 200)))
                                .show(&mut cols[1], |ui| {
                                    ui.set_min_width(ui.available_width());
                                    frame_rects.push(self.draw_loss_plot(ui, result));
                                });
                        });
                        self.chart_rects = frame_rects;
                        ui.add_space(10.0);

                        // ── Row 2: Eval tables ──
                        ui.columns(2, |cols| {
                            cols[0].vertical(|ui| {
                                ui.label(egui::RichText::new("Train")
                                    .size(14.0).color(egui::Color32::BLACK)
                                    .family(egui::FontFamily::Name("PoppinsBold".into())));
                                ui.add_space(4.0);
                                self.draw_eval_tables(ui, "train", &result.train_eval);
                            });
                            cols[1].vertical(|ui| {
                                ui.label(egui::RichText::new("Validation")
                                    .size(14.0).color(egui::Color32::BLACK)
                                    .family(egui::FontFamily::Name("PoppinsBold".into())));
                                ui.add_space(4.0);
                                self.draw_eval_tables(ui, "val", &result.val_eval);
                            });
                        });
                        ui.add_space(15.0);
                    }
                });
            });
    }
}

// ─────────────────────────────────────────────────────────────
// main + setup_fonts
// ─────────────────────────────────────────────────────────────

fn main() -> eframe::Result<()> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1250.0, 900.0])
            .with_title("LSTM Training"),
        ..Default::default()
    };

    eframe::run_native(
        "LSTM Training",
        options,
        Box::new(|cc| {
            setup_fonts(&cc.egui_ctx);
            Ok(Box::new(LSTMTrainingGUI::new()))
        }),
    )
}

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
