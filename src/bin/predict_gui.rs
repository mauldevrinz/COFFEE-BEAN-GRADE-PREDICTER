//! Coffee Grade Predictor GUI - WITH REAL-TIME TIMER
//! Prediction interface for trained coffee classifier model

use coffee_classifier::ml::{
    CoffeeDataLoader, NormalizationStats, CoffeeDataset, DataLoaderConfig, CoffeeCNN
};
use eframe::egui;
use egui::{Color32, RichText};
use std::path::PathBuf;
use ndarray::{Array1, Array2, Array3};
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════

#[derive(Debug, Clone)]
enum PredictionState {
    Idle,
    Loading(Instant),  // ✅ Store start time
    Success(PredictionResult),
    Error(String),
}

#[derive(Debug, Clone)]
struct PredictionResult {
    predicted_class: String,
    high_grade_prob: f32,
    low_grade_prob: f32,
    #[allow(dead_code)]
    confidence: f32,
    similarity_score: f32,
    anomaly_score: f32,
    is_anomaly: bool,
    #[allow(dead_code)]
    nearest_neighbors: Vec<(usize, f32, i64)>,
    #[allow(dead_code)]
    file_path: String,
    prediction_time: Duration,
}


#[derive(Clone)]
struct TrainingDataStore {
    features: Array2<f32>,
    labels: Array1<i64>,
}

// ═══════════════════════════════════════════════════════
// MAIN GUI APPLICATION
// ═══════════════════════════════════════════════════════

struct CoffeePredictorApp {
    file_path: String,
    state: PredictionState,
    training_store: Option<TrainingDataStore>,
    norm_stats: Option<NormalizationStats>,
    model: Option<CoffeeCNN>,
    rx: Option<Receiver<Result<PredictionResult, String>>>,
}

impl Default for CoffeePredictorApp {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            state: PredictionState::Idle,
            training_store: None,
            norm_stats: None,
            model: None,
            rx: None,
        }
    }
}

impl CoffeePredictorApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // ✅ FORCE LIGHT THEME ON ALL PLATFORMS
        cc.egui_ctx.set_visuals(egui::Visuals::light());
        
        let mut app = Self::default();
        println!("🔧 Loading model and training data...");

        // Load normalization stats
        if let Ok(stats) = load_normalization_stats("models/normalization_stats.json") {
            app.norm_stats = Some(stats);
            println!("✅ Normalization stats loaded");
        } else {
            println!("⚠️ Failed to load normalization stats");
        }

        // Load model
        if let Ok(model) = CoffeeCNN::load("models/trained_model.json") {
            app.model = Some(model);
            println!("✅ Model loaded");
        } else {
            println!("⚠️ Failed to load model");
        }

        // Load training data
        if let Ok(store) = load_training_data() {
            app.training_store = Some(store);
            println!("✅ Training data loaded");
        } else {
            println!("⚠️ Failed to load training data");
        }

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

        if self.norm_stats.is_none() || self.model.is_none() || self.training_store.is_none() {
            self.state = PredictionState::Error("Model or data not loaded!".to_string());
            return;
        }

        // ✅ START TIMER
        let start_time = Instant::now();
        self.state = PredictionState::Loading(start_time);

        let file_path = self.file_path.clone();
        let norm_stats = self.norm_stats.clone().unwrap();
        let mut model = self.model.clone().unwrap();
        let training_store = self.training_store.clone().unwrap();

        let (tx, rx) = channel();
        self.rx = Some(rx);

        thread::spawn(move || {
            let result = run_prediction(&file_path, &norm_stats, &mut model, &training_store, start_time);
            let _ = tx.send(result);
        });
    }
}

impl eframe::App for CoffeePredictorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for prediction results
        if let Some(rx) = &self.rx {
            if let Ok(result) = rx.try_recv() {
                match result {
                    Ok(pred) => self.state = PredictionState::Success(pred),
                    Err(e) => self.state = PredictionState::Error(e),
                }
                self.rx = None;
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(RichText::new("☕ Coffee Grade Predictor").size(24.0).strong());
            ui.add_space(10.0);

            // File selection
            ui.horizontal(|ui| {
                ui.label("CSV File:");
                ui.text_edit_singleline(&mut self.file_path);
                if ui.button("📁 Browse").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("CSV", &["csv"])
                        .pick_file()
                    {
                        self.file_path = path.display().to_string();
                    }
                }
            });

            ui.add_space(10.0);

            if ui.button(RichText::new("🔍 Predict").size(16.0)).clicked() {
                self.predict();
            }

            ui.add_space(20.0);
            ui.separator();
            ui.add_space(20.0);

            // Display results
            match &self.state {
                PredictionState::Idle => {
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("Select a CSV file and click Predict").size(14.0).color(Color32::GRAY));
                    });
                }
                PredictionState::Loading(start_time) => {
                    ui.vertical_centered(|ui| {
                        ui.spinner();
                        
                        // ✅ REAL-TIME ELAPSED TIME
                        let elapsed = start_time.elapsed();
                        let seconds = elapsed.as_secs();
                        let millis = elapsed.subsec_millis();
                        
                        ui.label(RichText::new(format!("Analyzing... {} s", seconds)).size(16.0));
                        ui.label(RichText::new(format!("({}.{:03} seconds)", seconds, millis)).size(12.0).color(Color32::GRAY));
                    });
                    
                    // ✅ Request repaint to update timer
                    ctx.request_repaint();
                }
                PredictionState::Error(msg) => {
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new(format!("❌ Error: {}", msg)).size(14.0).color(Color32::RED));
                    });
                }
                PredictionState::Success(result) => {
                    display_results(ui, result);
                }
            }
        });

        // Only repaint when loading to update timer
        if matches!(self.state, PredictionState::Loading(_)) {
            ctx.request_repaint();
        }
    }
}

// ═══════════════════════════════════════════════════════
// RESULTS DISPLAY
// ═══════════════════════════════════════════════════════

fn display_results(ui: &mut egui::Ui, result: &PredictionResult) {
    ui.heading("🎯 Prediction Result");
    ui.add_space(10.0);

    // ✅ DISPLAY PREDICTION TIME
    ui.horizontal(|ui| {
        ui.label(RichText::new("⏱️ Prediction Time:").size(14.0).strong());
        let seconds = result.prediction_time.as_secs();
        let millis = result.prediction_time.subsec_millis();
        
        let time_text = if seconds > 0 {
            format!("{} s", seconds)
        } else {
            format!("{} ms", millis)
        };
        
        ui.label(RichText::new(time_text).size(14.0).color(Color32::from_rgb(0, 150, 255)));
    });
    
    ui.add_space(10.0);

    // Main prediction display
    if result.is_anomaly {
        ui.vertical_centered(|ui| {
            ui.label(
                RichText::new(&result.predicted_class)
                    .size(32.0)
                    .strong()
                    .color(Color32::from_rgb(255, 165, 0))
            );
        });
    } else {
        let color = if result.predicted_class.contains("High") {
            Color32::from_rgb(0, 200, 0)
        } else {
            Color32::from_rgb(200, 0, 0)
        };

        ui.vertical_centered(|ui| {
            ui.label(
                RichText::new(&result.predicted_class)
                    .size(32.0)
                    .strong()
                    .color(color)
            );
        });
    }

    ui.add_space(20.0);

    // Confidence scores
    if !result.is_anomaly {
        ui.heading("📊 Confidence Scores");
        ui.add_space(10.0);

        // High grade bar
        ui.horizontal(|ui| {
            ui.label(RichText::new("High Grade:").size(14.0).strong());
            ui.add_space(10.0);

            let progress = result.high_grade_prob;
            let bar_color = if progress > 0.5 {
                Color32::from_rgb(0, 200, 0)
            } else {
                Color32::GRAY
            };

            let _bar_response = ui.add(
                egui::ProgressBar::new(progress)
                    .fill(bar_color)
                    .desired_width(300.0)
            );
            ui.label(RichText::new(format!("{:.1}%", progress * 100.0)).size(14.0).strong());
        });

        ui.add_space(5.0);

        // Low grade bar
        ui.horizontal(|ui| {
            ui.label(RichText::new("Low Grade:").size(14.0).strong());
            ui.add_space(10.0);

            let progress = result.low_grade_prob;
            let bar_color = if progress > 0.5 {
                Color32::from_rgb(200, 0, 0)
            } else {
                Color32::GRAY
            };

            let _bar_response = ui.add(
                egui::ProgressBar::new(progress)
                    .fill(bar_color)
                    .desired_width(300.0)
            );
            ui.label(RichText::new(format!("{:.1}%", progress * 100.0)).size(14.0).strong());
        });

        ui.add_space(20.0);
    }

    if result.anomaly_score > 0.5 {
        ui.label(RichText::new("❌ DUMMY DATA DETECTED").color(Color32::RED).strong());
        ui.label("Data has unnatural patterns (flat/constant signals)");
    } else if result.is_anomaly {
        ui.label(RichText::new("⚠️ OUT-OF-DISTRIBUTION").color(Color32::from_rgb(255, 165, 0)).strong());
        ui.label(format!("Data differs significantly from training set (similarity: {:.1}%)", result.similarity_score * 100.0));
    }

    ui.add_space(20.0);
}

// ═══════════════════════════════════════════════════════
// PREDICTION LOGIC
// ═══════════════════════════════════════════════════════

fn run_prediction(
    file_path: &str,
    norm_stats: &NormalizationStats,
    model: &mut CoffeeCNN,
    training_store: &TrainingDataStore,
    start_time: Instant,  // ✅ Accept start time
) -> Result<PredictionResult, String> {
    let loader = CoffeeDataLoader::new("data/raw");
    
    let data = loader.load_single_file(&PathBuf::from(file_path), norm_stats)
        .map_err(|e| format!("Failed to load CSV: {}", e))?;

    let test_features = extract_test_features(&data);
    let anomaly_score = compute_anomaly_score(&data);

    let input_cnn = data.permuted_axes([0, 2, 1]);
    let probs = model.predict(&input_cnn);

    let mut p_high = probs[[0, 0]];
    let mut p_low = probs[[0, 1]];

    const K: usize = 10;
    let neighbors = find_k_nearest_neighbors(&test_features, training_store, K);
    let avg_distance: f32 = neighbors.iter().map(|(_, d, _)| d).sum::<f32>() / K as f32;
    let similarity_score = (-avg_distance / 10.0).exp();

    if similarity_score > 0.7 && anomaly_score < 0.3 {
        let boost_factor = 1.0 + (similarity_score - 0.7) * 2.0;
        if p_high > p_low {
            p_high = (p_high * boost_factor).min(0.99);
            p_low = 1.0 - p_high;
        } else {
            p_low = (p_low * boost_factor).min(0.99);
            p_high = 1.0 - p_low;
        }
    }

    let confidence = p_high.max(p_low);
    let is_dummy = anomaly_score > 0.5;
    let is_dissimilar = similarity_score < 0.5 && anomaly_score > 0.3;
    let is_anomaly = is_dummy || is_dissimilar;

    let predicted_class = if is_dummy {
        "Unknown / Dummy Data".to_string()
    } else if is_dissimilar {
        "Unknown / Out-of-Distribution".to_string()
    } else if p_high >= p_low {
        "High Grade".to_string()
    } else {
        "Low Grade".to_string()
    };

    // ✅ CALCULATE PREDICTION TIME
    let prediction_time = start_time.elapsed();

    Ok(PredictionResult {
        predicted_class,
        high_grade_prob: p_high,
        low_grade_prob: p_low,
        confidence,
        similarity_score,
        anomaly_score,
        is_anomaly,
        nearest_neighbors: neighbors,
        file_path: file_path.to_string(),
        prediction_time,  // ✅ Include in result
    })
}

fn load_normalization_stats(path: &str) -> Result<NormalizationStats, String> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    serde_json::from_str(&json)
        .map_err(|e| format!("Failed to parse JSON: {}", e))
}

fn load_training_data() -> Result<TrainingDataStore, String> {
    let data_config = DataLoaderConfig {
        train_ratio: 1.0,
        val_ratio: 0.0,
        test_ratio: 0.0,
        selected_samples: None,
        min_timesteps: 300,
        max_timesteps: 300,
    };

    let loader = CoffeeDataLoader::with_config("data/raw", data_config);
    let (train_dataset, _, _) = loader.load_all_data()
        .map_err(|e| format!("Failed to load training data: {}", e))?;

    extract_training_features(&train_dataset)
        .map_err(|e| format!("Failed to extract features: {}", e))
}

fn extract_training_features(dataset: &CoffeeDataset) -> Result<TrainingDataStore, String> {
    let num_samples = dataset.samples.shape()[0];
    let timesteps = dataset.samples.shape()[1];
    let sensors = dataset.samples.shape()[2];
    let feature_dim = sensors * 4;

    let mut features = Array2::<f32>::zeros((num_samples, feature_dim));

    for i in 0..num_samples {
        for s in 0..sensors {
            let sensor_data: Vec<f32> = dataset.samples
                .slice(ndarray::s![i, .., s])
                .iter()
                .cloned()
                .collect();

            let mean = sensor_data.iter().sum::<f32>() / timesteps as f32;
            let variance = sensor_data.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / timesteps as f32;
            let std = variance.sqrt();
            let min = sensor_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = sensor_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            features[[i, s * 4 + 0]] = mean;
            features[[i, s * 4 + 1]] = std;
            features[[i, s * 4 + 2]] = min;
            features[[i, s * 4 + 3]] = max;
        }
    }

    Ok(TrainingDataStore {
        features,
        labels: dataset.labels.clone(),
    })
}

fn extract_test_features(data: &Array3<f32>) -> Array1<f32> {
    let timesteps = data.shape()[1];
    let sensors = data.shape()[2];
    let feature_dim = sensors * 4;

    let mut features = Array1::<f32>::zeros(feature_dim);

    for s in 0..sensors {
        let sensor_data: Vec<f32> = data
            .slice(ndarray::s![0, .., s])
            .iter()
            .cloned()
            .collect();

        let mean = sensor_data.iter().sum::<f32>() / timesteps as f32;
        let variance = sensor_data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / timesteps as f32;
        let std = variance.sqrt();
        let min = sensor_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = sensor_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        features[s * 4 + 0] = mean;
        features[s * 4 + 1] = std;
        features[s * 4 + 2] = min;
        features[s * 4 + 3] = max;
    }

    features
}

fn compute_anomaly_score(data: &Array3<f32>) -> f32 {
    let mut indicators = Vec::new();

    let extreme_count = data.iter().filter(|&&x| x.abs() > 4.0).count();
    let extreme_ratio = extreme_count as f32 / data.len() as f32;
    if extreme_ratio > 0.01 {
        indicators.push(extreme_ratio * 10.0);
    }

    for s in 0..8 {
        let sensor_data: Vec<f32> = data.slice(ndarray::s![0, .., s]).iter().cloned().collect();
        let mean = sensor_data.iter().sum::<f32>() / sensor_data.len() as f32;
        let variance: f32 = sensor_data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / sensor_data.len() as f32;

        if variance < 0.05 {
            indicators.push(1.0);
        }
    }

    if indicators.is_empty() {
        0.0
    } else {
        let sum: f32 = indicators.iter().sum();
        let max_possible = indicators.len() as f32 * 2.0;
        (sum / max_possible).min(1.0)
    }
}

fn find_k_nearest_neighbors(
    test_features: &Array1<f32>,
    training_store: &TrainingDataStore,
    k: usize,
) -> Vec<(usize, f32, i64)> {
    let num_train = training_store.features.shape()[0];
    let mut distances = Vec::with_capacity(num_train);

    for i in 0..num_train {
        let train_features = training_store.features.row(i);
        let dist = euclidean_distance(test_features, &train_features.to_owned());
        distances.push((i, dist, training_store.labels[i]));
    }

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k);
    distances
}

fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ═══════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([700.0, 800.0])
            .with_title("Coffee Grade Predictor"),
        ..Default::default()
    };

    eframe::run_native(
        "Coffee Predictor",
        options,
        Box::new(|cc| Ok(Box::new(CoffeePredictorApp::new(cc)))),
    )
}
