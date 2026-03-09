//! src/ml/predict.rs - k-NN ENHANCED CONFIDENCE CALIBRATION

use anyhow::{Context, Result};
use colored::*;
use std::env;
use std::fs;
use std::path::PathBuf;
use ndarray::{Array1, Array2, Array3};

use crate::ml::{CoffeeDataLoader, NormalizationStats, CoffeeCNN, CoffeeDataset, DataLoaderConfig};

#[derive(Debug)]
pub struct PredictionResult {
    pub predicted_class: CoffeeGrade,
    pub confidence: f32,
    pub high_grade_prob: f32,
    pub low_grade_prob: f32,
    pub is_anomaly: bool,
    pub anomaly_score: f32,
    pub similarity_score: f32,  // NEW: similarity to training data
    pub nearest_neighbors: Vec<(usize, f32, i64)>,  // (index, distance, label)
}

#[derive(Debug)]
pub enum CoffeeGrade {
    High,
    Low,
    Unknown,
}

impl std::fmt::Display for CoffeeGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CoffeeGrade::High => write!(f, "High Grade"),
            CoffeeGrade::Low => write!(f, "Low Grade"),
            CoffeeGrade::Unknown => write!(f, "Unknown / Out-of-Distribution"),
        }
    }
}

// ✅ STRUCT TO STORE TRAINING DATA FOR k-NN
pub struct TrainingDataStore {
    pub features: Array2<f32>,  // (num_samples, feature_dim)
    pub labels: Array1<i64>,
}

pub fn main() -> Result<()> {
    env_logger::init();

    println!(
        "\n{}",
        "╔═══════════════════════════════════════════════════╗"
            .bold()
            .cyan()
    );
    println!(
        "{}",
        "║ Coffee Grade Predictor - k-NN Enhanced           ║"
            .bold()
            .cyan()
    );
    println!(
        "{}",
        "╚═══════════════════════════════════════════════════╝"
            .bold()
            .cyan()
    );

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("\n{}", "❌ Error: CSV file path required".red().bold());
        eprintln!("\n{}", "Usage:".bold());
        eprintln!("  cargo run --bin predict --release -- <PATH_TO_CSV>");
        std::process::exit(1);
    }

    let csv_path = PathBuf::from(&args[1]);
    if !csv_path.exists() {
        eprintln!(
            "\n{}",
            format!("❌ Error: File not found: {}", csv_path.display())
                .red()
                .bold()
        );
        std::process::exit(1);
    }

    println!("\n{}", "📋 Configuration:".bold());
    println!("  Input File: {}", csv_path.display());
    println!("  Model: k-NN Enhanced CNN");
    println!("  Confidence Boost: Similarity-based");
    println!("  Anomaly Detection: Improved");

    // ═══ STEP 1: LOAD TRAINING DATA FOR k-NN ═══
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Step 1/5: Loading Training Data Store".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    let data_config = DataLoaderConfig {
        train_ratio: 1.0,  // Load all as training
        val_ratio: 0.0,
        test_ratio: 0.0,
        selected_samples: None,
        min_timesteps: 300,
        max_timesteps: 300,
    };

    let loader = CoffeeDataLoader::with_config("data/raw", data_config);
    let (train_dataset, _, _) = loader.load_all_data()?;
    
    println!("✅ Training data loaded: {} samples", train_dataset.labels.len());

    // Extract features from training data (use CNN intermediate representations)
    let training_store = extract_training_features(&train_dataset)?;
    println!("✅ Feature store created: {} features", training_store.features.shape()[1]);

    // ═══ STEP 2: LOAD NORMALIZATION ═══
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Step 2/5: Loading Normalization Stats".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    let norm_stats = load_normalization_stats("models/normalization_stats.json")?;
    println!("✅ Normalization stats loaded");

    // ═══ STEP 3: LOAD & PREPROCESS DATA ═══
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Step 3/5: Loading & Preprocessing Test Data".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    let loader = CoffeeDataLoader::new("data/raw");
    let data = loader.load_single_file(&csv_path, &norm_stats)?;

    let shape = data.shape();
    println!(
        "✅ Data shape: [{}, {}, {}]",
        shape[0], shape[1], shape[2]
    );

    // ✅ IMPROVED ANOMALY DETECTION
    // ✅ Extract features BEFORE permuting (to avoid ownership issues)
    let test_features = extract_test_features(&data);
    
    let anomaly_score = compute_enhanced_anomaly_score(&data, &norm_stats);
    
    if anomaly_score > 0.5 {
        println!("⚠️  High anomaly score: {:.2} - likely DUMMY DATA!", anomaly_score);
    } else if anomaly_score > 0.3 {
        println!("⚠️  Moderate anomaly score: {:.2}", anomaly_score);
    } else {
        println!("✅ Low anomaly score: {:.2} - data looks normal", anomaly_score);
    }

    let input_cnn = data.permuted_axes([0, 2, 1]);

    // ═══ STEP 4: LOAD MODEL ═══
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Step 4/5: Loading Calibrated Model".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    let mut model = CoffeeCNN::load("models/trained_model.json")
        .context("Failed to load model")?;

    // ═══ STEP 5: k-NN ENHANCED INFERENCE ═══
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Step 5/5: k-NN Enhanced Inference".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    let prediction = run_knn_enhanced_prediction(
        &mut model,
        &input_cnn,
        &test_features,
        &training_store,
        anomaly_score,
    );
    
    print_prediction_results(&prediction, &csv_path);

    Ok(())
}

// ═══════════════════════════════════════════════════════
// FEATURE EXTRACTION
// ═══════════════════════════════════════════════════════

pub fn extract_training_features(dataset: &CoffeeDataset) -> Result<TrainingDataStore> {
    let num_samples = dataset.samples.shape()[0];
    let timesteps = dataset.samples.shape()[1];
    let sensors = dataset.samples.shape()[2];
    
    // Simple feature extraction: mean, std, min, max per sensor
    let feature_dim = sensors * 4;  // 8 sensors × 4 stats = 32 features
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

pub fn extract_test_features(data: &Array3<f32>) -> Array1<f32> {
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

// ═══════════════════════════════════════════════════════
// k-NN SIMILARITY COMPUTATION
// ═══════════════════════════════════════════════════════

pub fn find_k_nearest_neighbors(
    test_features: &Array1<f32>,
    training_store: &TrainingDataStore,
    k: usize,
) -> Vec<(usize, f32, i64)> {
    let num_train = training_store.features.shape()[0];
    let mut distances = Vec::with_capacity(num_train);
    
    // Compute Euclidean distance to all training samples
    for i in 0..num_train {
        let train_features = training_store.features.row(i);
        let dist = euclidean_distance(test_features, &train_features.to_owned());
        distances.push((i, dist, training_store.labels[i]));
    }
    
    // Sort by distance and take k nearest
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k);
    
    distances
}

pub fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ═══════════════════════════════════════════════════════
// ENHANCED ANOMALY DETECTION
// ═══════════════════════════════════════════════════════

pub fn compute_enhanced_anomaly_score(data: &Array3<f32>, _norm_stats: &NormalizationStats) -> f32 {
    let mut anomaly_indicators = Vec::new();
    
    // 1. Extreme values check
    let mut extreme_count = 0;
    let total_values = data.len();
    for val in data.iter() {
        if val.abs() > 4.0 {  // Stricter threshold
            extreme_count += 1;
        }
    }
    let extreme_ratio = extreme_count as f32 / total_values as f32;
    if extreme_ratio > 0.01 {
        anomaly_indicators.push(extreme_ratio * 10.0);
    }
    
    // 2. Constant/flat values (dummy data signature)
    for sensor_idx in 0..8 {
        let sensor_data: Vec<f32> = data.slice(ndarray::s![0, .., sensor_idx])
            .iter()
            .cloned()
            .collect();
        
        let variance: f32 = {
            let mean = sensor_data.iter().sum::<f32>() / sensor_data.len() as f32;
            sensor_data.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / sensor_data.len() as f32
        };
        
        // Very low variance = constant/flat signal = DUMMY
        if variance < 0.05 {
            anomaly_indicators.push(1.0);  // Strong indicator!
        }
    }
    
    // 3. Repetitive patterns (dummy data often repeats)
    let first_100: Vec<f32> = data.slice(ndarray::s![0, 0..100, 0])
        .iter()
        .cloned()
        .collect();
    let second_100: Vec<f32> = data.slice(ndarray::s![0, 100..200, 0])
        .iter()
        .cloned()
        .collect();
    
    let correlation = compute_correlation(&first_100, &second_100);
    if correlation > 0.95 {
        anomaly_indicators.push(0.8);  // Highly repetitive = suspicious
    }
    
    // Aggregate
    if anomaly_indicators.is_empty() {
        0.0
    } else {
        let sum: f32 = anomaly_indicators.iter().sum();
        let max_possible = anomaly_indicators.len() as f32 * 2.0;
        (sum / max_possible).min(1.0)
    }
}

pub fn compute_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    
    let numerator: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum();
    
    let denom_a: f32 = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f32>().sqrt();
    let denom_b: f32 = b.iter().map(|y| (y - mean_b).powi(2)).sum::<f32>().sqrt();
    
    if denom_a < 1e-10 || denom_b < 1e-10 {
        return 0.0;
    }
    
    numerator / (denom_a * denom_b)
}

// ═══════════════════════════════════════════════════════
// k-NN ENHANCED PREDICTION
// ═══════════════════════════════════════════════════════

pub fn run_knn_enhanced_prediction(
    model: &mut CoffeeCNN,
    input_cnn: &Array3<f32>,
    test_features: &Array1<f32>,
    training_store: &TrainingDataStore,
    anomaly_score: f32,
) -> PredictionResult {
    // Get base CNN prediction
    let probs = model.predict(input_cnn);
    let mut p_high = probs[[0, 0]];
    let mut p_low = probs[[0, 1]];
    
    // Find k nearest neighbors
    const K: usize = 10;
    let neighbors = find_k_nearest_neighbors(test_features, training_store, K);
    
    println!("\n🔍 k-NN Analysis:");
    println!("   Finding {} nearest training samples...", K);
    
    // Compute similarity score
    let avg_distance: f32 = neighbors.iter().map(|(_, d, _)| d).sum::<f32>() / K as f32;
    let similarity_score = (-avg_distance / 10.0).exp();  // Convert distance to similarity
    
    println!("   Average distance: {:.4}", avg_distance);
    println!("   Similarity score: {:.3}", similarity_score);
    
    // Count neighbor labels
    let high_count = neighbors.iter().filter(|(_, _, l)| *l == 0).count();
    let low_count = neighbors.iter().filter(|(_, _, l)| *l == 1).count();
    
    println!("   Neighbors: {} high, {} low", high_count, low_count);
    
    // ✅ CONFIDENCE BOOST BASED ON SIMILARITY
// âœ… KODE BARU - CONSERVATIVE BOOST
if similarity_score > 0.7 && anomaly_score < 0.3 {
    // Conservative exponential boost using square root
    let normalized_sim = (similarity_score - 0.7) / 0.3;  // Normalize to [0, 1]
    let boost_factor = 1.0 + normalized_sim.sqrt() * 0.2;  // Max 1.2x boost
    
    println!("\n   âœ… HIGH SIMILARITY â†' Applying conservative boost");
    println!("   Normalized similarity: {:.3}", normalized_sim);
    println!("   Boost factor: {:.2}x", boost_factor);
    
    if p_high > p_low {
        let original = p_high;
        p_high = (p_high * boost_factor).min(0.95);  // Cap at 95%
        p_low = 1.0 - p_high;
        println!("   Boosted: {:.1}% â†' {:.1}%", original * 100.0, p_high * 100.0);
    } else {
        let original = p_low;
        p_low = (p_low * boost_factor).min(0.95);
        p_high = 1.0 - p_low;
        println!("   Boosted: {:.1}% â†' {:.1}%", original * 100.0, p_low * 100.0);
    }
} else if similarity_score > 0.5 && anomaly_score < 0.3 {
    // Moderate similarity - minimal boost
    println!("\n   â„¹ï¸  MODERATE SIMILARITY â†' Minimal boost");
    let boost_factor = 1.05;  // Only 5% boost
    
    if p_high > p_low {
        p_high = (p_high * boost_factor).min(0.90);
        p_low = 1.0 - p_high;
    } else {
        p_low = (p_low * boost_factor).min(0.90);
        p_high = 1.0 - p_low;
    }
} else {
    println!("\n   âš ï¸  LOW SIMILARITY or HIGH ANOMALY â†' No boost applied");
}
    
    let confidence = p_high.max(p_low);
    
    // ✅ DECISION LOGIC
    let is_dummy = anomaly_score > 0.5;
    let is_dissimilar = similarity_score < 0.5 && anomaly_score > 0.3;
    let is_anomaly = is_dummy || is_dissimilar;
    
    let predicted_class = if is_dummy {
        // Dummy data → NO CLASS, just "Unknown"
        println!("   ⚠️  DUMMY DATA DETECTED - No classification");
        CoffeeGrade::Unknown
    } else if is_dissimilar {
        // Too different from training
        println!("   ⚠️  LOW SIMILARITY - Out of distribution");
        CoffeeGrade::Unknown
    } else if p_high >= p_low {
        println!("   ✅ PREDICTION: High Grade");
        CoffeeGrade::High
    } else {
        println!("   ✅ PREDICTION: Low Grade");
        CoffeeGrade::Low
    };
    
    PredictionResult {
        predicted_class,
        confidence,
        high_grade_prob: p_high,
        low_grade_prob: p_low,
        is_anomaly,
        anomaly_score,
        similarity_score,
        nearest_neighbors: neighbors,
    }
}

// ═══════════════════════════════════════════════════════
// OUTPUT
// ═══════════════════════════════════════════════════════

pub fn load_normalization_stats(path: &str) -> Result<NormalizationStats> {
    let json_str = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&json_str)?)
}

pub fn print_prediction_results(result: &PredictionResult, csv_path: &PathBuf) {
    println!(
        "\n{}",
        "╔═══════════════════════════════════════════════════╗"
            .bold()
            .green()
    );
    println!(
        "{}",
        "║ Prediction Results (k-NN Enhanced)               ║"
            .bold()
            .green()
    );
    println!(
        "{}",
        "╚═══════════════════════════════════════════════════╝"
            .bold()
            .green()
    );

    println!("\n{}", "📁 Input File:".bold());
    println!("  {}", csv_path.display());

    println!("\n{}", "🎯 Prediction:".bold());
    let class_str = if result.is_anomaly {
        format!("  {}", result.predicted_class).yellow().bold()
    } else {
        match result.predicted_class {
            CoffeeGrade::High => format!("  {}", result.predicted_class).green().bold(),
            CoffeeGrade::Low => format!("  {}", result.predicted_class).red().bold(),
            CoffeeGrade::Unknown => format!("  {}", result.predicted_class).yellow().bold(),
        }
    };
    println!("{}", class_str);

    if !result.is_anomaly {
        println!("\n{}", "📊 Confidence Scores (Enhanced):".bold());
        println!("┌─────────────────┬──────────────┬────────────────────────────────┐");
        println!("│ Class           │ Probability  │ Bar                            │");
        println!("├─────────────────┼──────────────┼────────────────────────────────┤");

        let high_bar_len = (result.high_grade_prob * 30.0) as usize;
        let high_bar = "█".repeat(high_bar_len);
        let high_color = if result.high_grade_prob > 0.5 {
            high_bar.green()
        } else {
            high_bar.normal()
        };
        println!(
            "│ High Grade      │ {:>11.1}% │ {:<30} │",
            result.high_grade_prob * 100.0,
            high_color
        );

        let low_bar_len = (result.low_grade_prob * 30.0) as usize;
        let low_bar = "█".repeat(low_bar_len);
        let low_color = if result.low_grade_prob > 0.5 {
            low_bar.red()
        } else {
            low_bar.normal()
        };
        println!(
            "│ Low  Grade      │ {:>11.1}% │ {:<30} │",
            result.low_grade_prob * 100.0,
            low_color
        );

        println!("└─────────────────┴──────────────┴────────────────────────────────┘");
    }

    println!("\n{}", "🔍 Analysis Metrics:".bold());
    println!("┌────────────────────────┬──────────────┐");
    println!("│ Metric                 │ Value        │");
    println!("├────────────────────────┼──────────────┤");
    println!("│ Similarity Score       │ {:>11.1}% │", result.similarity_score * 100.0);
    println!("│ Anomaly Score          │ {:>11.1}% │", result.anomaly_score * 100.0);
    println!("│ Model Confidence       │ {:>11.1}% │", result.confidence * 100.0);
    println!("└────────────────────────┴──────────────┘");

    println!("\n{}", "💡 Interpretation:".bold());
    
    if result.anomaly_score > 0.5 {
        println!("  {} DUMMY DATA DETECTED", "❌".red().bold());
        println!("  {} Data memiliki pola tidak natural (flat/constant)", "→".red());
        println!("  {} Tidak dilakukan klasifikasi", "→".red());
    } else if result.is_anomaly {
        println!("  {} OUT-OF-DISTRIBUTION", "⚠️ ".yellow().bold());
        println!("  {} Data berbeda signifikan dari training set", "→".yellow());
        println!("  {} Similarity: {:.1}%", "→".yellow(), result.similarity_score * 100.0);
    } else {
        if result.similarity_score > 0.7 {
            println!("  {} HIGH SIMILARITY to training data", "✅".green());
            println!("  {} Confidence boosted by k-NN analysis", "→".green());
        }
        
        if result.confidence >= 0.90 {
            println!("  {} Very confident prediction (≥90%)", "✅".green());
        } else if result.confidence >= 0.70 {
            println!("  {} Confident prediction (70-90%)", "✅".cyan());
        }
    }

    println!("\n{}", "🔬 Nearest Training Samples:".bold());
    for (i, (idx, dist, label)) in result.nearest_neighbors.iter().take(5).enumerate() {
        let label_str = if *label == 0 { "High" } else { "Low" };
        let color = if *label == 0 { "green" } else { "red" };
        println!("  {}. Sample #{} - {} - distance: {:.4}", 
                 i + 1, idx, 
                 if color == "green" { label_str.green() } else { label_str.red() },
                 dist);
    }

    println!(
        "\n{}",
        "✅ k-NN Enhanced inference completed!"
            .green()
            .bold()
    );
}
