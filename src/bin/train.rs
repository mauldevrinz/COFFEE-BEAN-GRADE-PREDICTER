//! Coffee Classifier Training - WITH TIMER & DETAILED METRICS
//! Train 1D-CNN model with temperature calibration and visualization

use coffee_classifier::ml::*;
use anyhow::Result;
use colored::*;
use log::info;
use ndarray::{s, Array1};
use serde::Serialize;
use std::time::Instant;

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    println!(
        "\n{}",
        "╔═══════════════════════════════════════════════════╗"
            .bold()
            .cyan()
    );
    println!(
        "{}",
        "║      Coffee Arabica 1D-CNN Classifier         ║"
            .bold()
            .cyan()
    );
  
    println!(
        "{}",
        "╚═══════════════════════════════════════════════════╝"
            .bold()
            .cyan()
    );

    // ⏱️ START TOTAL TIMER
    let total_start = Instant::now();

    // Configuration
    let data_config = DataLoaderConfig {
        train_ratio: 0.8,
        val_ratio: 0.2,
        test_ratio: 0.0,
        selected_samples: None,
        min_timesteps: 300,
        max_timesteps: 300,
    };

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

  let training_config = TrainingConfig {
    batch_size: 8,
    learning_rate: 0.0042,
    num_epochs: 100,
    weight_decay: 0.0001,
    
    // ✅ LENGKAP dengan field patience
    patience: 25,                      // Stop after 15 epochs without improvement
    min_delta: 0.001,                  // 0.1% minimum improvement
    max_overfitting_gap: 0.50,         // Stop if train-val gap > 50%
    stop_on_perfect: false,            // Don't stop at 100% accuracy
};

    // ───────────────── PHASE 1: LOAD DATA ─────────────────
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Phase 1/7: Loading Dataset".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    
    let load_start = Instant::now();
    let loader = CoffeeDataLoader::with_config("data/raw", data_config);
    let (train_dataset, val_dataset, _) = loader.load_all_data()?;
    let load_time = load_start.elapsed();
    
    info!("✅ Dataset loaded in {:.2}s", load_time.as_secs_f32());

    println!("\n📊 Dataset Summary:");
    println!("   Train samples: {}", train_dataset.labels.len());
    println!("   Val samples:   {}", val_dataset.labels.len());
    
    let train_high = train_dataset.labels.iter().filter(|&&x| x == 0).count();
    let train_low = train_dataset.labels.iter().filter(|&&x| x == 1).count();
    let val_high = val_dataset.labels.iter().filter(|&&x| x == 0).count();
    let val_low = val_dataset.labels.iter().filter(|&&x| x == 1).count();
    
    println!("   Train: {} High, {} Low", train_high, train_low);
    println!("   Val:   {} High, {} Low", val_high, val_low);
    println!("   ⏱️  Loading time: {:.2}s", load_time.as_secs_f32());

    // ───────────────── PHASE 2: PREPROCESS ─────────────────
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Phase 2/7: Preprocessing".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    let preprocess_start = Instant::now();
    let preprocessor = DataPreprocessor::new(preprocess_config);
    let train_processed = preprocessor.preprocess_batch(&train_dataset.samples)?;
    let val_processed = preprocessor.preprocess_batch(&val_dataset.samples)?;
    let preprocess_time = preprocess_start.elapsed();

    let report = preprocessor.generate_report(
        &train_dataset.samples.slice(s![0, .., ..]).to_owned(),
        &train_processed.slice(s![0, .., ..]).to_owned(),
    );
    report.print();
    println!("   ⏱️  Preprocessing time: {:.2}s", preprocess_time.as_secs_f32());

    let train_cnn = train_processed.clone().permuted_axes([0, 2, 1]);
    let val_cnn = val_processed.clone().permuted_axes([0, 2, 1]);

    // ───────────────── PHASE 3: BUILD CNN ─────────────────
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Phase 3/7: Building 1D-CNN".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    let mut model = CoffeeCNN::new();

    // ───────────────── PHASE 4: TRAIN ─────────────────
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Phase 4/7: Training".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    let training_start = Instant::now();
    let trainer = Trainer::new(training_config);
    let metrics_history = trainer.train(
        &mut model,
        &train_cnn,
        &train_dataset.labels,
        &val_cnn,
        &val_dataset.labels,
    );
    let training_time = training_start.elapsed();

    let best_val_loss = metrics_history
        .iter()
        .map(|m| m.val_loss)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    let best_val_acc = metrics_history
        .iter()
        .map(|m| m.val_accuracy)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    println!("\n🏆 Training Summary:");
    println!("   Total epochs: {}", metrics_history.len());
    println!("   Best val loss: {:.4}", best_val_loss);
    println!("   Best val acc:  {:.2}%", best_val_acc * 100.0);
    println!("   ⏱️  Training time: {:.2}s ({:.2}min)", 
        training_time.as_secs_f32(), 
        training_time.as_secs_f32() / 60.0
    );

    // ───────────────── PHASE 5: SAVE METRICS & VISUALIZE ─────────────────
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Phase 5/7: Saving Metrics & Generating Plots".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    std::fs::create_dir_all("models")?;
    save_metrics_json(&metrics_history, "models/training_metrics.json")?;
    println!("✅ Training metrics saved: models/training_metrics.json");

    std::fs::create_dir_all("models/plots")?;
    match visualize_training(&metrics_history, "models/plots") {
        Ok(_) => {
            println!("\n✅ Training plots generated successfully!");
            println!("   📁 Location: models/plots/");
            println!("   • loss_curve.png");
            println!("   • accuracy_curve.png");
            println!("   • training_summary.png");
            println!("   • overfitting_analysis.png");
        },
        Err(e) => {
            eprintln!("\n⚠️  Warning: Failed to generate plots: {}", e);
            println!("\n📊 Alternative: Use Python visualization:");
            println!("   python plot_training.py models/training_metrics.json");
        }
    }

    // ───────────────── PHASE 6: CALIBRATE TEMPERATURE ─────────────────
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Phase 6/7: Temperature Calibration".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    let calibration_start = Instant::now();
    model.calibrate_temperature(&val_cnn, &val_dataset.labels);
    let calibration_time = calibration_start.elapsed();
    println!("   ⏱️  Calibration time: {:.2}s", calibration_time.as_secs_f32());

    // ───────────────── PHASE 7: FINAL EVALUATION ─────────────────
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Phase 7/7: Final Evaluation (Calibrated)".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    let eval_start = Instant::now();

    // Train set evaluation
    let train_probs = model.predict(&train_cnn);
    let mut train_preds = Array1::<i64>::zeros(train_probs.shape()[0]);

    for i in 0..train_probs.shape()[0] {
        let row = train_probs.row(i);
        let pred_class = if row[0] > row[1] { 0 } else { 1 };
        train_preds[i] = pred_class as i64;
    }

    let train_results = Evaluator::evaluate(&train_preds, &train_dataset.labels);
    Evaluator::print_results(&train_results, "Train");

    // Validation set evaluation
    let val_probs = model.predict(&val_cnn);
    let mut val_preds = Array1::<i64>::zeros(val_probs.shape()[0]);

    println!("\n📊 Calibrated Confidence Distribution:");
    let mut high_confidences = Vec::new();
    let mut low_confidences = Vec::new();

    for i in 0..val_probs.shape()[0] {
        let row = val_probs.row(i);
        let pred_class = if row[0] > row[1] { 0 } else { 1 };
        val_preds[i] = pred_class as i64;

        let confidence = row[0].max(row[1]);
        if val_dataset.labels[i] == 0 {
            high_confidences.push(confidence);
        } else {
            low_confidences.push(confidence);
        }
    }

    if !high_confidences.is_empty() {
        let avg_high_conf = high_confidences.iter().sum::<f32>() / high_confidences.len() as f32;
        println!("   High Grade samples: avg confidence = {:.1}%", avg_high_conf * 100.0);
    }
    
    if !low_confidences.is_empty() {
        let avg_low_conf = low_confidences.iter().sum::<f32>() / low_confidences.len() as f32;
        println!("   Low Grade samples:  avg confidence = {:.1}%", avg_low_conf * 100.0);
    }

    let val_results = Evaluator::evaluate(&val_preds, &val_dataset.labels);
    Evaluator::print_results(&val_results, "Validation");

    let eval_time = eval_start.elapsed();
    println!("   ⏱️  Evaluation time: {:.2}s", eval_time.as_secs_f32());

    // ───────────────── SAVE MODEL ─────────────────
    println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    println!("{}", " Saving Model".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());

    model.save("models/trained_model.json")?;

    let norm_stats_json = serde_json::to_string_pretty(&train_dataset.normalization_stats)?;
    std::fs::write("models/normalization_stats.json", norm_stats_json)?;
    println!("✅ Normalization stats saved");

    // ⏱️ TOTAL TIME SUMMARY
    let total_time = total_start.elapsed();
    
    println!(
        "\n{}",
        "╔═══════════════════════════════════════════════════╗"
            .bold()
            .green()
    );
    println!(
        "{}",
        "║      Training Complete! 🎉      ║"
            .bold()
            .green()
    );
    println!(
        "{}",
        "║   Model ready for high-confidence predictions  ║"
            .bold()
            .green()
    );
    println!(
        "{}",
        "╚═══════════════════════════════════════════════════╝"
            .bold()
            .green()
    );

    // ⏱️ DETAILED TIME BREAKDOWN
    println!("\n⏱️  Time Breakdown:");
    
    println!("│ {} │ {:>9.2}s │", "TOTAL".bold(), total_time.as_secs_f32());
    println!("└────────────────────────┴──────────────┘");
    println!("   ({:.2} minutes total)", total_time.as_secs_f32() / 60.0);

    println!("\n📂 Generated Files:");
    println!("   ✓ models/trained_model.json");
    println!("   ✓ models/normalization_stats.json");
    println!("   ✓ models/training_metrics.json");
    println!("   ✓ models/plots/loss_curve.png");
    println!("   ✓ models/plots/accuracy_curve.png");
    println!("   ✓ models/plots/training_summary.png");
    println!("   ✓ models/plots/overfitting_analysis.png");
    
    println!("\n🎨 Next Steps:");
    println!("   • View plots in: models/plots/");
    println!("   • Test prediction: cargo run --bin predict --release -- <CSV_FILE>");

    Ok(())
}

// ═══════════════════════════════════════════════════════════
// HELPER: SAVE METRICS TO JSON
// ═══════════════════════════════════════════════════════════

#[derive(Serialize)]
struct MetricsLog {
    epochs: Vec<usize>,
    train_loss: Vec<f32>,
    train_accuracy: Vec<f32>,
    val_loss: Vec<f32>,
    val_accuracy: Vec<f32>,
}

fn save_metrics_json(metrics: &[TrainingMetrics], path: &str) -> Result<()> {
    let log = MetricsLog {
        epochs: (1..=metrics.len()).collect(),
        train_loss: metrics.iter().map(|m| m.train_loss).collect(),
        train_accuracy: metrics.iter().map(|m| m.train_accuracy).collect(),
        val_loss: metrics.iter().map(|m| m.val_loss).collect(),
        val_accuracy: metrics.iter().map(|m| m.val_accuracy).collect(),
    };

    let json = serde_json::to_string_pretty(&log)?;
    std::fs::write(path, json)?;
    
    
    Ok(())
}

use coffee_classifier::ml::visualization::training_plot::visualize_training;
