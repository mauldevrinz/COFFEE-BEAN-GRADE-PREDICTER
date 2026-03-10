// src/training.rs - SIMPLIFIED: Stop when stagnant for 25 epochs

use ndarray::{Array1, Array3};
use colored::*;
use std::time::Instant;
use std::sync::mpsc::Sender;


use crate::ml::cnn::CoffeeCNN;

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
    pub num_epochs: usize,
    pub weight_decay: f64,
    
    // ✅ Simplified early stopping
    pub patience: usize,                    // Wait N epochs before stopping
    pub min_delta: f32,                     // Minimum improvement considered significant
    pub max_overfitting_gap: f32,          // Maximum allowed train-val gap
    pub stop_on_perfect: bool,             // Stop if 100% accuracy reached
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            learning_rate: 0.0042,
            num_epochs: 100,
            weight_decay: 0.0001,
            
            // ✅ Set patience to 25 epochs
            patience: 25,
            min_delta: 0.001,
            max_overfitting_gap: 0.50,
            stop_on_perfect: false,  // ❌ Disabled
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub train_loss: f32,
    pub train_accuracy: f32,
    pub val_loss: f32,
    pub val_accuracy: f32,
    /// True = final end-of-epoch metric; False = intra-epoch live preview
    pub is_final: bool,
}

// ✅ Tracker for both train and val
struct EarlyStoppingTracker {
    best_train_acc: f32,
    best_val_acc: f32,
    best_epoch: usize,
    epochs_without_improvement: usize,
}

impl EarlyStoppingTracker {
    fn new() -> Self {
        Self {
            best_train_acc: 0.0,
            best_val_acc: 0.0,
            best_epoch: 0,
            epochs_without_improvement: 0,
        }
    }
    
    fn update(&mut self, epoch: usize, train_acc: f32, val_acc: f32, min_delta: f32) -> bool {
        // Check if BOTH train and val improved
        let train_improved = train_acc > self.best_train_acc + min_delta;
        let val_improved = val_acc > self.best_val_acc + min_delta;
        
        if train_improved || val_improved {
            // At least one improved, update best values
            if train_improved {
                self.best_train_acc = train_acc;
            }
            if val_improved {
                self.best_val_acc = val_acc;
            }
            self.best_epoch = epoch;
            self.epochs_without_improvement = 0;
            return true;
        } else {
            // Neither improved
            self.epochs_without_improvement += 1;
            return false;
        }
    }
}

pub struct Trainer {
    config: TrainingConfig,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    pub fn train(
        &self,
        model: &mut CoffeeCNN,
        train_data: &Array3<f32>,
        train_labels: &Array1<i64>,
        val_data: &Array3<f32>,
        val_labels: &Array1<i64>,
    ) -> Vec<TrainingMetrics> {
        self.train_with_callback(model, train_data, train_labels, val_data, val_labels, None)
    }

    // ✅ FUNGSI BARU: Training dengan callback untuk GUI
    pub fn train_with_callback(
        &self,
        model: &mut CoffeeCNN,
        train_data: &Array3<f32>,
        train_labels: &Array1<i64>,
        val_data: &Array3<f32>,
        val_labels: &Array1<i64>,
        progress_sender: Option<Sender<TrainingMetrics>>,
    ) -> Vec<TrainingMetrics> {
        let mut metrics_history = Vec::new();
        let mut early_stopping = EarlyStoppingTracker::new();

        println!("\n🚀 Starting CNN training...");
        println!("   Architecture: 1D-CNN with backpropagation");
        println!("   Optimizer: SGD");
        println!("   Loss: Cross Entropy");
        println!("   Patience: {} epochs\n", self.config.patience);

        let training_start = Instant::now();

        // Keep last val metrics for intra-epoch preview updates
        let mut last_val_loss = 1.0f32;
        let mut last_val_accuracy = 0.0f32;

        for epoch in 0..self.config.num_epochs {
            let epoch_start = Instant::now();

            // Train — send live batch progress to GUI
            let train_start = Instant::now();
            let (train_loss, train_accuracy) =
                self.train_epoch_with_progress(
                    model,
                    train_data,
                    train_labels,
                    last_val_loss,
                    last_val_accuracy,
                    progress_sender.as_ref(),
                );
            let train_time = train_start.elapsed();

            // Validate
            let val_start = Instant::now();
            let (val_loss, val_accuracy) =
                self.evaluate(model, val_data, val_labels);
            let val_time = val_start.elapsed();

            last_val_loss = val_loss;
            last_val_accuracy = val_accuracy;

            let epoch_time = epoch_start.elapsed();

            let metrics = TrainingMetrics {
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                is_final: true,
            };

            metrics_history.push(metrics.clone());

            // Send final per-epoch metrics (overrides any intra-epoch previews)
            if let Some(ref sender) = progress_sender {
                let _ = sender.send(metrics.clone());
            }

            // Update tracker (silent) - check both train and val
            early_stopping.update(epoch, train_accuracy, val_accuracy, self.config.min_delta);

            println!(
                "Epoch {:3}/{}: Train Loss: {:.4}, Acc: {:.2}% ({:.2}s) | Val Loss: {:.4}, Acc: {:.2}% ({:.2}s) | Total: {:.2}s",
                epoch + 1,
                self.config.num_epochs,
                train_loss,
                train_accuracy * 100.0,
                train_time.as_secs_f32(),
                val_loss,
                val_accuracy * 100.0,
                val_time.as_secs_f32(),
                epoch_time.as_secs_f32()
            );

            // ✅ ONLY CHECK: Stagnation for 25 epochs
            if early_stopping.epochs_without_improvement >= self.config.patience {
                let total_training_time = training_start.elapsed();
                
                println!("\n⏹️  Training stopped: No improvement for {} epochs", self.config.patience);
                println!("   Training completed: {:.2}s ({:.2}min)\n", 
                    total_training_time.as_secs_f32(),
                    total_training_time.as_secs_f32() / 60.0
                );
                
                break;
            }
        }

        // If training completed all epochs
        if metrics_history.len() == self.config.num_epochs {
            let total_training_time = training_start.elapsed();
            println!("\n✅ Training completed: {:.2}s ({:.2}min)\n",
                total_training_time.as_secs_f32(),
                total_training_time.as_secs_f32() / 60.0
            );
        }

        self.print_training_statistics(&metrics_history);

        metrics_history
    }

    /// Like `train_epoch` but sends running-average metrics to the GUI every few batches.
    fn train_epoch_with_progress(
        &self,
        model: &mut CoffeeCNN,
        data: &Array3<f32>,
        labels: &Array1<i64>,
        last_val_loss: f32,
        last_val_accuracy: f32,
        sender: Option<&Sender<TrainingMetrics>>,
    ) -> (f32, f32) {
        let num_samples = data.shape()[0];
        let num_batches =
            (num_samples + self.config.batch_size - 1) / self.config.batch_size;

        let mut total_loss = 0.0f32;
        let mut total_correct = 0usize;
        let mut batches_done = 0usize;

        // Send a preview update every ~10% of batches (min 1)
        let update_every = (num_batches / 10).max(1);

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * self.config.batch_size;
            let end_idx = (start_idx + self.config.batch_size).min(num_samples);

            let batch_data =
                data.slice(ndarray::s![start_idx..end_idx, .., ..]).to_owned();
            let batch_labels =
                labels.slice(ndarray::s![start_idx..end_idx]).to_owned();

            let (batch_loss, batch_accuracy) = model.train_step(
                &batch_data,
                &batch_labels,
                self.config.learning_rate,
            );

            let batch_size = end_idx - start_idx;
            total_loss += batch_loss * batch_size as f32;
            total_correct += (batch_accuracy * batch_size as f32) as usize;
            batches_done += 1;

            // Print terminal progress every update_every batches
            if batches_done % update_every == 0 || batches_done == num_batches {
                let running_loss = total_loss / (batches_done * self.config.batch_size).min(num_samples) as f32;
                let running_acc  = total_correct as f32 / (batches_done * self.config.batch_size).min(num_samples) as f32;
                print!(
                    "\r   Batch {}/{} — Loss: {:.4}  Acc: {:.1}%   ",
                    batches_done, num_batches,
                    running_loss, running_acc * 100.0,
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }

            // Send running-average preview to GUI so graph updates during long epochs
            if let Some(sender) = sender {
                if batches_done % update_every == 0 {
                    let running_loss = total_loss / (batches_done * self.config.batch_size).min(num_samples) as f32;
                    let running_acc  = total_correct as f32 / (batches_done * self.config.batch_size).min(num_samples) as f32;
                    let _ = sender.send(TrainingMetrics {
                        train_loss: running_loss,
                        train_accuracy: running_acc,
                        val_loss: last_val_loss,
                        val_accuracy: last_val_accuracy,
                        is_final: false,
                    });
                }
            }
        }
        println!(); // newline after the last \r progress line

        let avg_loss = total_loss / num_samples as f32;
        let avg_accuracy = total_correct as f32 / num_samples as f32;
        (avg_loss, avg_accuracy)
    }

    #[allow(dead_code)]
    fn train_epoch(
        &self,
        model: &mut CoffeeCNN,
        data: &Array3<f32>,
        labels: &Array1<i64>,
    ) -> (f32, f32) {
        let num_samples = data.shape()[0];
        let num_batches =
            (num_samples + self.config.batch_size - 1) / self.config.batch_size;

        let mut total_loss = 0.0;
        let mut total_correct = 0;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * self.config.batch_size;
            let end_idx = (start_idx + self.config.batch_size).min(num_samples);

            let batch_data =
                data.slice(ndarray::s![start_idx..end_idx, .., ..]).to_owned();
            let batch_labels =
                labels.slice(ndarray::s![start_idx..end_idx]).to_owned();

            let (batch_loss, batch_accuracy) = model.train_step(
                &batch_data,
                &batch_labels,
                self.config.learning_rate,
            );

            total_loss += batch_loss * (end_idx - start_idx) as f32;
            total_correct += (batch_accuracy * (end_idx - start_idx) as f32) as usize;
        }

        let avg_loss = total_loss / num_samples as f32;
        let avg_accuracy = total_correct as f32 / num_samples as f32;

        (avg_loss, avg_accuracy)
    }

    fn evaluate(
        &self,
        model: &mut CoffeeCNN,
        data: &Array3<f32>,
        labels: &Array1<i64>,
    ) -> (f32, f32) {
        let num_samples = data.shape()[0];
        if num_samples == 0 {
            return (0.0, 0.0);
        }

        let predictions = model.predict(data);
        let mut total_loss = 0.0;
        let mut correct = 0;

        for i in 0..num_samples {
            let label = labels[i] as usize;
            let pred = predictions.row(i).to_owned();

            let prob = pred[label].max(1e-10);
            let loss = -prob.ln();
            if loss.is_finite() {
                total_loss += loss;
            }

            let mut max_idx = 0;
            let mut max_val = pred[0];
            for j in 1..pred.len() {
                if pred[j] > max_val {
                    max_val = pred[j];
                    max_idx = j;
                }
            }

            if max_idx == label {
                correct += 1;
            }
        }

        let avg_loss = total_loss / num_samples as f32;
        let accuracy = correct as f32 / num_samples as f32;

        (avg_loss, accuracy)
    }

    // ✅ Simplified statistics (no overfitting warnings)
    fn print_training_statistics(&self, metrics: &[TrainingMetrics]) {
        if metrics.is_empty() {
            return;
        }

        println!("{}", "────────────────────────────────────────────".cyan());
        println!("{}", " Training Statistics".bold().cyan());
        println!("{}", "────────────────────────────────────────────".cyan());

        let best_train_acc = metrics.iter()
            .map(|m| m.train_accuracy)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let best_val_acc = metrics.iter()
            .map(|m| m.val_accuracy)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let best_train_loss = metrics.iter()
            .map(|m| m.train_loss)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let best_val_loss = metrics.iter()
            .map(|m| m.val_loss)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let final_metrics = metrics.last().unwrap();

        println!("\n📊 Best Metrics:");
        println!("   Train Accuracy: {:.2}%", best_train_acc * 100.0);
        println!("   Val Accuracy:   {:.2}%", best_val_acc * 100.0);
        println!("   Train Loss:     {:.4}", best_train_loss);
        println!("   Val Loss:       {:.4}", best_val_loss);

        println!("\n📈 Final Metrics (Epoch {}):", metrics.len());
        println!("   Train Accuracy: {:.2}%", final_metrics.train_accuracy * 100.0);
        println!("   Val Accuracy:   {:.2}%", final_metrics.val_accuracy * 100.0);
        println!("   Train Loss:     {:.4}", final_metrics.train_loss);
        println!("   Val Loss:       {:.4}", final_metrics.val_loss);

        let gap = (final_metrics.train_accuracy - final_metrics.val_accuracy).abs();
        println!("\n📉 Model Analysis:");
        println!("   Accuracy gap: {:.2}%", gap * 100.0);
        
        println!("\n✅ Training completed!");
    }
}
