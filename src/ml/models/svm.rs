// src/ml/models/svm.rs - Linear SVM Coffee Quality Classifier
// Menggunakan SGD dengan hinge loss, konsisten dengan random_forest.rs

use anyhow::Result;
use ndarray::{Array1, Array2, s};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

// ============================================
// FEATURE EXTRACTION (sama dengan random_forest.rs)
// ============================================
pub fn extract_features_svm(sample: &Array2<f32>) -> Array1<f32> {
    let (channels, timesteps) = sample.dim();
    let mut features = Vec::with_capacity(channels * 6);

    for c in 0..channels {
        let channel = sample.slice(s![c, ..]);
        let values: Vec<f32> = channel.iter().cloned().collect();

        let mean = values.iter().sum::<f32>() / timesteps as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / timesteps as f32;
        let std = variance.sqrt();
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if timesteps % 2 == 0 {
            (sorted[timesteps / 2 - 1] + sorted[timesteps / 2]) / 2.0
        } else {
            sorted[timesteps / 2]
        };

        features.extend_from_slice(&[mean, std, min, max, range, median]);
    }

    Array1::from(features)
}

// ============================================
// SVM CONFIG
// ============================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVMConfig {
    /// Regularization parameter (trade-off antara margin dan error)
    pub c: f32,
    /// Learning rate untuk SGD
    pub learning_rate: f32,
    /// Jumlah epoch training
    pub n_epochs: usize,
    /// Ukuran mini-batch (1 = SGD murni)
    pub batch_size: usize,
}

impl Default for SVMConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            learning_rate: 0.01,
            n_epochs: 100,
            batch_size: 32,
        }
    }
}

// ============================================
// TRAINING METRICS
// ============================================
#[derive(Debug, Clone)]
pub struct SVMTrainingMetrics {
    pub n_epochs_trained: usize,
    pub train_accuracy: f32,
    pub n_support_vectors: usize,
    pub final_loss: f32,
}

// ============================================
// NORMALIZATION STATS (untuk feature scaling)
// ============================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl FeatureStats {
    pub fn fit(features: &Array2<f32>) -> Self {
        let (n_samples, n_features) = features.dim();
        let mut mean = vec![0.0f32; n_features];
        let mut std = vec![1.0f32; n_features];

        for j in 0..n_features {
            let col: Vec<f32> = (0..n_samples).map(|i| features[[i, j]]).collect();
            let m = col.iter().sum::<f32>() / n_samples as f32;
            let v = col.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n_samples as f32;
            mean[j] = m;
            std[j] = v.sqrt().max(1e-8);
        }

        Self { mean, std }
    }

    pub fn transform(&self, features: &Array2<f32>) -> Array2<f32> {
        let (n_samples, n_features) = features.dim();
        let mut out = Array2::<f32>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                out[[i, j]] = (features[[i, j]] - self.mean[j]) / self.std[j];
            }
        }
        out
    }

    pub fn transform_one(&self, features: &Array1<f32>) -> Array1<f32> {
        let n = features.len();
        let mut out = Array1::<f32>::zeros(n);
        for j in 0..n {
            out[j] = (features[j] - self.mean[j]) / self.std[j];
        }
        out
    }
}

// ============================================
// COFFEE SVM
// ============================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoffeeSVM {
    config: SVMConfig,
    weights: Vec<f32>,
    bias: f32,
    n_features: usize,
    feature_stats: Option<FeatureStats>,
}

impl CoffeeSVM {
    pub fn new(config: SVMConfig) -> Self {
        Self {
            config,
            weights: Vec::new(),
            bias: 0.0,
            n_features: 0,
            feature_stats: None,
        }
    }

    /// Konversi label 0/1 ke -1/+1 untuk SVM
    fn to_svm_label(label: i64) -> f32 {
        if label == 0 { 1.0 } else { -1.0 }
    }

    /// Konversi SVM output ke label 0/1
    fn from_svm_label(score: f32) -> i64 {
        if score >= 0.0 { 0 } else { 1 }
    }

    /// Ekstrak feature matrix dari dataset (samples, channels, timesteps) -> (samples, features)
    pub fn extract_feature_matrix(data: &Array2<f32>, _channels: usize) -> Array2<f32> {
        // data shape: (samples * channels, timesteps) — tapi di sini data sudah (samples, features)
        // Kita pakai extract_features_svm per sample
        // data di-pass sebagai (n_samples, n_features) sudah pre-extracted
        data.clone()
    }

    /// Fit SVM dari data (n_samples, channels, timesteps) shape Array3
    /// Menggunakan ndarray Array2 dengan shape (n_samples, n_features) yang sudah di-flatten
    pub fn fit_with_progress<F>(
        &mut self,
        data: &ndarray::Array3<f32>,   // (n_samples, channels, timesteps)
        labels: &Array1<i64>,
        progress_callback: F,
    ) -> Result<SVMTrainingMetrics>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        let n_samples = data.shape()[0];
        let channels = data.shape()[1];
        let timesteps = data.shape()[2];

        // Ekstrak features
        let mut feat_rows: Vec<Vec<f32>> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let sample = data.slice(s![i, .., ..]).to_owned();
            let f = extract_features_svm(&sample);
            feat_rows.push(f.to_vec());
        }
        let n_features = feat_rows[0].len();
        self.n_features = n_features;

        // Build feature matrix
        let mut feat_mat = Array2::<f32>::zeros((n_samples, n_features));
        for (i, row) in feat_rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                feat_mat[[i, j]] = v;
            }
        }

        // Normalisasi features (z-score)
        let stats = FeatureStats::fit(&feat_mat);
        let feat_norm = stats.transform(&feat_mat);
        self.feature_stats = Some(stats);

        // Inisialisasi weights
        self.weights = vec![0.0f32; n_features];
        self.bias = 0.0;

        let lr = self.config.learning_rate;
        let c = self.config.c;
        let lambda = 1.0 / (c * n_samples as f32); // regularization strength
        let n_epochs = self.config.n_epochs;
        let batch_size = self.config.batch_size.min(n_samples);

        // Buat array SVM labels (-1/+1)
        let svm_labels: Vec<f32> = labels.iter().map(|&l| Self::to_svm_label(l)).collect();

        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut final_loss = 0.0f32;

        for epoch in 0..n_epochs {
            // Shuffle indices setiap epoch
            self.shuffle_indices(&mut indices);

            let mut epoch_loss = 0.0f32;

            // Mini-batch SGD
            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch = &indices[batch_start..batch_end];

                // Akumulasi gradient
                let mut grad_w = vec![0.0f32; n_features];
                let mut grad_b = 0.0f32;
                let mut batch_loss = 0.0f32;

                for &idx in batch {
                    let x_i: Vec<f32> = (0..n_features).map(|j| feat_norm[[idx, j]]).collect();
                    let y_i = svm_labels[idx];

                    // Decision value: w·x + b
                    let decision: f32 = self.weights.iter().zip(x_i.iter()).map(|(w, x)| w * x).sum::<f32>() + self.bias;

                    // Hinge loss: max(0, 1 - y * decision)
                    let margin = y_i * decision;
                    let loss = (1.0 - margin).max(0.0);
                    batch_loss += loss;

                    // Gradient of hinge loss
                    if margin < 1.0 {
                        // Misclassified or within margin
                        for j in 0..n_features {
                            grad_w[j] -= y_i * x_i[j];
                        }
                        grad_b -= y_i;
                    }
                }

                let batch_len = batch.len() as f32;
                epoch_loss += batch_loss;

                // Update weights: w = (1 - lr*lambda)*w - lr/batch * grad_w
                for j in 0..n_features {
                    self.weights[j] = (1.0 - lr * lambda) * self.weights[j]
                        - lr * grad_w[j] / batch_len;
                }
                self.bias -= lr * grad_b / batch_len;
            }

            final_loss = epoch_loss / n_samples as f32;
            progress_callback(epoch + 1, n_epochs);
        }

        // Hitung train accuracy
        let train_preds = self.predict_from_matrix(&feat_norm);
        let train_correct = train_preds.iter().zip(labels.iter()).filter(|(p, l)| p == l).count();
        let train_accuracy = train_correct as f32 / n_samples as f32;

        // Hitung support vectors (samples di dalam atau di luar margin)
        let n_support_vectors = self.count_support_vectors(&feat_norm, &svm_labels);

        let _ = (channels, timesteps); // suppress unused warnings

        Ok(SVMTrainingMetrics {
            n_epochs_trained: n_epochs,
            train_accuracy,
            n_support_vectors,
            final_loss,
        })
    }

    fn shuffle_indices(&self, indices: &mut Vec<usize>) {
        // Simple Fisher-Yates shuffle menggunakan rand
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(12345);
        let mut rng_state = seed;
        for i in (1..indices.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }
    }

    fn count_support_vectors(&self, feat_norm: &Array2<f32>, svm_labels: &[f32]) -> usize {
        let n = feat_norm.shape()[0];
        let mut count = 0;
        for i in 0..n {
            let x_i: Vec<f32> = (0..self.n_features).map(|j| feat_norm[[i, j]]).collect();
            let decision: f32 = self.weights.iter().zip(x_i.iter()).map(|(w, x)| w * x).sum::<f32>() + self.bias;
            let margin = svm_labels[i] * decision;
            if margin <= 1.0 {
                count += 1;
            }
        }
        count
    }

    /// Prediksi dari feature matrix yang sudah dinormalisasi
    fn predict_from_matrix(&self, feat_norm: &Array2<f32>) -> Array1<i64> {
        let n = feat_norm.shape()[0];
        let mut preds = Array1::<i64>::zeros(n);
        for i in 0..n {
            let x_i: Vec<f32> = (0..self.n_features).map(|j| feat_norm[[i, j]]).collect();
            let decision: f32 = self.weights.iter().zip(x_i.iter()).map(|(w, x)| w * x).sum::<f32>() + self.bias;
            preds[i] = Self::from_svm_label(decision);
        }
        preds
    }

    /// Prediksi dengan confidence score menggunakan sigmoid pada decision value
    pub fn predict_with_confidence(&self, data: &ndarray::Array3<f32>) -> Vec<(i64, f32, f32)> {
        let n_samples = data.shape()[0];
        let n_features = self.n_features;
        let stats = self.feature_stats.as_ref().expect("Model belum ditraining");

        let mut feat_mat = Array2::<f32>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            let sample = data.slice(s![i, .., ..]).to_owned();
            let f = extract_features_svm(&sample);
            for (j, &v) in f.iter().enumerate() {
                feat_mat[[i, j]] = v;
            }
        }

        let feat_norm = stats.transform(&feat_mat);
        (0..n_samples).map(|i| {
            let x_i: Vec<f32> = (0..n_features).map(|j| feat_norm[[i, j]]).collect();
            let decision: f32 = self.weights.iter().zip(x_i.iter()).map(|(w, x)| w * x).sum::<f32>() + self.bias;
            let class = Self::from_svm_label(decision);
            // sigmoid to convert decision value to pseudo-probability
            let p_high = 1.0 / (1.0 + (-decision).exp());
            let p_low = 1.0 - p_high;
            (class, p_high, p_low)
        }).collect()
    }

    /// Prediksi dari data mentah (n_samples, channels, timesteps)
    pub fn predict(&self, data: &ndarray::Array3<f32>) -> Array1<i64> {
        let n_samples = data.shape()[0];
        let n_features = self.n_features;
        let stats = self.feature_stats.as_ref().expect("Model belum ditraining");

        let mut feat_mat = Array2::<f32>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            let sample = data.slice(s![i, .., ..]).to_owned();
            let f = extract_features_svm(&sample);
            for (j, &v) in f.iter().enumerate() {
                feat_mat[[i, j]] = v;
            }
        }

        let feat_norm = stats.transform(&feat_mat);
        self.predict_from_matrix(&feat_norm)
    }

    /// Kembalikan magnitude weight per feature (sebagai "feature importance" analog)
    pub fn weight_magnitudes(&self) -> Vec<f32> {
        self.weights.iter().map(|w| w.abs()).collect()
    }

    /// Evaluasi pada N epoch pertama (untuk kurva akurasi vs epoch)
    /// Karena kita sudah selesai training, ini dihitung dari snapshot weights
    /// Sebagai gantinya, kita track accuracy_curve saat training via callback

    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let model: Self = serde_json::from_str(&contents)?;
        Ok(model)
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn config(&self) -> &SVMConfig {
        &self.config
    }
}

// ============================================
// TRAINING DENGAN ACCURACY CURVE TRACKING
// ============================================
impl CoffeeSVM {
    /// Training dengan tracking accuracy per epoch untuk plotting
    /// Mengembalikan (SVMTrainingMetrics, accuracy_curve: Vec<(epoch, train_acc, val_acc)>)
    pub fn fit_with_curve<F>(
        &mut self,
        train_data: &ndarray::Array3<f32>,
        train_labels: &Array1<i64>,
        val_data: &ndarray::Array3<f32>,
        val_labels: &Array1<i64>,
        progress_callback: F,
    ) -> Result<(SVMTrainingMetrics, Vec<(usize, f32, f32)>)>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        let n_samples = train_data.shape()[0];
        let n_val = val_data.shape()[0];

        // Ekstrak features untuk train
        let train_feat_mat = self.build_feature_matrix(train_data);
        let n_features = train_feat_mat.shape()[1];
        self.n_features = n_features;

        // Fit normalization pada train set
        let stats = FeatureStats::fit(&train_feat_mat);
        let train_norm = stats.transform(&train_feat_mat);
        self.feature_stats = Some(stats.clone());

        // Ekstrak dan normalisasi val features
        let val_feat_mat = self.build_feature_matrix(val_data);
        let val_norm = stats.transform(&val_feat_mat);

        // Inisialisasi weights
        self.weights = vec![0.0f32; n_features];
        self.bias = 0.0;

        let lr = self.config.learning_rate;
        let c = self.config.c;
        let lambda = 1.0 / (c * n_samples as f32);
        let n_epochs = self.config.n_epochs;
        let batch_size = self.config.batch_size.min(n_samples);

        let svm_train_labels: Vec<f32> = train_labels.iter().map(|&l| Self::to_svm_label(l)).collect();

        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut accuracy_curve: Vec<(usize, f32, f32)> = Vec::new();
        let mut final_loss = 0.0f32;

        // Checkpoints: setiap 10% epoch
        let checkpoint_interval = (n_epochs / 10).max(1);

        for epoch in 0..n_epochs {
            self.shuffle_indices(&mut indices);

            let mut epoch_loss = 0.0f32;

            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch = &indices[batch_start..batch_end];

                let mut grad_w = vec![0.0f32; n_features];
                let mut grad_b = 0.0f32;
                let mut batch_loss = 0.0f32;

                for &idx in batch {
                    let x_i: Vec<f32> = (0..n_features).map(|j| train_norm[[idx, j]]).collect();
                    let y_i = svm_train_labels[idx];
                    let decision: f32 = self.weights.iter().zip(x_i.iter()).map(|(w, x)| w * x).sum::<f32>() + self.bias;
                    let margin = y_i * decision;
                    let loss = (1.0 - margin).max(0.0);
                    batch_loss += loss;

                    if margin < 1.0 {
                        for j in 0..n_features {
                            grad_w[j] -= y_i * x_i[j];
                        }
                        grad_b -= y_i;
                    }
                }

                let batch_len = batch.len() as f32;
                epoch_loss += batch_loss;

                for j in 0..n_features {
                    self.weights[j] = (1.0 - lr * lambda) * self.weights[j]
                        - lr * grad_w[j] / batch_len;
                }
                self.bias -= lr * grad_b / batch_len;
            }

            final_loss = epoch_loss / n_samples as f32;
            progress_callback(epoch + 1, n_epochs);

            // Simpan checkpoint accuracy
            if (epoch + 1) % checkpoint_interval == 0 || epoch + 1 == n_epochs {
                let train_preds = self.predict_from_matrix(&train_norm);
                let train_correct = train_preds.iter().zip(train_labels.iter()).filter(|(p, l)| p == l).count();
                let train_acc = train_correct as f32 / n_samples as f32;

                let val_preds = self.predict_from_matrix(&val_norm);
                let val_correct = val_preds.iter().zip(val_labels.iter()).filter(|(p, l)| p == l).count();
                let val_acc = val_correct as f32 / n_val as f32;

                accuracy_curve.push((epoch + 1, train_acc, val_acc));
            }
        }

        // Final metrics
        let train_preds = self.predict_from_matrix(&train_norm);
        let train_correct = train_preds.iter().zip(train_labels.iter()).filter(|(p, l)| p == l).count();
        let train_accuracy = train_correct as f32 / n_samples as f32;
        let n_support_vectors = self.count_support_vectors(&train_norm, &svm_train_labels);

        let metrics = SVMTrainingMetrics {
            n_epochs_trained: n_epochs,
            train_accuracy,
            n_support_vectors,
            final_loss,
        };

        Ok((metrics, accuracy_curve))
    }

    fn build_feature_matrix(&self, data: &ndarray::Array3<f32>) -> Array2<f32> {
        let n_samples = data.shape()[0];
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let sample = data.slice(s![i, .., ..]).to_owned();
            let f = extract_features_svm(&sample);
            rows.push(f.to_vec());
        }
        let n_features = rows[0].len();
        let mut mat = Array2::<f32>::zeros((n_samples, n_features));
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                mat[[i, j]] = v;
            }
        }
        mat
    }
}
