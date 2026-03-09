// src/ml/models/mlp.rs - Multi-Layer Perceptron Coffee Quality Classifier
// Arsitektur: Input(48) → Dense(128, ReLU) → Dense(64, ReLU) → Output(2, Softmax)
// Feature extraction sama dengan RF/SVM (mean, std, min, max, range, median)

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

// ============================================
// FEATURE EXTRACTION (sama dengan rf/svm)
// ============================================
pub fn extract_features_mlp(sample: &Array2<f32>) -> Array1<f32> {
    let (channels, timesteps) = sample.dim();
    let mut features = Vec::with_capacity(channels * 6);
    for c in 0..channels {
        let ch = sample.slice(s![c, ..]);
        let vals: Vec<f32> = ch.iter().cloned().collect();
        let mean = vals.iter().sum::<f32>() / timesteps as f32;
        let var  = vals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / timesteps as f32;
        let std  = var.sqrt();
        let min  = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max  = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let mut sorted = vals.clone();
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
// ACTIVATION FUNCTIONS
// ============================================

#[inline] fn relu(x: f32) -> f32 { x.max(0.0) }
#[inline] fn relu_deriv(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum = exp.iter().sum::<f32>().max(1e-10);
    exp.iter().map(|&e| e / sum).collect()
}

// ============================================
// DENSE LAYER
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DenseLayer {
    weights: Vec<Vec<f32>>,  // (out, in)
    bias: Vec<f32>,          // (out)
    in_size: usize,
    out_size: usize,
    use_relu: bool,
}

impl DenseLayer {
    fn new(in_size: usize, out_size: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        // He initialization untuk ReLU, Xavier untuk linear
        let std = if use_relu {
            (2.0 / in_size as f32).sqrt()
        } else {
            (1.0 / in_size as f32).sqrt()
        };
        let normal = Normal::new(0.0f64, std as f64).unwrap();
        let weights = (0..out_size)
            .map(|_| (0..in_size).map(|_| normal.sample(&mut rng) as f32).collect())
            .collect();
        let bias = vec![0.0f32; out_size];
        Self { weights, bias, in_size, out_size, use_relu }
    }

    /// Forward: input (batch, in) → (pre_act, output) each (batch, out)
    fn forward(&self, input: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let batch = input.len();
        let mut pre_act = vec![vec![0.0f32; self.out_size]; batch];
        let mut output  = vec![vec![0.0f32; self.out_size]; batch];
        for b in 0..batch {
            for o in 0..self.out_size {
                let z = self.bias[o]
                    + (0..self.in_size).map(|i| self.weights[o][i] * input[b][i]).sum::<f32>();
                pre_act[b][o] = z;
                output[b][o] = if self.use_relu { relu(z) } else { z };
            }
        }
        (pre_act, output)
    }

    /// Backward: returns grad_input (batch, in), updates weights/bias
    fn backward(
        &mut self,
        input: &[Vec<f32>],
        pre_act: &[Vec<f32>],
        grad_output: &[Vec<f32>],
        lr: f32,
        clip: f32,
    ) -> Vec<Vec<f32>> {
        let batch = input.len();
        let scale = 1.0 / batch as f32;

        // Apply ReLU derivative to grad_output if needed
        let grad_act: Vec<Vec<f32>> = (0..batch).map(|b| {
            (0..self.out_size).map(|o| {
                let g = grad_output[b][o];
                if self.use_relu { g * relu_deriv(pre_act[b][o]) } else { g }
            }).collect()
        }).collect();

        let mut grad_input = vec![vec![0.0f32; self.in_size]; batch];
        let mut dw = vec![vec![0.0f32; self.in_size]; self.out_size];
        let mut db = vec![0.0f32; self.out_size];

        for b in 0..batch {
            for o in 0..self.out_size {
                let g = grad_act[b][o];
                db[o] += g * scale;
                for i in 0..self.in_size {
                    dw[o][i] += g * input[b][i] * scale;
                    grad_input[b][i] += g * self.weights[o][i];
                }
            }
        }

        for o in 0..self.out_size {
            self.bias[o] -= lr * db[o].clamp(-clip, clip);
            for i in 0..self.in_size {
                self.weights[o][i] -= lr * dw[o][i].clamp(-clip, clip);
            }
        }

        grad_input
    }
}

// ============================================
// MLP CONFIG
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPConfig {
    pub hidden1: usize,
    pub hidden2: usize,
    pub n_epochs: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub clip_grad: f32,
}

impl Default for MLPConfig {
    fn default() -> Self {
        Self {
            hidden1: 128,
            hidden2: 64,
            n_epochs: 150,
            learning_rate: 0.01,
            batch_size: 16,
            clip_grad: 5.0,
        }
    }
}

// ============================================
// TRAINING METRICS
// ============================================

#[derive(Debug, Clone)]
pub struct MLPTrainingMetrics {
    pub n_epochs_trained: usize,
    pub train_accuracy: f32,
    pub train_loss: f32,
}

// ============================================
// FEATURE NORMALIZATION (z-score)
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeatureNorm {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl FeatureNorm {
    fn fit(mat: &Array2<f32>) -> Self {
        let (n, d) = mat.dim();
        let mut mean = vec![0.0f32; d];
        let mut std  = vec![1.0f32; d];
        for j in 0..d {
            let col: Vec<f32> = (0..n).map(|i| mat[[i, j]]).collect();
            let m = col.iter().sum::<f32>() / n as f32;
            let v = col.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n as f32;
            mean[j] = m;
            std[j]  = v.sqrt().max(1e-8);
        }
        Self { mean, std }
    }

    fn transform_row(&self, row: &[f32]) -> Vec<f32> {
        row.iter().enumerate().map(|(j, &v)| (v - self.mean[j]) / self.std[j]).collect()
    }
}

// ============================================
// COFFEE MLP
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoffeeMLP {
    config: MLPConfig,
    layer1: DenseLayer,
    layer2: DenseLayer,
    layer_out: DenseLayer,
    norm: Option<FeatureNorm>,
}

impl CoffeeMLP {
    pub fn new(config: MLPConfig) -> Self {
        let n_features = 48; // 8 channels × 6 stats
        let h1 = config.hidden1;
        let h2 = config.hidden2;
        Self {
            layer1:    DenseLayer::new(n_features, h1, true),
            layer2:    DenseLayer::new(h1, h2, true),
            layer_out: DenseLayer::new(h2, 2, false),
            config,
            norm: None,
        }
    }

    // ── Feature extraction ───────────────────────────────────────
    fn build_feat_matrix(data: &Array3<f32>) -> Array2<f32> {
        let n = data.shape()[0];
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let sample = data.slice(s![i, .., ..]).to_owned();
            rows.push(extract_features_mlp(&sample).to_vec());
        }
        let d = rows[0].len();
        let mut mat = Array2::<f32>::zeros((n, d));
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                mat[[i, j]] = v;
            }
        }
        mat
    }

    fn mat_to_vecs(mat: &Array2<f32>, norm: &FeatureNorm) -> Vec<Vec<f32>> {
        let n = mat.shape()[0];
        (0..n).map(|i| {
            let row: Vec<f32> = (0..mat.shape()[1]).map(|j| mat[[i, j]]).collect();
            norm.transform_row(&row)
        }).collect()
    }

    // ── Forward (returns logits + caches for backprop) ───────────
    fn forward_batch(
        &mut self,
        inputs: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (pre1, out1) = self.layer1.forward(inputs);
        let (pre2, out2) = self.layer2.forward(&out1);
        let (pre_out, logits) = self.layer_out.forward(&out2);
        (pre1, out1, pre2, out2, pre_out, logits)
    }

    // ── One training step ────────────────────────────────────────
    fn train_step(
        &mut self,
        inputs: &[Vec<f32>],
        labels: &[i64],
        lr: f32,
        clip: f32,
    ) -> (f32, usize) {
        let batch = inputs.len();
        let (pre1, out1, pre2, out2, _pre_out, logits) = self.forward_batch(inputs);

        // Softmax + cross-entropy loss
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        let mut d_logits = vec![vec![0.0f32; 2]; batch];

        for b in 0..batch {
            let probs = softmax_vec(&logits[b]);
            let label = labels[b] as usize;
            total_loss += -(probs[label].max(1e-10).ln());
            if (probs[0] >= probs[1]) == (label == 0) { correct += 1; }
            // Gradient: d(CE+Softmax) / d(logit) = probs - one_hot
            for c in 0..2 { d_logits[b][c] = probs[c]; }
            d_logits[b][label] -= 1.0;
        }

        // Backprop
        let d2 = self.layer_out.backward(&out2, &vec![vec![0.0f32; 2]; batch], &d_logits, lr, clip);
        let d1 = self.layer2.backward(&out1, &pre2, &d2, lr, clip);
        let _  = self.layer1.backward(inputs, &pre1, &d1, lr, clip);

        (total_loss / batch as f32, correct)
    }

    // ── Predict (returns Array2 probabilities) ───────────────────
    pub fn predict(&mut self, data: &Array3<f32>) -> Array2<f32> {
        let norm = self.norm.as_ref().expect("Model belum ditraining").clone();
        let feat = Self::build_feat_matrix(data);
        let inputs = Self::mat_to_vecs(&feat, &norm);
        let n = inputs.len();
        let mut out = Array2::<f32>::zeros((n, 2));
        for (i, x) in inputs.iter().enumerate() {
            let (_, out1) = self.layer1.forward(&[x.clone()]);
            let (_, out2) = self.layer2.forward(&out1);
            let (_, logits) = self.layer_out.forward(&out2);
            let probs = softmax_vec(&logits[0]);
            out[[i, 0]] = probs[0];
            out[[i, 1]] = probs[1];
        }
        out
    }

    // ── Training with accuracy + loss curve ─────────────────────
    pub fn fit_with_curve<F>(
        &mut self,
        train_data: &Array3<f32>,
        train_labels: &Array1<i64>,
        val_data: &Array3<f32>,
        val_labels: &Array1<i64>,
        progress_callback: F,
    ) -> Result<(MLPTrainingMetrics, Vec<(usize, f32, f32)>, Vec<(usize, f32, f32)>)>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        let n_train = train_data.shape()[0];
        let n_val   = val_data.shape()[0];

        // Build feature matrices
        let train_feat = Self::build_feat_matrix(train_data);
        let val_feat   = Self::build_feat_matrix(val_data);

        // Fit normalization on train set only
        let norm = FeatureNorm::fit(&train_feat);
        self.norm = Some(norm.clone());

        let train_inputs: Vec<Vec<f32>> = Self::mat_to_vecs(&train_feat, &norm);
        let val_inputs:   Vec<Vec<f32>> = Self::mat_to_vecs(&val_feat,   &norm);
        let train_lbls: Vec<i64> = train_labels.iter().cloned().collect();
        let val_lbls:   Vec<i64> = val_labels.iter().cloned().collect();

        let n_epochs   = self.config.n_epochs;
        let batch_size = self.config.batch_size.min(n_train);
        let lr         = self.config.learning_rate;
        let clip       = self.config.clip_grad;
        let checkpoint = (n_epochs / 10).max(1);

        let mut indices: Vec<usize> = (0..n_train).collect();
        let mut accuracy_curve: Vec<(usize, f32, f32)> = Vec::new();
        let mut loss_curve:     Vec<(usize, f32, f32)> = Vec::new();
        let mut last_train_loss = 0.0f32;
        let mut last_train_acc  = 0.0f32;

        for epoch in 0..n_epochs {
            self.shuffle_indices(&mut indices);

            let mut ep_loss = 0.0f32;
            let mut ep_correct = 0usize;

            for batch_start in (0..n_train).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_train);
                let batch: Vec<Vec<f32>> = indices[batch_start..batch_end]
                    .iter().map(|&i| train_inputs[i].clone()).collect();
                let lbls: Vec<i64> = indices[batch_start..batch_end]
                    .iter().map(|&i| train_lbls[i]).collect();

                let (loss, correct) = self.train_step(&batch, &lbls, lr, clip);
                ep_loss    += loss * batch.len() as f32;
                ep_correct += correct;
            }

            last_train_loss = ep_loss / n_train as f32;
            last_train_acc  = ep_correct as f32 / n_train as f32;

            progress_callback(epoch + 1, n_epochs);

            if (epoch + 1) % checkpoint == 0 || epoch + 1 == n_epochs {
                // Val accuracy
                let val_correct: usize = val_inputs.iter().zip(val_lbls.iter()).filter(|(x, &l)| {
                    let (_, o1) = self.layer1.forward(&[(*x).clone()]);
                    let (_, o2) = self.layer2.forward(&o1);
                    let (_, lg) = self.layer_out.forward(&o2);
                    let p = softmax_vec(&lg[0]);
                    (p[0] >= p[1]) == (l == 0)
                }).count();
                let val_acc  = val_correct as f32 / n_val as f32;

                // Val loss
                let val_loss: f32 = val_inputs.iter().zip(val_lbls.iter()).map(|(x, &l)| {
                    let (_, o1) = self.layer1.forward(&[x.clone()]);
                    let (_, o2) = self.layer2.forward(&o1);
                    let (_, lg) = self.layer_out.forward(&o2);
                    let p = softmax_vec(&lg[0]);
                    -(p[l as usize].max(1e-10).ln())
                }).sum::<f32>() / n_val as f32;

                accuracy_curve.push((epoch + 1, last_train_acc, val_acc));
                loss_curve.push((epoch + 1, last_train_loss, val_loss));
            }
        }

        Ok((
            MLPTrainingMetrics {
                n_epochs_trained: n_epochs,
                train_accuracy: last_train_acc,
                train_loss: last_train_loss,
            },
            accuracy_curve,
            loss_curve,
        ))
    }

    fn shuffle_indices(&self, indices: &mut Vec<usize>) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now().duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64).unwrap_or(99);
        let mut rng = seed;
        for i in (1..indices.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }
    }

    pub fn config(&self) -> &MLPConfig { &self.config }

    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut f = File::create(path)?;
        f.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let mut f = File::open(path)?;
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        Ok(serde_json::from_str(&s)?)
    }
}
