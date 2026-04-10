// src/ml/models/lstm.rs - LSTM Coffee Quality Classifier
// Implementasi dari scratch: LSTM Cell + BPTT + Linear Classifier

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

// ============================================
// ACTIVATION FUNCTIONS
// ============================================

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid_deriv(y: f32) -> f32 {
    y * (1.0 - y)
}

#[inline]
fn tanh_deriv(y: f32) -> f32 {
    1.0 - y * y
}

fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Array1<f32> = logits.mapv(|x| (x - max).exp());
    let sum = exp.sum().max(1e-10);
    exp / sum
}

// ============================================
// LSTM CONFIG
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMConfig {
    /// Ukuran hidden state LSTM
    pub hidden_size: usize,
    /// Jumlah epoch training
    pub n_epochs: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Ukuran mini-batch
    pub batch_size: usize,
    /// Stride untuk mereduksi timesteps (3 → 100 steps dari 300)
    pub stride: usize,
    /// Gradient clipping threshold
    pub clip_grad: f32,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            hidden_size: 64,
            n_epochs: 50,
            learning_rate: 0.005,
            batch_size: 8,
            stride: 3,
            clip_grad: 5.0,
        }
    }
}

// ============================================
// TRAINING METRICS
// ============================================

#[derive(Debug, Clone)]
pub struct LSTMTrainingMetrics {
    pub n_epochs_trained: usize,
    pub train_accuracy: f32,
    pub train_loss: f32,
}

// ============================================
// LINEAR LAYER (classifier head)
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinearLayer {
    weights: Vec<Vec<f32>>, // (out_size, in_size)
    bias: Vec<f32>,         // (out_size)
    in_size: usize,
    out_size: usize,
}

impl LinearLayer {
    fn new(in_size: usize, out_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / in_size as f32).sqrt();
        let normal = Normal::new(0.0f64, std as f64).unwrap();

        let weights: Vec<Vec<f32>> = (0..out_size)
            .map(|_| (0..in_size).map(|_| normal.sample(&mut rng) as f32).collect())
            .collect();
        let bias = vec![0.0f32; out_size];

        Self { weights, bias, in_size, out_size }
    }

    /// Forward: input (batch, in_size) → output (batch, out_size)
    fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input.iter().map(|x| {
            (0..self.out_size).map(|o| {
                self.bias[o] + (0..self.in_size).map(|i| self.weights[o][i] * x[i]).sum::<f32>()
            }).collect()
        }).collect()
    }

    /// Backward: returns grad_input (batch, in_size), updates weights
    fn backward(&mut self, input: &[Vec<f32>], grad_output: &[Vec<f32>], lr: f32) -> Vec<Vec<f32>> {
        let batch = input.len();
        let mut grad_input = vec![vec![0.0f32; self.in_size]; batch];

        let mut dw = vec![vec![0.0f32; self.in_size]; self.out_size];
        let mut db = vec![0.0f32; self.out_size];

        for b in 0..batch {
            for o in 0..self.out_size {
                let g = grad_output[b][o];
                db[o] += g;
                for i in 0..self.in_size {
                    dw[o][i] += g * input[b][i];
                    grad_input[b][i] += g * self.weights[o][i];
                }
            }
        }

        let scale = 1.0 / batch as f32;
        for o in 0..self.out_size {
            self.bias[o] -= lr * db[o] * scale;
            for i in 0..self.in_size {
                self.weights[o][i] -= lr * dw[o][i] * scale;
            }
        }

        grad_input
    }
}

// ============================================
// LSTM CELL
// Gate ordering: input(i), forget(f), cell(g), output(o)
// W_combined: (4*H, I+H)
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LSTMCell {
    // Combined weight matrix (4*hidden, input+hidden)
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
    input_size: usize,
    hidden_size: usize,
}

/// Per-timestep cache needed for BPTT
#[derive(Clone)]
struct StepCache {
    x: Vec<f32>,       // input (input_size)
    h_prev: Vec<f32>,  // previous hidden (hidden_size)
    c_prev: Vec<f32>,  // previous cell (hidden_size)
    i_g: Vec<f32>,     // input gate after sigmoid (hidden_size)
    f_g: Vec<f32>,     // forget gate after sigmoid
    g_g: Vec<f32>,     // cell gate after tanh
    o_g: Vec<f32>,     // output gate after sigmoid
    c: Vec<f32>,       // cell state (hidden_size)
    h: Vec<f32>,       // hidden state (hidden_size)
}

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let combined = input_size + hidden_size;
        let std = (1.0 / (combined as f32)).sqrt();
        let normal = Normal::new(0.0f64, std as f64).unwrap();

        let w: Vec<Vec<f32>> = (0..4 * hidden_size)
            .map(|_| (0..combined).map(|_| normal.sample(&mut rng) as f32).collect())
            .collect();

        // Forget gate bias = 1.0 (helps with vanishing gradients)
        let mut b = vec![0.0f32; 4 * hidden_size];
        for j in hidden_size..2 * hidden_size {
            b[j] = 1.0;
        }

        Self { w, b, input_size, hidden_size }
    }

    /// Forward one timestep for one sample.
    fn step_forward(&self, x: &[f32], h_prev: &[f32], c_prev: &[f32]) -> StepCache {
        let h = self.hidden_size;
        let combined_size = self.input_size + h;

        // Concatenate [h_prev, x]
        let mut xh = Vec::with_capacity(combined_size);
        xh.extend_from_slice(h_prev);
        xh.extend_from_slice(x);

        // Compute all 4 gate pre-activations: z = W * xh + b
        let mut z = vec![0.0f32; 4 * h];
        for row in 0..4 * h {
            z[row] = self.b[row]
                + (0..combined_size).map(|c| self.w[row][c] * xh[c]).sum::<f32>();
        }

        // Apply activations
        let i_g: Vec<f32> = z[0..h].iter().map(|&v| sigmoid(v)).collect();
        let f_g: Vec<f32> = z[h..2*h].iter().map(|&v| sigmoid(v)).collect();
        let g_g: Vec<f32> = z[2*h..3*h].iter().map(|&v| v.tanh()).collect();
        let o_g: Vec<f32> = z[3*h..4*h].iter().map(|&v| sigmoid(v)).collect();

        // Cell and hidden states
        let c_new: Vec<f32> = (0..h).map(|j| f_g[j] * c_prev[j] + i_g[j] * g_g[j]).collect();
        let h_new: Vec<f32> = (0..h).map(|j| o_g[j] * c_new[j].tanh()).collect();

        StepCache {
            x: x.to_vec(),
            h_prev: h_prev.to_vec(),
            c_prev: c_prev.to_vec(),
            i_g,
            f_g,
            g_g,
            o_g,
            c: c_new,
            h: h_new,
        }
    }

    /// BPTT backward for one timestep.
    /// Returns (dh_prev, dc_prev, dW accumulated, db accumulated)
    fn step_backward(
        &self,
        cache: &StepCache,
        dh: &[f32],   // gradient of loss w.r.t. h_t
        dc_next: &[f32], // gradient flowing from c_{t+1} (upstream cell grad)
    ) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
        let h = self.hidden_size;
        let combined_size = self.input_size + h;

        // dh_t flows through o_t * tanh(c_t), and dc_next from future cell
        // d(tanh(c_t)) = dh * o_t
        let tanh_c: Vec<f32> = cache.c.iter().map(|&v| v.tanh()).collect();
        let d_tanh_c: Vec<f32> = (0..h).map(|j| dh[j] * cache.o_g[j]).collect();

        // dc_t = d_tanh_c * tanh_deriv(tanh(c_t)) + dc_next * f_{t+1}
        // (dc_next already includes the f_{t+1} multiplication from the next step)
        let dc: Vec<f32> = (0..h).map(|j| d_tanh_c[j] * tanh_deriv(tanh_c[j]) + dc_next[j]).collect();

        // Gate gradients
        let d_o: Vec<f32> = (0..h).map(|j| dh[j] * tanh_c[j] * sigmoid_deriv(cache.o_g[j])).collect();
        let d_i: Vec<f32> = (0..h).map(|j| dc[j] * cache.g_g[j] * sigmoid_deriv(cache.i_g[j])).collect();
        let d_f: Vec<f32> = (0..h).map(|j| dc[j] * cache.c_prev[j] * sigmoid_deriv(cache.f_g[j])).collect();
        let d_g: Vec<f32> = (0..h).map(|j| dc[j] * cache.i_g[j] * tanh_deriv(cache.g_g[j])).collect();

        // Concatenate gate grads: [d_i | d_f | d_g | d_o]
        let mut d_gates = Vec::with_capacity(4 * h);
        d_gates.extend_from_slice(&d_i);
        d_gates.extend_from_slice(&d_f);
        d_gates.extend_from_slice(&d_g);
        d_gates.extend_from_slice(&d_o);

        // Concatenate [h_prev, x]
        let mut xh = Vec::with_capacity(combined_size);
        xh.extend_from_slice(&cache.h_prev);
        xh.extend_from_slice(&cache.x);

        // dW = d_gates (outer) xh
        let mut dw = vec![vec![0.0f32; combined_size]; 4 * h];
        for row in 0..4 * h {
            for col in 0..combined_size {
                dw[row][col] = d_gates[row] * xh[col];
            }
        }

        // db = d_gates
        let db = d_gates.clone();

        // Gradient to previous hidden state (first h cols of W^T @ d_gates)
        let mut dh_prev = vec![0.0f32; h];
        for j in 0..h {
            for row in 0..4 * h {
                dh_prev[j] += self.w[row][j] * d_gates[row];
            }
        }

        // dc_prev = dc * f_t
        let dc_prev: Vec<f32> = (0..h).map(|j| dc[j] * cache.f_g[j]).collect();

        (dh_prev, dc_prev, dw, db)
    }

    fn apply_gradients(&mut self, dw: &[Vec<f32>], db: &[f32], lr: f32, clip: f32) {
        let combined = self.input_size + self.hidden_size;
        for row in 0..4 * self.hidden_size {
            self.b[row] -= lr * db[row].clamp(-clip, clip);
            for col in 0..combined {
                self.w[row][col] -= lr * dw[row][col].clamp(-clip, clip);
            }
        }
    }
}

// ============================================
// COFFEE LSTM
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoffeeLSTM {
    config: LSTMConfig,
    cell: LSTMCell,
    output_layer: LinearLayer,
    n_classes: usize,
}

impl CoffeeLSTM {
    pub fn new(config: LSTMConfig) -> Self {
        let n_channels = 8; // sensor channels
        let hidden_size = config.hidden_size;
        let cell = LSTMCell::new(n_channels, hidden_size);
        let output_layer = LinearLayer::new(hidden_size, 2);

        Self {
            config,
            cell,
            output_layer,
            n_classes: 2,
        }
    }

    /// Forward pass for a single sample.
    /// Returns (output logits, caches for BPTT)
    fn forward_sample(&self, sequence: &Array2<f32>) -> (Vec<f32>, Vec<StepCache>) {
        // sequence: (channels, timesteps), already strided
        let timesteps = sequence.shape()[1];
        let h = self.config.hidden_size;

        let mut h_state = vec![0.0f32; h];
        let mut c_state = vec![0.0f32; h];
        let mut caches = Vec::with_capacity(timesteps);

        for t in 0..timesteps {
            let x: Vec<f32> = (0..sequence.shape()[0]).map(|c| sequence[[c, t]]).collect();
            let cache = self.cell.step_forward(&x, &h_state, &c_state);
            h_state = cache.h.clone();
            c_state = cache.c.clone();
            caches.push(cache);
        }

        // Output layer: h_T → logits
        let logits_batch = self.output_layer.forward(&[h_state]);
        (logits_batch.into_iter().next().unwrap(), caches)
    }

    /// Predict probabilities for a batch.
    /// data: (n_samples, channels, timesteps)
    pub fn predict(&self, data: &Array3<f32>) -> Array2<f32> {
        let n = data.shape()[0];
        let stride = self.config.stride.max(1);
        let timesteps_orig = data.shape()[2];
        let effective_t = (timesteps_orig + stride - 1) / stride;

        let mut out = Array2::<f32>::zeros((n, self.n_classes));

        for i in 0..n {
            let sample_full = data.slice(s![i, .., ..]).to_owned();
            // Apply stride
            let sample = self.apply_stride(&sample_full, effective_t, stride);
            let (logits, _) = self.forward_sample(&sample);
            let probs = softmax(&Array1::from(logits));
            for c in 0..self.n_classes {
                out[[i, c]] = probs[c];
            }
        }
        out
    }

    fn apply_stride(&self, sample: &Array2<f32>, effective_t: usize, stride: usize) -> Array2<f32> {
        let channels = sample.shape()[0];
        let mut out = Array2::<f32>::zeros((channels, effective_t));
        for t in 0..effective_t {
            let src = (t * stride).min(sample.shape()[1] - 1);
            for c in 0..channels {
                out[[c, t]] = sample[[c, src]];
            }
        }
        out
    }

    /// Train with accuracy curve tracking.
    pub fn fit_with_curve<F>(
        &mut self,
        train_data: &Array3<f32>,
        train_labels: &Array1<i64>,
        val_data: &Array3<f32>,
        val_labels: &Array1<i64>,
        progress_callback: F,
    ) -> Result<(LSTMTrainingMetrics, Vec<(usize, f32, f32)>, Vec<(usize, f32, f32)>)>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        let n_train = train_data.shape()[0];
        let _n_val = val_data.shape()[0];
        let stride = self.config.stride.max(1);
        let timesteps_orig = train_data.shape()[2];
        let effective_t = (timesteps_orig + stride - 1) / stride;
        let n_epochs = self.config.n_epochs;
        let batch_size = self.config.batch_size.min(n_train);
        let lr = self.config.learning_rate;
        let clip = self.config.clip_grad;

        let mut accuracy_curve: Vec<(usize, f32, f32)> = Vec::new();
        let mut loss_curve: Vec<(usize, f32, f32)> = Vec::new();
        let _checkpoint_interval = (n_epochs / 10).max(1);

        let mut indices: Vec<usize> = (0..n_train).collect();
        let mut final_train_loss = 0.0f32;
        let mut final_train_acc = 0.0f32;

        for epoch in 0..n_epochs {
            self.shuffle_indices(&mut indices);

            let mut epoch_loss = 0.0f32;
            let mut epoch_correct = 0usize;

            for batch_start in (0..n_train).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_train);
                let batch_idx = &indices[batch_start..batch_end];
                let bsz = batch_idx.len();

                // Accumulate gradients over batch
                let h_size = self.config.hidden_size;
                let combined = 8 + h_size; // 8 channels
                let mut dw_acc = vec![vec![0.0f32; combined]; 4 * h_size];
                let mut db_acc = vec![0.0f32; 4 * h_size];
                let mut dw_out_acc: Vec<(Vec<Vec<f32>>, Vec<f32>)> = Vec::new();

                let mut batch_loss = 0.0f32;
                let mut batch_correct = 0usize;

                for &idx in batch_idx {
                    let sample_full = train_data.slice(s![idx, .., ..]).to_owned();
                    let sample = self.apply_stride(&sample_full, effective_t, stride);
                    let label = train_labels[idx] as usize;

                    // Forward pass
                    let (logits, caches) = self.forward_sample(&sample);
                    let probs = softmax(&Array1::from(logits.clone()));

                    // Cross-entropy loss
                    let loss = -(probs[label].max(1e-10).ln());
                    batch_loss += loss;

                    // Prediction
                    let pred = if probs[0] >= probs[1] { 0 } else { 1 };
                    if pred == label { batch_correct += 1; }

                    // Gradient of cross-entropy + softmax
                    let mut d_logits = probs.to_vec();
                    d_logits[label] -= 1.0;

                    // Get last hidden state
                    let h_last = caches.last().unwrap().h.clone();

                    // Backward through output layer
                    let d_h_last = self.output_layer.backward(
                        &[h_last],
                        &[d_logits.clone()],
                        0.0, // accumulate manually
                    );
                    // Manually accumulate output layer gradients
                    dw_out_acc.push((
                        {
                            let h_last2 = caches.last().unwrap().h.clone();
                            let out_size = self.n_classes;
                            let in_size = h_size;
                            let mut dw = vec![vec![0.0f32; in_size]; out_size];
                            for o in 0..out_size {
                                for i in 0..in_size {
                                    dw[o][i] = d_logits[o] * h_last2[i];
                                }
                            }
                            dw
                        },
                        d_logits.clone(),
                    ));

                    // BPTT
                    let mut dh = d_h_last[0].clone();
                    let mut dc = vec![0.0f32; h_size];

                    for t in (0..caches.len()).rev() {
                        let (dh_prev, dc_prev, dw_step, db_step) =
                            self.cell.step_backward(&caches[t], &dh, &dc);

                        // Accumulate
                        for row in 0..4 * h_size {
                            db_acc[row] += db_step[row];
                            for col in 0..combined {
                                dw_acc[row][col] += dw_step[row][col];
                            }
                        }

                        dh = dh_prev;
                        dc = dc_prev;
                    }
                }

                epoch_loss += batch_loss;
                epoch_correct += batch_correct;

                // Average gradients over batch
                let scale = 1.0 / bsz as f32;
                for row in 0..4 * h_size {
                    db_acc[row] *= scale;
                    for col in 0..combined {
                        dw_acc[row][col] *= scale;
                    }
                }

                // Update LSTM weights
                self.cell.apply_gradients(&dw_acc, &db_acc, lr, clip);

                // Update output layer weights (average over batch)
                if !dw_out_acc.is_empty() {
                    let out_size = self.n_classes;
                    let in_size = h_size;
                    let mut dw_out_avg = vec![vec![0.0f32; in_size]; out_size];
                    let mut db_out_avg = vec![0.0f32; out_size];
                    for (dw, db) in &dw_out_acc {
                        for o in 0..out_size {
                            db_out_avg[o] += db[o] * scale;
                            for i in 0..in_size {
                                dw_out_avg[o][i] += dw[o][i] * scale;
                            }
                        }
                    }
                    for o in 0..out_size {
                        self.output_layer.bias[o] -= lr * db_out_avg[o].clamp(-clip, clip);
                        for i in 0..in_size {
                            self.output_layer.weights[o][i] -= lr * dw_out_avg[o][i].clamp(-clip, clip);
                        }
                    }
                }
            }

            final_train_loss = epoch_loss / n_train as f32;
            final_train_acc = epoch_correct as f32 / n_train as f32;

            progress_callback(epoch + 1, n_epochs);

            // Checkpoint accuracy and loss (setiap 5 epoch, max 100)
            if epoch + 1 <= 100 && ((epoch + 1) % 5 == 1 || epoch + 1 == n_epochs) {
                let val_acc = self.eval_accuracy(val_data, val_labels, effective_t, stride);
                let val_loss = self.eval_loss(val_data, val_labels, effective_t, stride);
                accuracy_curve.push((epoch + 1, final_train_acc, val_acc));
                loss_curve.push((epoch + 1, final_train_loss, val_loss));
            }
        }

        Ok((
            LSTMTrainingMetrics {
                n_epochs_trained: n_epochs,
                train_accuracy: final_train_acc,
                train_loss: final_train_loss,
            },
            accuracy_curve,
            loss_curve,
        ))
    }

    fn eval_accuracy(
        &self,
        data: &Array3<f32>,
        labels: &Array1<i64>,
        effective_t: usize,
        stride: usize,
    ) -> f32 {
        let n = data.shape()[0];
        let mut correct = 0;
        for i in 0..n {
            let sample_full = data.slice(s![i, .., ..]).to_owned();
            let sample = self.apply_stride(&sample_full, effective_t, stride);
            let (logits, _) = self.forward_sample(&sample);
            let probs = softmax(&Array1::from(logits));
            let pred = if probs[0] >= probs[1] { 0 } else { 1 };
            if pred == labels[i] as usize { correct += 1; }
        }
        correct as f32 / n as f32
    }

    fn eval_loss(
        &self,
        data: &Array3<f32>,
        labels: &Array1<i64>,
        effective_t: usize,
        stride: usize,
    ) -> f32 {
        let n = data.shape()[0];
        let mut total_loss = 0.0f32;
        for i in 0..n {
            let sample_full = data.slice(s![i, .., ..]).to_owned();
            let sample = self.apply_stride(&sample_full, effective_t, stride);
            let (logits, _) = self.forward_sample(&sample);
            let probs = softmax(&Array1::from(logits));
            let label = labels[i] as usize;
            total_loss += -(probs[label].max(1e-10).ln());
        }
        total_loss / n as f32
    }

    fn shuffle_indices(&self, indices: &mut Vec<usize>) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(42);
        let mut rng = seed;
        for i in (1..indices.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }
    }

    pub fn config(&self) -> &LSTMConfig {
        &self.config
    }

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
}
