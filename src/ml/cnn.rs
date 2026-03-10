// src/cnn.rs - WITH TEMPERATURE SCALING FOR CALIBRATED CONFIDENCE

use ndarray::{Array1, Array2, Array3, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use anyhow::Result;

// ============================================
// ACTIVATION FUNCTIONS
// ============================================

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[allow(dead_code)]
fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

// ✅ TEMPERATURE SCALING SOFTMAX
fn softmax_with_temperature(logits: &Array1<f32>, temperature: f32) -> Array1<f32> {
    // Divide logits by temperature BEFORE softmax
    let scaled_logits = logits.mapv(|x| x / temperature);
    let max = scaled_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits = scaled_logits.mapv(|x| (x - max).exp());
    let sum_exp = exp_logits.sum();
    
    if sum_exp < 1e-10 {
        let n = logits.len();
        return Array1::from_elem(n, 1.0 / n as f32);
    }
    
    exp_logits / sum_exp
}

fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    softmax_with_temperature(logits, 1.0)
}

// ============================================
// CONV1D LAYER WITH BACKPROPAGATION
// ============================================

pub struct Conv1D {
    pub weights: Array3<f32>,
    pub bias: Array1<f32>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    last_input: Option<Array3<f32>>,
    last_output: Option<Array3<f32>>,
    // im2col matrix stored for backward pass: [batch, col_size, out_length]
    last_col: Option<Array3<f32>>,
}

impl Conv1D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (in_channels * kernel_size) as f32).sqrt() * 0.1;
        let normal = Normal::new(0.0, std).unwrap();
        
        let weights = Array3::from_shape_fn(
            (out_channels, in_channels, kernel_size),
            |_| normal.sample(&mut rng)
        );
        let bias = Array1::zeros(out_channels);
        
        Self {
            weights,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            last_input: None,
            last_output: None,
            last_col: None,
        }
    }
    
    pub fn forward(&mut self, input: &Array3<f32>) -> Array3<f32> {
        self.last_input = Some(input.clone());

        let (batch_size, in_ch, input_length) = input.dim();
        let out_length = (input_length - self.kernel_size) / self.stride + 1;
        let col_size = in_ch * self.kernel_size;

        // Build im2col matrix: col[b, c*kernel+k, pos] = input[b, c, pos*stride+k]
        // Shape: [batch, col_size, out_length]
        let mut col = Array3::<f32>::zeros((batch_size, col_size, out_length));
        for b in 0..batch_size {
            for c in 0..in_ch {
                for k in 0..self.kernel_size {
                    let col_idx = c * self.kernel_size + k;
                    for pos in 0..out_length {
                        col[[b, col_idx, pos]] = input[[b, c, pos * self.stride + k]];
                    }
                }
            }
        }

        // weights_2d: [out_ch, col_size]
        let weights_2d = self.weights
            .view()
            .into_shape((self.out_channels, col_size))
            .expect("Conv1D weight reshape failed");

        // output = weights_2d · col[b] + bias, then ReLU
        // weights_2d: [out_ch, col_size], col[b]: [col_size, out_length] → [out_ch, out_length]
        let mut output = Array3::<f32>::zeros((batch_size, self.out_channels, out_length));
        for b in 0..batch_size {
            let col_b = col.slice(s![b, .., ..]);           // [col_size, out_length]
            let out_b = weights_2d.dot(&col_b);             // [out_ch, out_length]
            for oc in 0..self.out_channels {
                for pos in 0..out_length {
                    output[[b, oc, pos]] = relu(out_b[[oc, pos]] + self.bias[oc]);
                }
            }
        }

        self.last_output = Some(output.clone());
        self.last_col = Some(col);
        output
    }
    
    pub fn backward(&mut self, grad_output: &Array3<f32>, learning_rate: f32) -> Array3<f32> {
        let input = self.last_input.as_ref().unwrap();
        let output = self.last_output.as_ref().unwrap();
        let col = self.last_col.as_ref().unwrap();

        let (batch_size, in_ch, _input_length) = input.dim();
        let out_length = grad_output.shape()[2];
        let col_size = in_ch * self.kernel_size;

        // Apply ReLU gradient: zero out positions where output ≤ 0
        let mut grad_relu = grad_output.clone();
        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for pos in 0..out_length {
                    if output[[b, oc, pos]] <= 0.0 {
                        grad_relu[[b, oc, pos]] = 0.0;
                    }
                }
            }
        }

        let weights_2d = self.weights
            .view()
            .into_shape((self.out_channels, col_size))
            .expect("Conv1D weight reshape failed");

        // Accumulate grad_weights and grad_col across the batch using matrix multiply
        // grad_w += grad_relu[b] · col[b].T  →  [out_ch, out_length] · [out_length, col_size] = [out_ch, col_size]
        // grad_col[b] = weights_2d.T · grad_relu[b]  →  [col_size, out_ch] · [out_ch, out_length] = [col_size, out_length]
        let mut grad_weights_2d = Array2::<f32>::zeros((self.out_channels, col_size));
        let mut grad_bias = Array1::<f32>::zeros(self.out_channels);
        let mut grad_col = Array3::<f32>::zeros(col.dim());

        for b in 0..batch_size {
            let grad_b = grad_relu.slice(s![b, .., ..]); // [out_ch, out_length]
            let col_b  = col.slice(s![b, .., ..]);        // [col_size, out_length]

            grad_weights_2d += &grad_b.dot(&col_b.t());  // [out_ch, col_size]
            let gc_b = weights_2d.t().dot(&grad_b);       // [col_size, out_length]
            grad_col.slice_mut(s![b, .., ..]).assign(&gc_b);

            for oc in 0..self.out_channels {
                grad_bias[oc] += grad_b.slice(s![oc, ..]).sum();
            }
        }

        // Average over batch and update weights with gradient clipping
        grad_weights_2d /= batch_size as f32;
        grad_bias /= batch_size as f32;

        let mut weights_2d_mut = self.weights
            .view_mut()
            .into_shape((self.out_channels, col_size))
            .expect("Conv1D weight reshape (mut) failed");
        for oc in 0..self.out_channels {
            for ci in 0..col_size {
                weights_2d_mut[[oc, ci]] -= learning_rate * grad_weights_2d[[oc, ci]].clamp(-10.0, 10.0);
            }
            self.bias[oc] -= learning_rate * grad_bias[oc].clamp(-10.0, 10.0);
        }

        // col2im: scatter grad_col back into grad_input
        let mut grad_input = Array3::<f32>::zeros(input.dim());
        for b in 0..batch_size {
            for c in 0..in_ch {
                for k in 0..self.kernel_size {
                    let col_idx = c * self.kernel_size + k;
                    for pos in 0..out_length {
                        grad_input[[b, c, pos * self.stride + k]] += grad_col[[b, col_idx, pos]];
                    }
                }
            }
        }

        grad_input
    }
}

// ============================================
// MAXPOOL1D LAYER WITH BACKPROPAGATION
// ============================================

pub struct MaxPool1D {
    pub kernel_size: usize,
    pub stride: usize,
    last_indices: Option<Array3<usize>>,
}

impl MaxPool1D {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
            last_indices: None,
        }
    }
    
    pub fn forward(&mut self, input: &Array3<f32>) -> Array3<f32> {
        let (batch_size, channels, input_length) = input.dim();
        let output_length = (input_length - self.kernel_size) / self.stride + 1;
        
        let mut output = Array3::zeros((batch_size, channels, output_length));
        let mut indices = Array3::zeros((batch_size, channels, output_length));
        
        for b in 0..batch_size {
            for c in 0..channels {
                for pos in 0..output_length {
                    let start = pos * self.stride;
                    let end = (start + self.kernel_size).min(input_length);
                    
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = start;
                    
                    for i in start..end {
                        if input[[b, c, i]] > max_val {
                            max_val = input[[b, c, i]];
                            max_idx = i;
                        }
                    }
                    
                    output[[b, c, pos]] = max_val;
                    indices[[b, c, pos]] = max_idx;
                }
            }
        }
        
        self.last_indices = Some(indices);
        output
    }
    
    pub fn backward(&self, grad_output: &Array3<f32>, input_shape: (usize, usize, usize)) -> Array3<f32> {
        let indices = self.last_indices.as_ref().unwrap();
        let mut grad_input: Array3<f32> = Array3::zeros(input_shape);
        
        let (batch_size, channels, output_length) = grad_output.dim();
        
        for b in 0..batch_size {
            for c in 0..channels {
                for pos in 0..output_length {
                    let max_idx = indices[[b, c, pos]];
                    grad_input[[b, c, max_idx]] += grad_output[[b, c, pos]];
                }
            }
        }
        
        grad_input
    }
}

// ============================================
// FULLY CONNECTED LAYER WITH BACKPROPAGATION
// ============================================

pub struct FullyConnected {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    last_input: Option<Array1<f32>>,
    last_output: Option<Array1<f32>>,
}

impl FullyConnected {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / input_size as f32).sqrt() * 0.1;
        let normal = Normal::new(0.0, std).unwrap();
        
        let weights = Array2::from_shape_fn((output_size, input_size), |_| normal.sample(&mut rng));
        let bias = Array1::zeros(output_size);
        
        Self {
            weights,
            bias,
            last_input: None,
            last_output: None,
        }
    }
    
    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        self.last_input = Some(input.clone());
        let output = self.weights.dot(input) + &self.bias;
        self.last_output = Some(output.clone());
        output
    }
    
    pub fn backward(&mut self, grad_output: &Array1<f32>, learning_rate: f32, apply_relu: bool) -> Array1<f32> {
        let input = self.last_input.as_ref().unwrap();
        let output = self.last_output.as_ref().unwrap();
        
        let grad = if apply_relu {
            grad_output.iter()
                .zip(output.iter())
                .map(|(g, o)| if *o > 0.0 { *g } else { 0.0 })
                .collect::<Vec<_>>()
        } else {
            grad_output.to_vec()
        };
        
        let grad_array = Array1::from_vec(grad);
        
        let mut grad_weights: Array2<f32> = Array2::zeros(self.weights.dim());
        let mut grad_bias: Array1<f32> = Array1::zeros(self.bias.len());
        
        for i in 0..self.weights.shape()[0] {
            grad_bias[i] = grad_array[i];
            for j in 0..self.weights.shape()[1] {
                grad_weights[[i, j]] = grad_array[i] * input[j];
            }
        }
        
        let grad_input = self.weights.t().dot(&grad_array);
        
        for i in 0..self.weights.shape()[0] {
            self.bias[i] -= learning_rate * grad_bias[i].clamp(-10.0, 10.0);
            for j in 0..self.weights.shape()[1] {
                self.weights[[i, j]] -= learning_rate * grad_weights[[i, j]].clamp(-10.0, 10.0);
            }
        }
        
        grad_input
    }
}

// ============================================
// CNN MODEL WITH TEMPERATURE SCALING
// ============================================

pub struct CoffeeCNN {
    conv1: Conv1D,
    pool1: MaxPool1D,
    conv2: Conv1D,
    pool2: MaxPool1D,
    fc1: FullyConnected,
    fc2: FullyConnected,
    conv1_output_shape: Option<(usize, usize, usize)>,
    pool1_output_shape: Option<(usize, usize, usize)>,
    conv2_output_shape: Option<(usize, usize, usize)>,
    pub temperature: f32, // ✅ TEMPERATURE PARAMETER
}

impl CoffeeCNN {
    pub fn new() -> Self {
        println!("\n✅ Model Architecture:");
        println!("  Conv1: 8 → 32 (kernel=7) + ReLU + MaxPool");
        println!("  Conv2: 32 → 64 (kernel=5) + ReLU + MaxPool");
        println!("  FC1: 4544 → 128 + ReLU");
        println!("  FC2: 128 → 2");
        println!("  Output: Temperature-Scaled Softmax (T=1.0)");
        println!("  Calibration: Enabled\n");
        
        Self {
            conv1: Conv1D::new(8, 32, 7),
            pool1: MaxPool1D::new(2, 2),
            conv2: Conv1D::new(32, 64, 5),
            pool2: MaxPool1D::new(2, 2),
            fc1: FullyConnected::new(4544, 128),
            fc2: FullyConnected::new(128, 2),
            conv1_output_shape: None,
            pool1_output_shape: None,
            conv2_output_shape: None,
            temperature: 1.0, // Start with T=1.0
        }
    }
    
    pub fn forward(&mut self, input: &Array3<f32>) -> Array2<f32> {
        let batch_size = input.shape()[0];
        
        let mut x = self.conv1.forward(input);
        self.conv1_output_shape = Some(x.dim());
        
        x = self.pool1.forward(&x);
        self.pool1_output_shape = Some(x.dim());
        
        x = self.conv2.forward(&x);
        self.conv2_output_shape = Some(x.dim());
        
        x = self.pool2.forward(&x);
        
        let mut output = Array2::zeros((batch_size, 2));
        
        for b in 0..batch_size {
            let flat = x.slice(ndarray::s![b, .., ..]).iter().cloned().collect::<Vec<_>>();
            let flat_array = Array1::from_vec(flat);
            
            let fc1_out = self.fc1.forward(&flat_array);
            let relu_out = fc1_out.mapv(relu);
            let logits = self.fc2.forward(&relu_out);
            
            // ✅ USE TEMPERATURE SCALING
            let probs = softmax_with_temperature(&logits, self.temperature);
            output.row_mut(b).assign(&probs);
        }
        
        output
    }
    
    pub fn predict(&mut self, inputs: &Array3<f32>) -> Array2<f32> {
        self.forward(inputs)
    }
    
    // ✅ CALIBRATE TEMPERATURE ON VALIDATION SET
    pub fn calibrate_temperature(&mut self, val_data: &Array3<f32>, val_labels: &Array1<i64>) {
        println!("\n🔧 Calibrating Temperature Parameter...");
        
        let num_samples = val_data.shape()[0];
        if num_samples == 0 {
            println!("⚠️  No validation data for calibration");
            return;
        }
        
        // Store original temperature
        let original_temp = self.temperature;
        
        // Try different temperature values
        let temp_candidates = vec![0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0];
        let mut best_temp = 1.0;
        let mut best_nll = f32::INFINITY;
        
        for &temp in &temp_candidates {
            self.temperature = temp;
            
            // Compute negative log-likelihood on validation set
            let mut total_nll = 0.0;
            
            for i in 0..num_samples {
                let sample = val_data.slice(ndarray::s![i..i+1, .., ..]).to_owned();
                let probs = self.forward(&sample);
                
                let label = val_labels[i] as usize;
                let prob = probs[[0, label]].max(1e-10);
                let nll = -prob.ln();
                
                if nll.is_finite() {
                    total_nll += nll;
                }
            }
            
            let avg_nll = total_nll / num_samples as f32;
            println!("  T={:.1} → NLL={:.4}", temp, avg_nll);
            
            if avg_nll < best_nll {
                best_nll = avg_nll;
                best_temp = temp;
            }
        }
        
        // Set best temperature
        self.temperature = best_temp;
        
        println!("\n✅ Calibration Complete!");
        println!("  Original T: {:.2}", original_temp);
        println!("  Optimal T: {:.2}", best_temp);
        println!("  Best NLL: {:.4}", best_nll);
        
        if best_temp > 1.5 {
            println!("  → Model was OVERCONFIDENT, temperature increased");
        } else if best_temp < 0.9 {
            println!("  → Model was UNDERCONFIDENT, temperature decreased");
        } else {
            println!("  → Model confidence is well-calibrated");
        }
    }
    
    pub fn train_step(&mut self, inputs: &Array3<f32>, labels: &Array1<i64>, learning_rate: f32) -> (f32, f32) {
        let batch_size = inputs.shape()[0];
        
        let mut x = self.conv1.forward(inputs);
        self.conv1_output_shape = Some(x.dim());
        
        x = self.pool1.forward(&x);
        self.pool1_output_shape = Some(x.dim());
        
        x = self.conv2.forward(&x);
        self.conv2_output_shape = Some(x.dim());
        
        x = self.pool2.forward(&x);
        let pool2_output = x.clone();
        
        let mut total_loss = 0.0;
        let mut correct = 0;
        
        let mut grad_pool2_output: Array3<f32> = Array3::zeros(pool2_output.dim());
        
        for b in 0..batch_size {
            let flat = pool2_output.slice(ndarray::s![b, .., ..]).iter().cloned().collect::<Vec<_>>();
            let flat_array = Array1::from_vec(flat);
            
            let fc1_out = self.fc1.forward(&flat_array);
            let fc1_relu = fc1_out.mapv(relu);
            let logits = self.fc2.forward(&fc1_relu);
            
            // During training, use normal softmax (T=1.0)
            let probs = softmax(&logits);
            
            let label = labels[b] as usize;
            let prob = probs[label].max(1e-10);
            let loss = -prob.ln();
            
            if loss.is_finite() {
                total_loss += loss;
            }
            
            let pred_class = if probs[0] > probs[1] { 0 } else { 1 };
            if pred_class == label {
                correct += 1;
            }
            
            let mut grad_logits = Array1::zeros(2);
            for c in 0..2 {
                grad_logits[c] = if c == label { probs[c] - 1.0 } else { probs[c] };
            }
            
            let grad_fc1_relu = self.fc2.backward(&grad_logits, learning_rate, false);
            let grad_flat = self.fc1.backward(&grad_fc1_relu, learning_rate, true);
            
            let (_, channels, width) = pool2_output.dim();
            for c in 0..channels {
                for w in 0..width {
                    let idx = c * width + w;
                    grad_pool2_output[[b, c, w]] = grad_flat[idx];
                }
            }
        }
        
        let grad_conv2_out = self.pool2.backward(&grad_pool2_output, self.conv2_output_shape.unwrap());
        let grad_pool1_out = self.conv2.backward(&grad_conv2_out, learning_rate);
        let grad_conv1_out = self.pool1.backward(&grad_pool1_out, self.conv1_output_shape.unwrap());
        let _grad_input = self.conv1.backward(&grad_conv1_out, learning_rate);
        
        let avg_loss = total_loss / batch_size as f32;
        let accuracy = correct as f32 / batch_size as f32;
        
        (avg_loss, accuracy)
    }
}

// ============================================
// MODEL SERIALIZATION (SAVE/LOAD)
// ============================================

#[derive(Serialize, Deserialize)]
struct SerializableModel {
    conv1_weights: Vec<f32>,
    conv1_bias: Vec<f32>,
    conv2_weights: Vec<f32>,
    conv2_bias: Vec<f32>,
    fc1_weights: Vec<f32>,
    fc1_bias: Vec<f32>,
    fc2_weights: Vec<f32>,
    fc2_bias: Vec<f32>,
    temperature: f32, // ✅ SAVE TEMPERATURE
}

impl CoffeeCNN {
    pub fn save(&self, path: &str) -> Result<()> {
        let model = SerializableModel {
            conv1_weights: self.conv1.weights.iter().cloned().collect(),
            conv1_bias: self.conv1.bias.to_vec(),
            conv2_weights: self.conv2.weights.iter().cloned().collect(),
            conv2_bias: self.conv2.bias.to_vec(),
            fc1_weights: self.fc1.weights.iter().cloned().collect(),
            fc1_bias: self.fc1.bias.to_vec(),
            fc2_weights: self.fc2.weights.iter().cloned().collect(),
            fc2_bias: self.fc2.bias.to_vec(),
            temperature: self.temperature, // ✅ SAVE
        };
        
        let json = serde_json::to_string_pretty(&model)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        
        println!("✅ Model saved to {} (T={:.2})", path, self.temperature);
        Ok(())
    }
    
    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        
        let model: SerializableModel = serde_json::from_str(&json)?;
        
        let mut cnn = Self::new();
        
        // Restore weights
        for (i, val) in model.conv1_weights.iter().enumerate() {
            let out_c = i / (cnn.conv1.in_channels * cnn.conv1.kernel_size);
            let remainder = i % (cnn.conv1.in_channels * cnn.conv1.kernel_size);
            let in_c = remainder / cnn.conv1.kernel_size;
            let k = remainder % cnn.conv1.kernel_size;
            cnn.conv1.weights[[out_c, in_c, k]] = *val;
        }
        cnn.conv1.bias = Array1::from_vec(model.conv1_bias);
        
        for (i, val) in model.conv2_weights.iter().enumerate() {
            let out_c = i / (cnn.conv2.in_channels * cnn.conv2.kernel_size);
            let remainder = i % (cnn.conv2.in_channels * cnn.conv2.kernel_size);
            let in_c = remainder / cnn.conv2.kernel_size;
            let k = remainder % cnn.conv2.kernel_size;
            cnn.conv2.weights[[out_c, in_c, k]] = *val;
        }
        cnn.conv2.bias = Array1::from_vec(model.conv2_bias);
        
        let fc1_shape = cnn.fc1.weights.dim();
        for i in 0..fc1_shape.0 {
            for j in 0..fc1_shape.1 {
                cnn.fc1.weights[[i, j]] = model.fc1_weights[i * fc1_shape.1 + j];
            }
        }
        cnn.fc1.bias = Array1::from_vec(model.fc1_bias);
        
        let fc2_shape = cnn.fc2.weights.dim();
        for i in 0..fc2_shape.0 {
            for j in 0..fc2_shape.1 {
                cnn.fc2.weights[[i, j]] = model.fc2_weights[i * fc2_shape.1 + j];
            }
        }
        cnn.fc2.bias = Array1::from_vec(model.fc2_bias);
        
        // ✅ RESTORE TEMPERATURE
        cnn.temperature = model.temperature;
        
        println!("✅ Model loaded from {} (T={:.2})", path, cnn.temperature);
        Ok(cnn)
    }
}

// ============================================
// CLONE IMPLEMENTATION
// ============================================

impl Clone for Conv1D {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            last_input: None, // Don't clone cached values
            last_output: None,
        }
    }
}

impl Clone for MaxPool1D {
    fn clone(&self) -> Self {
        Self {
            kernel_size: self.kernel_size,
            stride: self.stride,
            last_indices: None, // Don't clone cached values
        }
    }
}

impl Clone for FullyConnected {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
            last_input: None, // Don't clone cached values
            last_output: None,
        }
    }
}

impl Clone for CoffeeCNN {
    fn clone(&self) -> Self {
        Self {
            conv1: self.conv1.clone(),
            pool1: self.pool1.clone(),
            conv2: self.conv2.clone(),
            pool2: self.pool2.clone(),
            fc1: self.fc1.clone(),
            fc2: self.fc2.clone(),
            conv1_output_shape: None, // Don't clone cached shapes
            pool1_output_shape: None,
            conv2_output_shape: None,
            temperature: self.temperature,
        }
    }
}

