// src/ml/models/random_forest.rs - Random Forest Coffee Quality Classifier
// Konsisten dengan CoffeeCNN: train_step(), predict(), save(), load()

use anyhow::Result;
use ndarray::{Array1, Array2, s};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};

// ============================================
// FEATURE EXTRACTION
// ============================================
// Mengubah input timeseries (channels x timesteps) menjadi feature vector statistik.
// Setiap channel: mean, std, min, max, range, median → 6 fitur × n_channels

pub fn extract_features(sample: &Array2<f32>) -> Array1<f32> {
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
// DECISION TREE NODE
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNode {
    Leaf {
        class: i64,
        probability: f32,
    },
    Split {
        feature_idx: usize,
        threshold: f32,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    fn predict(&self, features: &Array1<f32>) -> (i64, f32) {
        match self {
            TreeNode::Leaf { class, probability } => (*class, *probability),
            TreeNode::Split { feature_idx, threshold, left, right } => {
                if features[*feature_idx] <= *threshold {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
        }
    }
}

// ============================================
// DECISION TREE BUILDER
// ============================================

fn gini_impurity(labels: &[i64]) -> f32 {
    if labels.is_empty() {
        return 0.0;
    }
    let mut counts = HashMap::new();
    for &l in labels {
        *counts.entry(l).or_insert(0usize) += 1;
    }
    let n = labels.len() as f32;
    1.0 - counts.values().map(|&c| (c as f32 / n).powi(2)).sum::<f32>()
}

fn majority_class_and_prob(labels: &[i64]) -> (i64, f32) {
    if labels.is_empty() {
        return (0, 0.5);
    }
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &l in labels {
        *counts.entry(l).or_insert(0) += 1;
    }
    let n = labels.len() as f32;
    let (&class, &count) = counts.iter().max_by_key(|(_, c)| *c).unwrap();
    (class, count as f32 / n)
}

fn build_tree(
    features: &Array2<f32>,
    labels: &[i64],
    indices: &[usize],
    max_depth: usize,
    min_samples_split: usize,
    max_features: usize,
    rng: &mut impl Rng,
) -> TreeNode {
    let (class, probability) = majority_class_and_prob(&indices.iter().map(|&i| labels[i]).collect::<Vec<_>>());

    if indices.len() < min_samples_split || max_depth == 0 {
        return TreeNode::Leaf { class, probability };
    }

    // Semua label sama → leaf
    if indices.iter().all(|&i| labels[i] == labels[indices[0]]) {
        return TreeNode::Leaf { class, probability: 1.0 };
    }

    let n_features = features.ncols();
    let feature_subset: Vec<usize> = {
        let mut all: Vec<usize> = (0..n_features).collect();
        all.shuffle(rng);
        all.truncate(max_features.min(n_features));
        all
    };

    let mut best_gini = f32::INFINITY;
    let mut best_feature = 0;
    let mut best_threshold = 0.0f32;

    for &feat in &feature_subset {
        let mut values: Vec<f32> = indices.iter().map(|&i| features[[i, feat]]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();

        for window in values.windows(2) {
            let threshold = (window[0] + window[1]) / 2.0;
            let (left_idx, right_idx): (Vec<_>, Vec<_>) =
                indices.iter().partition(|&&i| features[[i, feat]] <= threshold);

            if left_idx.is_empty() || right_idx.is_empty() {
                continue;
            }

            let left_labels: Vec<i64> = left_idx.iter().map(|&&i| labels[i]).collect();
            let right_labels: Vec<i64> = right_idx.iter().map(|&&i| labels[i]).collect();

            let nl = left_idx.len() as f32;
            let nr = right_idx.len() as f32;
            let n = indices.len() as f32;
            let gini = (nl / n) * gini_impurity(&left_labels)
                + (nr / n) * gini_impurity(&right_labels);

            if gini < best_gini {
                best_gini = gini;
                best_feature = feat;
                best_threshold = threshold;
            }
        }
    }

    if best_gini >= gini_impurity(&indices.iter().map(|&i| labels[i]).collect::<Vec<_>>()) {
        return TreeNode::Leaf { class, probability };
    }

    let (left_idx, right_idx): (Vec<_>, Vec<_>) =
        indices.iter().cloned().partition(|&i| features[[i, best_feature]] <= best_threshold);

    let left = build_tree(features, labels, &left_idx, max_depth - 1, min_samples_split, max_features, rng);
    let right = build_tree(features, labels, &right_idx, max_depth - 1, min_samples_split, max_features, rng);

    TreeNode::Split {
        feature_idx: best_feature,
        threshold: best_threshold,
        left: Box::new(left),
        right: Box::new(right),
    }
}

// ============================================
// RANDOM FOREST CONFIG
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestConfig {
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
    /// Jumlah fitur yang dipertimbangkan per split (0 = sqrt(n_features))
    pub max_features: usize,
    /// Fraction data bootstrap per tree (0.0 = 1.0 / all data)
    pub bootstrap_fraction: f32,
}

impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            max_depth: 10,
            min_samples_split: 4,
            max_features: 0, // auto = sqrt(n_features)
            bootstrap_fraction: 0.8,
        }
    }
}

// ============================================
// TRAINING METRICS RF
// ============================================

#[derive(Debug, Clone)]
pub struct RFTrainingMetrics {
    pub train_accuracy: f32,
    pub oob_accuracy: Option<f32>,
    pub n_trees_built: usize,
    pub feature_importance: Vec<f32>,
}

// ============================================
// COFFEE RANDOM FOREST
// ============================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoffeeRandomForest {
    pub config: RandomForestConfig,
    trees: Vec<TreeNode>,
    pub feature_importance: Vec<f32>,
    n_features: usize,
}

impl CoffeeRandomForest {
    pub fn new(config: RandomForestConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            feature_importance: Vec::new(),
            n_features: 0,
        }
    }

    /// Train dari Array3 (samples x channels x timesteps) dan label.
    /// Sama dengan CoffeeCNN: menerima Array3<f32> agar interface konsisten.
    pub fn fit(
        &mut self,
        inputs: &ndarray::Array3<f32>,
        labels: &Array1<i64>,
    ) -> Result<RFTrainingMetrics> {
        self.fit_with_progress(inputs, labels, |_, _| {})
    }

    /// Sama dengan fit(), dengan callback progres per-tree: `on_tree_done(tree_idx, total_trees)`
    pub fn fit_with_progress<F>(
        &mut self,
        inputs: &ndarray::Array3<f32>,
        labels: &Array1<i64>,
        on_tree_done: F,
    ) -> Result<RFTrainingMetrics>
    where
        F: Fn(usize, usize),
    {
        let n_samples = inputs.dim().0;
        let n_channels = inputs.dim().1;

        // Extract feature matrix: (n_samples, n_features)
        let sample_features: Vec<Array1<f32>> = (0..n_samples)
            .map(|i| extract_features(&inputs.slice(s![i, .., ..]).to_owned()))
            .collect();

        let n_features = sample_features[0].len();
        self.n_features = n_features;

        let feature_matrix: Array2<f32> = {
            let mut mat = Array2::zeros((n_samples, n_features));
            for (i, feat) in sample_features.iter().enumerate() {
                mat.row_mut(i).assign(feat);
            }
            mat
        };

        let labels_vec: Vec<i64> = labels.iter().cloned().collect();

        let max_features = if self.config.max_features == 0 {
            ((n_features as f32).sqrt() as usize).max(1)
        } else {
            self.config.max_features
        };

        let bootstrap_n = ((n_samples as f32 * self.config.bootstrap_fraction) as usize).max(1);
        let mut rng = rand::thread_rng();
        let mut feature_importance = vec![0.0f32; n_features];
        self.trees = Vec::with_capacity(self.config.n_trees);

        let n_trees = self.config.n_trees;
        for tree_idx in 0..n_trees {
            // Bootstrap sampling
            let bootstrap_indices: Vec<usize> = (0..bootstrap_n)
                .map(|_| rng.gen_range(0..n_samples))
                .collect();

            let tree = build_tree(
                &feature_matrix,
                &labels_vec,
                &bootstrap_indices,
                self.config.max_depth,
                self.config.min_samples_split,
                max_features,
                &mut rng,
            );

            self.accumulate_feature_importance(&tree, &mut feature_importance);
            self.trees.push(tree);
            on_tree_done(tree_idx + 1, n_trees);
        }

        // Normalize feature importance
        let importance_sum: f32 = feature_importance.iter().sum();
        if importance_sum > 0.0 {
            feature_importance.iter_mut().for_each(|x| *x /= importance_sum);
        }
        self.feature_importance = feature_importance.clone();

        // Compute training accuracy
        let train_accuracy = self.evaluate_accuracy(&feature_matrix, labels);

        let sensor_names = ["tgs2600", "mq135", "mq3", "mq6", "mq7", "tgs2602", "tgs2611", "tgs2620"];
        let stat_names = ["mean", "std", "min", "max", "range", "median"];
        println!("\n📊 Feature Importance (top 10):");
        let mut importance_indexed: Vec<(usize, f32)> = feature_importance.iter().cloned().enumerate().collect();
        importance_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (idx, score) in importance_indexed.iter().take(10) {
            let sensor = sensor_names.get(idx / 6).unwrap_or(&"?");
            let stat = stat_names.get(idx % 6).unwrap_or(&"?");
            println!("  [{:>5.1}%] {} - {}", score * 100.0, sensor, stat);
        }

        let _ = n_channels; // suppress unused warning

        Ok(RFTrainingMetrics {
            train_accuracy,
            oob_accuracy: None,
            n_trees_built: self.trees.len(),
            feature_importance,
        })
    }

    /// Prediksi dari Array3<f32> — sama dengan CoffeeCNN::predict() signature
    pub fn predict(&self, inputs: &ndarray::Array3<f32>) -> Array2<f32> {
        let n_samples = inputs.dim().0;
        let mut probs = Array2::zeros((n_samples, 2));

        for i in 0..n_samples {
            let features = extract_features(&inputs.slice(s![i, .., ..]).to_owned());
            let (high_votes, low_votes) = self.vote(&features);
            let total = high_votes + low_votes;
            if total > 0.0 {
                probs[[i, 0]] = high_votes / total; // P(High Grade)
                probs[[i, 1]] = low_votes / total;  // P(Low Grade)
            } else {
                probs[[i, 0]] = 0.5;
                probs[[i, 1]] = 0.5;
            }
        }

        probs
    }

    /// Prediksi single sample, return (class, confidence)
    pub fn predict_single(&self, input: &Array2<f32>) -> (i64, f32) {
        let features = extract_features(input);
        let (high_votes, low_votes) = self.vote(&features);
        let total = high_votes + low_votes;
        if total == 0.0 {
            return (0, 0.5);
        }
        if high_votes >= low_votes {
            (0, high_votes / total)
        } else {
            (1, low_votes / total)
        }
    }

    fn vote(&self, features: &Array1<f32>) -> (f32, f32) {
        let mut high_votes = 0.0f32;
        let mut low_votes = 0.0f32;
        for tree in &self.trees {
            let (class, prob) = tree.predict(features);
            if class == 0 { high_votes += prob; } else { low_votes += prob; }
        }
        (high_votes, low_votes)
    }

    fn evaluate_accuracy(&self, features: &Array2<f32>, labels: &Array1<i64>) -> f32 {
        let n = features.nrows();
        let correct = (0..n).filter(|&i| {
            let feat = features.row(i).to_owned();
            let (high, low) = self.vote(&feat);
            let pred = if high >= low { 0i64 } else { 1i64 };
            pred == labels[i]
        }).count();
        if n == 0 { return 0.0; }
        correct as f32 / n as f32
    }

    fn accumulate_feature_importance(&self, node: &TreeNode, importance: &mut Vec<f32>) {
        match node {
            TreeNode::Leaf { .. } => {}
            TreeNode::Split { feature_idx, left, right, .. } => {
                if *feature_idx < importance.len() {
                    importance[*feature_idx] += 1.0;
                }
                self.accumulate_feature_importance(left, importance);
                self.accumulate_feature_importance(right, importance);
            }
        }
    }

    /// Save model ke JSON — sama dengan CoffeeCNN::save()
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        println!("✅ Random Forest model disimpan ke: {}", path);
        Ok(())
    }

    /// Load model dari JSON — sama dengan CoffeeCNN::load()
    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        let model: Self = serde_json::from_str(&content)?;
        println!("✅ Random Forest model dimuat dari: {}", path);
        Ok(model)
    }

    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Evaluasi akurasi menggunakan hanya N trees pertama (untuk kurva akurasi vs N trees).
    pub fn evaluate_at_n_trees(&self, inputs: &ndarray::Array3<f32>, labels: &Array1<i64>, n: usize) -> f32 {
        let n_use = n.min(self.trees.len());
        if n_use == 0 || inputs.dim().0 == 0 {
            return 0.0;
        }
        let n_samples = inputs.dim().0;
        let correct = (0..n_samples).filter(|&i| {
            let features = extract_features(&inputs.slice(s![i, .., ..]).to_owned());
            let mut high: f32 = 0.0;
            let mut low: f32 = 0.0;
            for tree in self.trees.iter().take(n_use) {
                let (class, prob) = tree.predict(&features);
                if class == 0 { high += prob; } else { low += prob; }
            }
            let pred = if high >= low { 0i64 } else { 1i64 };
            pred == labels[i]
        }).count();
        correct as f32 / n_samples as f32
    }
}
