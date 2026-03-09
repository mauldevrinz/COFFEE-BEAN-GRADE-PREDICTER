// src/preprocessing.rs - Data Cleaning & Preprocessing
use anyhow::Result;
use log::info;
use ndarray::{Array2, Array3, s};
use statrs::statistics::{Data, OrderStatistics};

#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    pub remove_outliers: bool,
    pub outlier_threshold: f64,
    pub handle_missing: bool,
    pub missing_strategy: MissingStrategy,
    pub apply_smoothing: bool,
    pub smoothing_window: usize,
    pub clip_values: bool,
    pub clip_min: f32,
    pub clip_max: f32,
}

#[derive(Debug, Clone)]
pub enum MissingStrategy {
    Mean,
    Forward,
    Interpolate,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            remove_outliers: true,
            outlier_threshold: 1.5,
            handle_missing: true,
            missing_strategy: MissingStrategy::Interpolate,
            apply_smoothing: false,
            smoothing_window: 5,
            clip_values: true,
            clip_min: -10.0,
            clip_max: 10.0,
        }
    }
}

pub struct DataPreprocessor {
    config: PreprocessingConfig,
}

impl DataPreprocessor {
    pub fn new(config: PreprocessingConfig) -> Self {
        Self { config }
    }
    
    pub fn with_default() -> Self {
        Self {
            config: PreprocessingConfig::default(),
        }
    }
    
    pub fn preprocess_sample(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let mut processed = data.clone();
        
        info!("🔧 Starting preprocessing...");
        
        if self.has_invalid_values(&processed) {
            if self.config.handle_missing {
                info!("  ⚠️  Found invalid values, handling...");
                processed = self.handle_missing_values(&processed)?;
            } else {
                anyhow::bail!("Data contains NaN/Inf values");
            }
        }
        
        if self.config.clip_values {
            info!("  ✂️  Clipping values to [{}, {}]", self.config.clip_min, self.config.clip_max);
            processed = self.clip_values(&processed);
        }
        
        if self.config.remove_outliers {
            info!("  🎯 Removing outliers (threshold: {})", self.config.outlier_threshold);
            processed = self.remove_outliers(&processed)?;
        }
        
        if self.config.apply_smoothing {
            info!("  📊 Applying smoothing (window: {})", self.config.smoothing_window);
            processed = self.apply_smoothing(&processed)?;
        }
        
        info!("✅ Preprocessing complete");
        
        Ok(processed)
    }
    
    pub fn preprocess_batch(&self, data: &Array3<f32>) -> Result<Array3<f32>> {
        let num_samples = data.shape()[0];
        let mut processed = Array3::<f32>::zeros(data.raw_dim());
        
        info!("🔧 Preprocessing {} samples...", num_samples);
        
        for i in 0..num_samples {
            let sample = data.slice(s![i, .., ..]);
            let sample_2d = sample.to_owned();
            let processed_sample = self.preprocess_sample(&sample_2d)?;
            processed.slice_mut(s![i, .., ..]).assign(&processed_sample);
        }
        
        info!("✅ Batch preprocessing complete");
        
        Ok(processed)
    }
    
    fn has_invalid_values(&self, data: &Array2<f32>) -> bool {
        data.iter().any(|&x| x.is_nan() || x.is_infinite())
    }
    
    fn handle_missing_values(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let mut result = data.clone();
        
        match self.config.missing_strategy {
            MissingStrategy::Mean => {
                for col in 0..data.shape()[1] {
                    let col_data: Vec<f32> = data.column(col)
                        .iter()
                        .copied()
                        .filter(|x| !x.is_nan() && !x.is_infinite())
                        .collect();
                    
                    if col_data.is_empty() {
                        continue;
                    }
                    
                    let mean = col_data.iter().sum::<f32>() / col_data.len() as f32;
                    
                    for row in 0..data.shape()[0] {
                        if result[[row, col]].is_nan() || result[[row, col]].is_infinite() {
                            result[[row, col]] = mean;
                        }
                    }
                }
            },
            
            MissingStrategy::Forward => {
                for col in 0..data.shape()[1] {
                    let mut last_valid = 0.0;
                    for row in 0..data.shape()[0] {
                        if result[[row, col]].is_nan() || result[[row, col]].is_infinite() {
                            result[[row, col]] = last_valid;
                        } else {
                            last_valid = result[[row, col]];
                        }
                    }
                }
            },
            
            MissingStrategy::Interpolate => {
                for col in 0..data.shape()[1] {
                    let mut i = 0;
                    while i < data.shape()[0] {
                        if result[[i, col]].is_nan() || result[[i, col]].is_infinite() {
                            let mut j = i + 1;
                            while j < data.shape()[0] && 
                                  (result[[j, col]].is_nan() || result[[j, col]].is_infinite()) {
                                j += 1;
                            }
                            
                            if j < data.shape()[0] && i > 0 {
                                let start_val = result[[i - 1, col]];
                                let end_val = result[[j, col]];
                                let steps = (j - i + 1) as f32;
                                
                                for k in i..j {
                                    let ratio = (k - i + 1) as f32 / steps;
                                    result[[k, col]] = start_val + (end_val - start_val) * ratio;
                                }
                            }
                            
                            i = j;
                        } else {
                            i += 1;
                        }
                    }
                }
            },
        }
        
        Ok(result)
    }
    
    fn clip_values(&self, data: &Array2<f32>) -> Array2<f32> {
        data.mapv(|x| x.clamp(self.config.clip_min, self.config.clip_max))
    }
    
    fn remove_outliers(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let mut result = data.clone();
        
        for col in 0..data.shape()[1] {
            let col_data: Vec<f64> = data.column(col)
                .iter()
                .map(|&x| x as f64)
                .collect();
            
            let mut stats = Data::new(col_data);
            let q1 = stats.quantile(0.25);
            let q3 = stats.quantile(0.75);
            let iqr = q3 - q1;
            
            let lower_bound = q1 - self.config.outlier_threshold * iqr;
            let upper_bound = q3 + self.config.outlier_threshold * iqr;
            
            let median = stats.median() as f32;
            
            for row in 0..data.shape()[0] {
                let val = result[[row, col]] as f64;
                if val < lower_bound || val > upper_bound {
                    result[[row, col]] = median;
                }
            }
        }
        
        Ok(result)
    }
    
    fn apply_smoothing(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let mut result = data.clone();
        let window = self.config.smoothing_window;
        let half_window = window / 2;
        
        for col in 0..data.shape()[1] {
            for row in half_window..(data.shape()[0] - half_window) {
                let start = row - half_window;
                let end = row + half_window + 1;
                
                let window_data = data.slice(s![start..end, col]);
                let mean = window_data.mean().unwrap_or(data[[row, col]]);
                result[[row, col]] = mean;
            }
        }
        
        Ok(result)
    }
    
    pub fn generate_report(&self, original: &Array2<f32>, processed: &Array2<f32>) -> PreprocessingReport {
        let original_stats = self.compute_stats(original);
        let processed_stats = self.compute_stats(processed);
        
        let outliers_removed = original_stats.outliers_count.saturating_sub(processed_stats.outliers_count);
        let missing_filled = original_stats.missing_count.saturating_sub(processed_stats.missing_count);
        
        PreprocessingReport {
            original_stats,
            processed_stats,
            outliers_removed,
            missing_filled,
        }
    }
    
    fn compute_stats(&self, data: &Array2<f32>) -> DataStats {
        let missing_count = data.iter().filter(|x| x.is_nan() || x.is_infinite()).count();
        
        let mut outliers_count = 0;
        for col in 0..data.shape()[1] {
            let col_data: Vec<f64> = data.column(col)
                .iter()
                .copied()
                .filter(|x| !x.is_nan() && !x.is_infinite())
                .map(|x| x as f64)
                .collect();
            
            if col_data.is_empty() {
                continue;
            }
            
            let mut stats = Data::new(col_data.clone());
            let q1 = stats.quantile(0.25);
            let q3 = stats.quantile(0.75);
            let iqr = q3 - q1;
            let lower = q1 - 1.5 * iqr;
            let upper = q3 + 1.5 * iqr;
            
            outliers_count += col_data.iter()
                .filter(|&&x| x < lower || x > upper)
                .count();
        }
        
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(0.0);
        
        DataStats {
            mean,
            std,
            missing_count,
            outliers_count,
        }
    }
}

#[derive(Debug)]
pub struct DataStats {
    pub mean: f32,
    pub std: f32,
    pub missing_count: usize,
    pub outliers_count: usize,
}

#[derive(Debug)]
pub struct PreprocessingReport {
    pub original_stats: DataStats,
    pub processed_stats: DataStats,
    pub outliers_removed: usize,
    pub missing_filled: usize,
}

impl PreprocessingReport {
    pub fn print(&self) {
        use colored::*;
        
        println!("\n{}", "═══════════════════════════════════════════════════".cyan().bold());
        println!("{}", "           Preprocessing Report".cyan().bold());
        println!("{}", "═══════════════════════════════════════════════════".cyan().bold());
        
        println!("\n{}", "📊 Original Data:".bold());
        println!("  Mean:              {:.4}", self.original_stats.mean);
        println!("  Std Dev:           {:.4}", self.original_stats.std);
        println!("  Missing Values:    {}", self.original_stats.missing_count);
        println!("  Outliers:          {}", self.original_stats.outliers_count);
        
        println!("\n{}", "✨ Processed Data:".bold());
        println!("  Mean:              {:.4}", self.processed_stats.mean);
        println!("  Std Dev:           {:.4}", self.processed_stats.std);
        println!("  Missing Values:    {}", self.processed_stats.missing_count);
        println!("  Outliers:          {}", self.processed_stats.outliers_count);
        
        println!("\n{}", "🔧 Changes Applied:".bold().green());
        println!("  Outliers Removed:  {}", self.outliers_removed.to_string().green());
        println!("  Missing Filled:    {}", self.missing_filled.to_string().green());
        
        println!();
    }
}