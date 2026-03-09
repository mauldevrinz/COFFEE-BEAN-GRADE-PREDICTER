// src/data_loader.rs - WITH Z-SCORE NORMALIZATION ADDED (COMPLETE)

use ndarray::{Array1, Array2, Array3};
use walkdir::WalkDir;
use csv::ReaderBuilder;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationStats {
    pub mean: Array1<f32>,
    pub std: Array1<f32>,
}

#[derive(Debug, Clone)]
pub struct CoffeeDataset {
    pub samples: Array3<f32>,
    pub labels: Array1<i64>,
    pub normalization_stats: NormalizationStats,
}

#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub train_ratio: f32,
    pub val_ratio: f32,
    pub test_ratio: f32,
    pub selected_samples: Option<Vec<String>>,
    pub min_timesteps: usize,
    pub max_timesteps: usize,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.8,
            val_ratio: 0.2,
            test_ratio: 0.0,
            selected_samples: None,
            min_timesteps: 300,
            max_timesteps: 300,
        }
    }
}

pub struct CoffeeDataLoader {
    root_path: PathBuf,
    config: DataLoaderConfig,
}

// ============================================
// ✅ Z-SCORE NORMALIZATION FUNCTION (ADDED!)
// ============================================
fn normalize_dataset_zscore(dataset: &mut CoffeeDataset) {
    let (num_samples, timesteps, features) = dataset.samples.dim();
    
    println!("\n🔧 NORMALIZING DATA (Z-SCORE)...");
    
    // Calculate global statistics BEFORE normalization
    let total_elements = (num_samples * timesteps * features) as f32;
    let global_mean_before: f32 = dataset.samples.iter().sum::<f32>() / total_elements;
    
    println!("   📊 BEFORE Normalization:");
    println!("      Global Mean: {:.4}", global_mean_before);
    println!("      Range: [{:.2}, {:.2}]", 
        dataset.samples.iter().cloned().fold(f32::INFINITY, f32::min),
        dataset.samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    // Normalize each feature (sensor) independently across all samples and timesteps
    for f in 0..features {
        let mut values = Vec::new();
        
        // Collect all values for this feature
        for s in 0..num_samples {
            for t in 0..timesteps {
                values.push(dataset.samples[[s, t, f]]);
            }
        }
        
        // Calculate statistics for this feature
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        let std = (variance + 1e-8).sqrt();
        
        // Update normalization stats
        dataset.normalization_stats.mean[f] = mean;
        dataset.normalization_stats.std[f] = std;
        
        // Apply Z-score normalization: z = (x - mean) / std
        for s in 0..num_samples {
            for t in 0..timesteps {
                dataset.samples[[s, t, f]] = (dataset.samples[[s, t, f]] - mean) / std;
            }
        }
        
        if f < 3 {
            println!("      Sensor {} → mean={:.4}, std={:.4}", f, mean, std);
        }
    }
    
    // Calculate statistics AFTER normalization
    let global_mean_after: f32 = dataset.samples.iter().sum::<f32>() / total_elements;
    let global_std_after: f32 = {
        let variance = dataset.samples.iter()
            .map(|x| (x - global_mean_after).powi(2))
            .sum::<f32>() / total_elements;
        variance.sqrt()
    };
    
    println!("\n   ✅ AFTER Normalization:");
    println!("      Global Mean: {:.6} (should be ~0)", global_mean_after);
    println!("      Global Std:  {:.4} (should be ~1)", global_std_after);
    println!("      Range: [{:.4}, {:.4}]", 
        dataset.samples.iter().cloned().fold(f32::INFINITY, f32::min),
        dataset.samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    println!("✅ Normalization complete!\n");
}

impl CoffeeDataLoader {
    pub fn new(root_path: impl AsRef<Path>) -> Self {
        Self {
            root_path: root_path.as_ref().to_path_buf(),
            config: DataLoaderConfig::default(),
        }
    }

    pub fn with_config(root_path: impl AsRef<Path>, config: DataLoaderConfig) -> Self {
        Self {
            root_path: root_path.as_ref().to_path_buf(),
            config,
        }
    }

    pub fn load_all_data(&self) -> Result<(CoffeeDataset, CoffeeDataset, CoffeeDataset)> {
        let high_path = self.root_path.join("high_grade");
        let high_samples_by_folder = self.load_grade_samples_by_folder(&high_path, 0)?;
        
        let low_path = self.root_path.join("low_grade");
        let low_samples_by_folder = self.load_grade_samples_by_folder(&low_path, 1)?;

        println!("\n📊 Loaded from disk:");
        println!("   High grade: {} sample folders", high_samples_by_folder.len());
        for (folder, files) in &high_samples_by_folder {
            println!("      {} → {} files", folder, files.len());
        }
        println!("   Low grade: {} sample folders", low_samples_by_folder.len());
        for (folder, files) in &low_samples_by_folder {
            println!("      {} → {} files", folder, files.len());
        }

        let (train_high, val_high, test_high) = 
            self.stratified_split_by_folder(&high_samples_by_folder);
        let (train_low, val_low, test_low) = 
            self.stratified_split_by_folder(&low_samples_by_folder);

        println!("\n🎯 Stratified Split (per sample folder):");
        println!("   Train: {} high + {} low = {} total", 
            train_high.len(), train_low.len(), train_high.len() + train_low.len());
        println!("   Val:   {} high + {} low = {} total", 
            val_high.len(), val_low.len(), val_high.len() + val_low.len());
        println!("   Test:  {} high + {} low = {} total", 
            test_high.len(), test_low.len(), test_high.len() + test_low.len());

        // ✅ CRITICAL: Create datasets AND normalize!
        let mut train_dataset = self.combine_and_create_dataset(train_high, train_low)?;
        normalize_dataset_zscore(&mut train_dataset);
        
        let mut val_dataset = self.combine_and_create_dataset(val_high, val_low)?;
        // Use training stats for validation
        val_dataset.normalization_stats = train_dataset.normalization_stats.clone();
        self.apply_normalization(&mut val_dataset);
        
        let mut test_dataset = self.combine_and_create_dataset(test_high, test_low)?;
        // Use training stats for test
        test_dataset.normalization_stats = train_dataset.normalization_stats.clone();
        self.apply_normalization(&mut test_dataset);

        println!("\n✅ Dataset Summary:");
        self.print_dataset_balance(&train_dataset, "Training");
        self.print_dataset_balance(&val_dataset, "Validation");
        if test_dataset.labels.len() > 0 {
            self.print_dataset_balance(&test_dataset, "Test");
        }

        Ok((train_dataset, val_dataset, test_dataset))
    }

    // ✅ APPLY NORMALIZATION USING EXISTING STATS (ADDED!)
    fn apply_normalization(&self, dataset: &mut CoffeeDataset) {
        let (num_samples, timesteps, features) = dataset.samples.dim();
        
        for f in 0..features {
            let mean = dataset.normalization_stats.mean[f];
            let std = dataset.normalization_stats.std[f];
            
            for s in 0..num_samples {
                for t in 0..timesteps {
                    dataset.samples[[s, t, f]] = (dataset.samples[[s, t, f]] - mean) / std;
                }
            }
        }
    }

    pub fn load_single_file(
        &self,
        path: &Path,
        norm_stats: &NormalizationStats,
    ) -> Result<Array3<f32>> {
        // Load CSV file
        let data = self.load_csv_file(path)?;
        
        let (rows, cols) = data.dim();
        println!("✅ Raw CSV loaded: {} rows × {} columns", rows, cols);
        
        // Validate shape
        if cols != 8 {
            anyhow::bail!(
                "Invalid number of sensors. Expected 8, got {}.\n\
                CSV must have 9 columns: 1 timestamp + 8 sensors",
                cols
            );
        }
        
        if rows < self.config.min_timesteps {
            anyhow::bail!(
                "Insufficient timesteps. Expected at least {}, got {}",
                self.config.min_timesteps,
                rows
            );
        }
        
        // Trim or pad to 300 timesteps
        let trimmed = self.trim_or_pad_data(data);
        println!("✅ After trim/pad: {} rows × {} columns", trimmed.shape()[0], trimmed.shape()[1]);
        
        // Normalize using training stats
        let (timesteps, features) = trimmed.dim();
        let mut normalized = Array2::zeros((timesteps, features));
        
        for f in 0..features {
            for t in 0..timesteps {
                normalized[[t, f]] = (trimmed[[t, f]] - norm_stats.mean[f]) / norm_stats.std[f];
            }
        }
        
        println!("✅ After normalize: {} rows × {} columns", normalized.shape()[0], normalized.shape()[1]);
        
        // Convert to 3D: (1, timesteps, features) for batch=1
        let mut result = Array3::zeros((1, timesteps, features));
        result.slice_mut(ndarray::s![0, .., ..]).assign(&normalized);
        
        println!("✅ Final shape for CNN: {:?} (batch=1, time={}, sensors={})", 
            result.shape(), timesteps, features);
        
        Ok(result)
    }

    fn load_grade_samples_by_folder(
        &self,
        grade_path: &Path,
        label: i64,
    ) -> Result<HashMap<String, Vec<(Array2<f32>, i64)>>> {
        let mut samples_by_folder: HashMap<String, Vec<(Array2<f32>, i64)>> = HashMap::new();

        for entry in std::fs::read_dir(grade_path)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                let folder_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                let mut folder_samples = Vec::new();
                
                for file_entry in WalkDir::new(&path)
                    .min_depth(1)
                    .max_depth(1)
                    .into_iter()
                    .filter_map(|e| e.ok())
                {
                    if file_entry.file_type().is_file() 
                        && file_entry.path().extension().map_or(false, |e| e == "csv") 
                    {
                        if let Ok(data) = self.load_csv_file(file_entry.path()) {
                            if data.shape()[0] >= self.config.min_timesteps {
                                let trimmed = self.trim_or_pad_data(data);
                                folder_samples.push((trimmed, label));
                            }
                        }
                    }
                }
                
                if !folder_samples.is_empty() {
                    samples_by_folder.insert(folder_name, folder_samples);
                }
            }
        }

        Ok(samples_by_folder)
    }

    fn stratified_split_by_folder(
        &self,
        samples_by_folder: &HashMap<String, Vec<(Array2<f32>, i64)>>,
    ) -> (Vec<(Array2<f32>, i64)>, Vec<(Array2<f32>, i64)>, Vec<(Array2<f32>, i64)>) {
        let mut all_train = Vec::new();
        let mut all_val = Vec::new();
        let mut all_test = Vec::new();
        let mut rng = thread_rng();

        for (folder_name, samples) in samples_by_folder {
            let mut folder_samples = samples.clone();
            folder_samples.shuffle(&mut rng);

            let total = folder_samples.len();
            let train_count = (total as f32 * self.config.train_ratio) as usize;
            let val_count = (total as f32 * self.config.val_ratio) as usize;

            let train: Vec<_> = folder_samples.iter().take(train_count).cloned().collect();
            let val: Vec<_> = folder_samples.iter()
                .skip(train_count)
                .take(val_count)
                .cloned()
                .collect();
            let test: Vec<_> = folder_samples.iter()
                .skip(train_count + val_count)
                .cloned()
                .collect();

            println!("      {} → train: {}, val: {}, test: {}", 
                folder_name, train.len(), val.len(), test.len());

            all_train.extend(train);
            all_val.extend(val);
            all_test.extend(test);
        }

        (all_train, all_val, all_test)
    }

    fn combine_and_create_dataset(
        &self,
        high_samples: Vec<(Array2<f32>, i64)>,
        low_samples: Vec<(Array2<f32>, i64)>,
    ) -> Result<CoffeeDataset> {
        let mut all_samples = high_samples;
        all_samples.extend(low_samples);

        if all_samples.is_empty() {
            return Ok(CoffeeDataset {
                samples: Array3::zeros((0, 300, 8)),
                labels: Array1::zeros(0),
                normalization_stats: NormalizationStats {
                    mean: Array1::zeros(8),
                    std: Array1::ones(8),
                },
            });
        }

        let mut rng = thread_rng();
        all_samples.shuffle(&mut rng);

        let num_samples = all_samples.len();
        let (timesteps, features) = all_samples[0].0.dim();

        let mut samples_array = Array3::zeros((num_samples, timesteps, features));
        let mut labels_array = Array1::zeros(num_samples);

        for (i, (sample, label)) in all_samples.iter().enumerate() {
            samples_array.slice_mut(ndarray::s![i, .., ..]).assign(sample);
            labels_array[i] = *label;
        }

        // Initialize normalization stats (will be filled later)
        let mean = Array1::zeros(features);
        let std = Array1::ones(features);

        Ok(CoffeeDataset {
            samples: samples_array,
            labels: labels_array,
            normalization_stats: NormalizationStats { mean, std },
        })
    }

    fn load_csv_file(&self, path: &Path) -> Result<Array2<f32>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(true)  // Skip header row
            .from_path(path)
            .context(format!("Failed to read CSV: {:?}", path))?;

        let mut data = Vec::new();
        
        for result in reader.records() {
            let record = result?;
            
            // Skip timestamp (column 0), parse sensors (columns 1-8)
            let row: Result<Vec<f32>, _> = record
                .iter()
                .skip(1)  // SKIP TIMESTAMP COLUMN!
                .map(|s| s.parse::<f32>()
                    .map_err(|e| anyhow::anyhow!("Parse error: {}", e)))
                .collect();
            
            data.push(row?);
        }

        if data.is_empty() {
            anyhow::bail!("Empty CSV file: {:?}", path);
        }

        let num_rows = data.len();      // Should be ~300 (timesteps)
        let num_cols = data[0].len();   // Should be 8 (sensors)
        
        // Flatten data: row-major order
        let flat_data: Vec<f32> = data.into_iter().flatten().collect();
        
        // Create Array2 with shape (timesteps, sensors) = (rows, cols)
        Array2::from_shape_vec((num_rows, num_cols), flat_data)
            .context("Failed to create array from CSV data")
    }

    fn trim_or_pad_data(&self, data: Array2<f32>) -> Array2<f32> {
        let target_len = self.config.max_timesteps;
        let (current_len, features) = data.dim();

        if current_len == target_len {
            data
        } else if current_len > target_len {
            // Trim to target length
            data.slice(ndarray::s![0..target_len, ..]).to_owned()
        } else {
            // Pad with zeros
            let mut padded = Array2::zeros((target_len, features));
            padded.slice_mut(ndarray::s![0..current_len, ..]).assign(&data);
            padded
        }
    }

    fn print_dataset_balance(&self, dataset: &CoffeeDataset, name: &str) {
        let high_count = dataset.labels.iter().filter(|&&l| l == 0).count();
        let low_count = dataset.labels.iter().filter(|&&l| l == 1).count();
        let total = dataset.labels.len();

        println!("   {}: {} samples ({} high, {} low)", name, total, high_count, low_count);
    }
}
