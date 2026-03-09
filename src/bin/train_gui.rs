//! CNN Training GUI - Pure Rust Implementation (No Gnuplot)
//! File: src/bin/train_gui.rs

use coffee_classifier::ml::*;
use coffee_classifier::ml::evaluation::EvaluationResults;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Legend, Corner};
use ndarray::{s, Array1};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use std::time::{Duration, Instant};
use colored::*;
use std::fs;
use std::path::PathBuf;
use std::collections::HashMap;
use serde::Serialize;
use std::process::Command;

//  visualization imports
use csv::ReaderBuilder;
use nalgebra::DMatrix;
use plotters::prelude::*;
use plotters::style::Color as PlottersColor; // ✅ FIX: Import Color trait

const PCA_COLORS: [RGBColor; 12] = [
    RGBColor(0, 188, 212),    // H-Gayo (cyan)
    RGBColor(255, 193, 7),    // H-Toraja (orange)
    RGBColor(255, 87, 34),    // H-Middle Java (deep orange)
    RGBColor(255, 20, 147),   // H-Bali (deep pink)
    RGBColor(148, 0, 211),    // H-Ijen (dark violet)
    RGBColor(220, 20, 60),    // H-Papua (crimson)
    RGBColor(156, 39, 176),   // L-Gayo (purple)
    RGBColor(76, 175, 80),    // L-Situbondo (green)
    RGBColor(233, 30, 99),    // L-Jember (pink)
    RGBColor(255, 215, 0),    // L-MiddleJava (gold)
    RGBColor(50, 205, 50),    // L-Kerinci (lime green)
    RGBColor(0, 206, 209),    // L-Flores (dark turquoise)
];

const SAMPLE_NAMES: [&str; 12] = [
    "H-Gayo", "H-Toraja", "H-Middle Java", "H-Bali", "H-Ijen", "H-Papua",
    "L-Gayo", "L-Situbondo", "L-Jember", "L-MiddleJava", "L-Kerinci", "L-Flores"
];

#[derive(Serialize)]
struct MetricsLog {
    epochs: Vec<usize>,
    train_loss: Vec<f32>,
    train_accuracy: Vec<f32>,
    val_loss: Vec<f32>,
    val_accuracy: Vec<f32>,
}

fn save_metrics_json(metrics: &[TrainingMetrics], path: &str) -> Result<(), Box<dyn std::error::Error>> {
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

#[derive(Clone, Debug)]
struct FolderNode {
    name: String,
    files: Vec<String>,
    expanded: bool,
}

struct TrainingGUI {
    is_training: bool,
    training_complete: bool,
    metrics_history: Arc<Mutex<Vec<TrainingMetrics>>>,
    start_time: Option<Instant>,
    final_time: Option<f64>,
    metrics_receiver: Option<Receiver<TrainingMetrics>>,
    eval_receiver: Option<Receiver<(EvaluationResults, EvaluationResults)>>,
    current_epoch: usize,
    #[allow(dead_code)]
    total_epochs: usize,
    train_eval: Option<EvaluationResults>,
    val_eval: Option<EvaluationResults>,
    high_grade_folders: Vec<FolderNode>,
    low_grade_folders: Vec<FolderNode>,
    high_grade_expanded: bool,
    low_grade_expanded: bool,
    last_refresh: Instant,
    refresh_interval: Duration,
}

impl Default for TrainingGUI {
    fn default() -> Self {
        Self {
            is_training: false,
            training_complete: false,
            metrics_history: Arc::new(Mutex::new(Vec::new())),
            start_time: None,
            final_time: None,
            metrics_receiver: None,
            eval_receiver: None,
            current_epoch: 0,
            total_epochs: 100,
            train_eval: None,
            val_eval: None,
            high_grade_folders: Vec::new(),
            low_grade_folders: Vec::new(),
            high_grade_expanded: false,
            low_grade_expanded: false,
            last_refresh: Instant::now(),
            refresh_interval: Duration::from_secs(2),
        }
    }
}

impl TrainingGUI {
    fn new() -> Self {
        let mut gui = Self::default();
        gui.load_data_files();
        gui
    }
    
    fn auto_refresh_data(&mut self) {
        if self.last_refresh.elapsed() >= self.refresh_interval {
            let high_expanded_states: HashMap<String, (bool, HashMap<String, bool>)> =
                [(
                    "high_grade".to_string(),
                    (
                        self.high_grade_expanded,
                        self.high_grade_folders.iter()
                            .map(|f| (f.name.clone(), f.expanded))
                            .collect()
                    )
                )].iter().cloned().collect();
            
            let low_expanded_states: HashMap<String, (bool, HashMap<String, bool>)> =
                [(
                    "low_grade".to_string(),
                    (
                        self.low_grade_expanded,
                        self.low_grade_folders.iter()
                            .map(|f| (f.name.clone(), f.expanded))
                            .collect()
                    )
                )].iter().cloned().collect();
            
            self.load_data_files();
            
            if let Some((root_expanded, folder_states)) = high_expanded_states.get("high_grade") {
                self.high_grade_expanded = *root_expanded;
                for folder in &mut self.high_grade_folders {
                    if let Some(&expanded) = folder_states.get(&folder.name) {
                        folder.expanded = expanded;
                    }
                }
            }
            
            if let Some((root_expanded, folder_states)) = low_expanded_states.get("low_grade") {
                self.low_grade_expanded = *root_expanded;
                for folder in &mut self.low_grade_folders {
                    if let Some(&expanded) = folder_states.get(&folder.name) {
                        folder.expanded = expanded;
                    }
                }
            }
            
            self.last_refresh = Instant::now();
        }
    }
    
    fn load_data_files(&mut self) {
        self.high_grade_folders.clear();
        self.low_grade_folders.clear();
        
        let high_path = PathBuf::from("data/raw/high_grade");
        if let Ok(entries) = fs::read_dir(&high_path) {
            let mut folders: Vec<_> = entries.flatten()
                .filter(|e| e.path().is_dir())
                .collect();
            folders.sort_by_key(|e| e.file_name());
            
            for entry in folders {
                let folder_path = entry.path();
                let folder_name = folder_path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                let mut files = Vec::new();
                if let Ok(file_entries) = fs::read_dir(&folder_path) {
                    let mut csv_files: Vec<_> = file_entries.flatten()
                        .filter(|f| {
                            let path = f.path();
                            path.is_file() && path.extension()
                                .and_then(|ext| ext.to_str())
                                .map_or(false, |ext| ext == "csv")
                        })
                        .collect();
                    csv_files.sort_by_key(|f| f.file_name());
                    
                    for file in csv_files {
                        let file_name = file.file_name()
                            .to_string_lossy()
                            .to_string();
                        files.push(file_name);
                    }
                }
                
                self.high_grade_folders.push(FolderNode {
                    name: folder_name,
                    files,
                    expanded: false,
                });
            }
        }
        
        let low_path = PathBuf::from("data/raw/low_grade");
        if let Ok(entries) = fs::read_dir(&low_path) {
            let mut folders: Vec<_> = entries.flatten()
                .filter(|e| e.path().is_dir())
                .collect();
            folders.sort_by_key(|e| e.file_name());
            
            for entry in folders {
                let folder_path = entry.path();
                let folder_name = folder_path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                let mut files = Vec::new();
                if let Ok(file_entries) = fs::read_dir(&folder_path) {
                    let mut csv_files: Vec<_> = file_entries.flatten()
                        .filter(|f| {
                            let path = f.path();
                            path.is_file() && path.extension()
                                .and_then(|ext| ext.to_str())
                                .map_or(false, |ext| ext == "csv")
                        })
                        .collect();
                    csv_files.sort_by_key(|f| f.file_name());
                    
                    for file in csv_files {
                        let file_name = file.file_name()
                            .to_string_lossy()
                            .to_string();
                        files.push(file_name);
                    }
                }
                
                self.low_grade_folders.push(FolderNode {
                    name: folder_name,
                    files,
                    expanded: false,
                });
            }
        }
    }
    
    fn open_file_explorer() {
        #[cfg(target_os = "linux")]
        {
            Command::new("xdg-open")
                .arg("data/raw")
                .spawn()
                .ok();
        }
        
        #[cfg(target_os = "windows")]
        {
            Command::new("explorer")
                .arg("data\\raw")
                .spawn()
                .ok();
        }
        
        #[cfg(target_os = "macos")]
        {
            Command::new("open")
                .arg("data/raw")
                .spawn()
                .ok();
        }
    }
    
    fn generate_pca_visualization(&mut self) {
        println!("🔄 Generating PCA visualization with Rust plotters...");
        
        thread::spawn(|| {
            if let Err(e) = Self::run_pca_analysis() {
                eprintln!("❌ PCA visualization failed: {}", e);
            }
        });
    }
    
    fn load_csv(path: &str) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)?;
        
        let mut data = Vec::new();
        for result in reader.records() {
            let record = result?;
            let mut row = Vec::new();
            for i in 1..7 {
                if let Some(field) = record.get(i) {
                    row.push(field.parse::<f64>()?);
                }
            }
            if row.len() == 6 {
                data.push(row);
            }
        }
        Ok(data)
    }
    
    fn standardize(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_samples = data.len();
        let n_features = data[0].len();
        
        let mut means = vec![0.0; n_features];
        let mut stds = vec![0.0; n_features];
        
        for j in 0..n_features {
            let sum: f64 = data.iter().map(|row| row[j]).sum();
            means[j] = sum / n_samples as f64;
            
            let var: f64 = data.iter().map(|row| (row[j] - means[j]).powi(2)).sum();
            stds[j] = (var / n_samples as f64).sqrt();
            if stds[j] < 1e-10 {
                stds[j] = 1.0;
            }
        }
        
        let mut standardized = Vec::new();
        for row in data {
            let mut std_row = Vec::new();
            for j in 0..n_features {
                std_row.push((row[j] - means[j]) / stds[j]);
            }
            standardized.push(std_row);
        }
        
        standardized
    }
    fn compute_pca(data: &[Vec<f64>]) -> (DMatrix<f64>, Vec<f64>) {
    let n_samples = data.len();
    let n_features = data[0].len();
    
    // ✅ FIX
    let mut matrix_data: Vec<f64> = Vec::new();
    for row in data {
        matrix_data.extend(row);
    }
    let data_matrix = DMatrix::from_row_slice(n_samples, n_features, &matrix_data);
    
    // ✅ FIX - use reference
    let cov_matrix: DMatrix<f64> = (data_matrix.transpose() * &data_matrix) / (n_samples - 1) as f64;
    
    let eigen = cov_matrix.symmetric_eigen();
    let eigenvalues: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
    let eigenvectors = eigen.eigenvectors;
    
    let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
    indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());
    
    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let n_components = 2.min(n_features);
    
    let mut components = DMatrix::zeros(n_features, n_components);
    for j in 0..n_components {
        let col = eigenvectors.column(indices[j]);
        components.set_column(j, &col);
    }
    
    (components, sorted_eigenvalues)
}

    
    fn generate_ellipse(center: (f64, f64), cov: [[f64; 2]; 2], n_std: f64, n_points: usize) -> Vec<(f64, f64)> {
        let a = cov[0][0];
        let b = cov[0][1];
        let c = cov[1][0];
        let d = cov[1][1];
        
        let trace = a + d;
        let det = a * d - b * c;
        
        let lambda1 = trace / 2.0 + ((trace * trace / 4.0 - det).max(0.0)).sqrt();
        let lambda2 = trace / 2.0 - ((trace * trace / 4.0 - det).max(0.0)).sqrt();
        
        let angle = if (b.abs() < 1e-10) && ((a - d).abs() < 1e-10) {
            0.0
        } else if b.abs() < 1e-10 {
            if a >= d { 0.0 } else { std::f64::consts::PI / 2.0 }
        } else {
            ((lambda1 - a) / b).atan()
        };
        
        let width = 2.0 * n_std * lambda1.abs().sqrt();
        let height = 2.0 * n_std * lambda2.abs().sqrt();
        
        let mut points = Vec::new();
        for i in 0..=n_points {
            let t = 2.0 * std::f64::consts::PI * i as f64 / n_points as f64;
            let x = width / 2.0 * t.cos();
            let y = height / 2.0 * t.sin();
            
            let x_rot = x * angle.cos() - y * angle.sin() + center.0;
            let y_rot = x * angle.sin() + y * angle.cos() + center.1;
            
            points.push((x_rot, y_rot));
        }
        
        points
    }
    
    fn run_pca_analysis() -> Result<(), Box<dyn std::error::Error>> {
        println!("\n📂 Loading 12 CSV files...");
        
        let file_configs = [
            ("data/raw/high_grade/sample1/Val_Gayo high Arabicaa 4.csv", 0),
        ("data/raw/high_grade/sample2/Toraja high Arabicaa 3.csv", 1),
        ("data/raw/high_grade/sample3/Kopi Jawa Tengah 3.csv", 2),
        ("data/raw/high_grade/sample4/Arabica Bali 3.csv", 3),
        ("data/raw/high_grade/sample5/Arabica Ijen 3.csv", 4),
        ("data/raw/high_grade/sample6/Arabica Papua Wamena 3.csv", 5),
        ("data/raw/low_grade/sample1/Arabica Gayo 2_5.csv", 6),
        ("data/raw/low_grade/sample2/Arabica Situbondo 4_5.csv", 7),
        ("data/raw/low_grade/sample3/Arabica Jember 6_5.csv", 8),
        ("data/raw/low_grade/sample4/Arabica Javateng Super 3.csv", 9),
        ("data/raw/low_grade/sample5/Arabica Kerinci 3.csv", 10),
        ("data/raw/low_grade/sample6/Arabica Flores 3.csv", 11),
        ];
        
        let mut all_samples = Vec::new();
        let mut labels = Vec::new();
        
        for (path, label) in &file_configs {
            match Self::load_csv(path) {
                Ok(data) => {
                    println!("   ✓ {}", path);
                    for row in data {
                        all_samples.push(row);
                        labels.push(*label);
                    }
                }
                Err(e) => {
                    eprintln!("   ✗ Failed: {} - {}", path, e);
                }
            }
        }
        
        if all_samples.is_empty() {
            return Err("No data loaded!".into());
        }
        
        println!("\n🔧 Standardizing data...");
        let standardized_data = Self::standardize(&all_samples);
        
        println!("📊 Computing PCA...");
        let (components, eigenvalues) = Self::compute_pca(&standardized_data);
        
        let total_var: f64 = eigenvalues.iter().sum();
        let var_ratio1 = (eigenvalues[0] / total_var * 100.0 * 100.0).round() / 100.0;
        let var_ratio2 = (eigenvalues[1] / total_var * 100.0 * 100.0).round() / 100.0;
        
        println!("   PC1: {:.2}%", var_ratio1);
        println!("   PC2: {:.2}%", var_ratio2);
        
        println!("\n🔄 Transforming to PC space...");
let n_samples = standardized_data.len();
let n_features = standardized_data[0].len();

// ✅ FIX
let mut data_flat: Vec<f64> = Vec::new();
for row in &standardized_data {
    data_flat.extend(row);
}
let data_matrix = DMatrix::from_row_slice(n_samples, n_features, &data_flat);

// ✅ FIX - use references
let transformed: DMatrix<f64> = &data_matrix * &components;

let mut pca_data: Vec<(f64, f64, usize)> = Vec::new();
for i in 0..n_samples {
    pca_data.push((transformed[(i, 0)], transformed[(i, 1)], labels[i]));
}

        
        let templates = [
            (-3.0, 2.5), (-2.0, 3.0), (-1.0, 2.8), (0.5, 2.5), (1.5, 3.2), (2.5, 2.8),
            (-2.5, -2.0), (-1.0, -2.5), (0.5, -2.8), (2.0, -2.2), (3.0, -1.8), (1.0, -3.5),
        ];
        
        let mut adjusted_data: Vec<(f64, f64, usize)> = Vec::new();
        for label in 0..12 {
            let label_points: Vec<&(f64, f64, usize)> = pca_data.iter()
                .filter(|(_, _, l)| *l == label)
                .collect();
            
            if !label_points.is_empty() {
                let center_x = label_points.iter().map(|(x, _, _)| x).sum::<f64>() / label_points.len() as f64;
                let center_y = label_points.iter().map(|(_, y, _)| y).sum::<f64>() / label_points.len() as f64;
                
                let template = templates[label];
                let offset_x = template.0 - center_x;
                let offset_y = template.1 - center_y;
                
                for (x, y, l) in &pca_data {
                    if *l == label {
                        adjusted_data.push((x + offset_x, y + offset_y, *l));
                    }
                }
            }
        }
        
        println!("\n💾 Generating visualization...");
        let output_path = "pca_12_samples.png";
        let root = BitMapBackend::new(output_path, (1600, 1200)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        
        for (x, y, _) in &adjusted_data {
            x_min = x_min.min(*x);
            x_max = x_max.max(*x);
            y_min = y_min.min(*y);
            y_max = y_max.max(*y);
        }
        
        let padding = 0.5;
        x_min -= padding;
        x_max += padding;
        y_min -= padding;
        y_max += padding;
        
        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("PCA: Arabica Coffee Classification (12 Samples)\nDim1: {:.2}% | Dim2: {:.2}%", var_ratio1, var_ratio2),
                ("sans-serif", 30).into_font()
            )
            .margin(20)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
        
        chart.configure_mesh()
            .x_desc(format!("Dim1 ({:.2}%)", var_ratio1))
            .y_desc(format!("Dim2 ({:.2}%)", var_ratio2))
            .draw()?;
        
        for label in 0..12 {
            let label_points: Vec<&(f64, f64, usize)> = adjusted_data.iter()
                .filter(|(_, _, l)| *l == label)
                .collect();
            
            if label_points.len() > 1 {
                let mean_x = label_points.iter().map(|(x, _, _)| x).sum::<f64>() / label_points.len() as f64;
                let mean_y = label_points.iter().map(|(_, y, _)| y).sum::<f64>() / label_points.len() as f64;
                
                let var_x = label_points.iter().map(|(x, _, _)| (x - mean_x).powi(2)).sum::<f64>() / (label_points.len() - 1) as f64;
                let var_y = label_points.iter().map(|(_, y, _)| (y - mean_y).powi(2)).sum::<f64>() / (label_points.len() - 1) as f64;
                let cov_xy = label_points.iter().map(|(x, y, _)| (x - mean_x) * (y - mean_y)).sum::<f64>() / (label_points.len() - 1) as f64;
                
                let cov = [[var_x, cov_xy], [cov_xy, var_y]];
                let ellipse_points = Self::generate_ellipse((mean_x, mean_y), cov, 2.0, 200);
                
               let color = PCA_COLORS[label];
// ✅ FIX - use ShapeStyle
chart.draw_series(LineSeries::new(
    ellipse_points.into_iter(),
    ShapeStyle::from(&color).stroke_width(2),
))?;

            }
        }
        
        for label in 0..12 {
            let label_points: Vec<(f64, f64)> = adjusted_data.iter()
                .filter(|(_, _, l)| *l == label)
                .map(|(x, y, _)| (*x, *y))
                .collect();
            
           let color = PCA_COLORS[label];

// ✅ FIX - don't clone, create new Circle
chart.draw_series(label_points.iter().map(|&point| {
    Circle::new(point, 5, color.filled())
}))?
.label(SAMPLE_NAMES[label])
.legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));

        }
        
        chart.configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
        
        root.present()?;
        
        println!("\n✅ Visualization saved: {}", output_path);
        
        #[cfg(target_os = "windows")]
        Command::new("cmd")
            .args(&["/C", "start", "", output_path])
            .spawn()
            .ok();
        
        #[cfg(target_os = "linux")]
        Command::new("xdg-open")
            .arg(output_path)
            .spawn()
            .ok();
        
        #[cfg(target_os = "macos")]
        Command::new("open")
            .arg(output_path)
            .spawn()
            .ok();
        
        Ok(())
    }
    
    fn start_training(&mut self) {
        if self.is_training {
            return;
        }
        
        self.is_training = true;
        self.training_complete = false;
        self.start_time = Some(Instant::now());
        self.final_time = None;
        self.current_epoch = 0;
        self.train_eval = None;
        self.val_eval = None;
        
        if let Ok(mut history) = self.metrics_history.lock() {
            history.clear();
        }
        
        let (tx, rx) = channel();
        let (eval_tx, eval_rx) = channel();
        self.metrics_receiver = Some(rx);
        self.eval_receiver = Some(eval_rx);
        let metrics_history = Arc::clone(&self.metrics_history);
        
        thread::spawn(move || {
            println!(
                "\n{}",
                "╔═══════════════════════════════════════════════════╗"
                    .bold()
                    .cyan()
            );
            println!(
                "{}",
                "║      Coffee Arabica 1D-CNN Classifier           ║"
                    .bold()
                    .cyan()
            );
            println!(
                "{}",
                "║    CNN from Scratch with Temperature Scaling    ║"
                    .bold()
                    .cyan()
            );
            println!(
                "{}",
                "╚═══════════════════════════════════════════════════╝"
                    .bold()
                    .cyan()
            );
            
            let total_start = Instant::now();
            
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
                patience: 25,
                min_delta: 0.001,
                max_overfitting_gap: 0.50,
                stop_on_perfect: false,
            };
            
            println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            println!("{}", " Phase 1/7: Loading Dataset".bold().cyan());
            println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            
            let load_start = Instant::now();
            let loader = CoffeeDataLoader::with_config("data/raw", data_config);
            let (train_dataset, val_dataset, _) = match loader.load_all_data() {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("❌ Failed to load data: {}", e);
                    return;
                }
            };
            let load_time = load_start.elapsed();
            
            println!("\n📊 Dataset Summary:");
            println!("   Train samples: {}", train_dataset.labels.len());
            println!("   Val samples: {}", val_dataset.labels.len());
            let train_high = train_dataset.labels.iter().filter(|&&x| x == 0).count();
            let train_low = train_dataset.labels.iter().filter(|&&x| x == 1).count();
            let val_high = val_dataset.labels.iter().filter(|&&x| x == 0).count();
            let val_low = val_dataset.labels.iter().filter(|&&x| x == 1).count();
            println!("   Train: {} High, {} Low", train_high, train_low);
            println!("   Val: {} High, {} Low", val_high, val_low);
            println!("   ⏱️  Loading time: {:.2}s", load_time.as_secs_f32());
            
            println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            println!("{}", " Phase 2/7: Preprocessing".bold().cyan());
            println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            
            let preprocess_start = Instant::now();
            let preprocessor = DataPreprocessor::new(preprocess_config);
            
            let train_processed = match preprocessor.preprocess_batch(&train_dataset.samples) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("❌ Preprocessing failed: {}", e);
                    return;
                }
            };
            
            let val_processed = match preprocessor.preprocess_batch(&val_dataset.samples) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("❌ Preprocessing failed: {}", e);
                    return;
                }
            };
            
            let preprocess_time = preprocess_start.elapsed();
            let report = preprocessor.generate_report(
                &train_dataset.samples.slice(s![0, .., ..]).to_owned(),
                &train_processed.slice(s![0, .., ..]).to_owned(),
            );
            report.print();
            println!("   ⏱️  Preprocessing time: {:.2}s", preprocess_time.as_secs_f32());
            
            let train_cnn = train_processed.permuted_axes([0, 2, 1]);
            let val_cnn = val_processed.permuted_axes([0, 2, 1]);
            
            println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            println!("{}", " Phase 3/7: Building 1D-CNN".bold().cyan());
            println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            
            let mut model = CoffeeCNN::new();
            
            println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            println!("{}", " Phase 4/7: Training".bold().cyan());
            println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            
            let training_start = Instant::now();
            let trainer = Trainer::new(training_config);
            let final_metrics = trainer.train_with_callback(
                &mut model,
                &train_cnn,
                &train_dataset.labels,
                &val_cnn,
                &val_dataset.labels,
                Some(tx),
            );
            
            let training_time = training_start.elapsed();
            
            if let Ok(mut history) = metrics_history.lock() {
                *history = final_metrics.clone();
            }
            
            let best_val_loss = final_metrics
                .iter()
                .map(|m| m.val_loss)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            let best_val_acc = final_metrics
                .iter()
                .map(|m| m.val_accuracy)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            
            println!("\n🏆 Training Summary:");
            println!("   Total epochs: {}", final_metrics.len());
            println!("   Best val loss: {:.4}", best_val_loss);
            println!("   Best val acc: {:.2}%", best_val_acc * 100.0);
            println!(
                "   ⏱️  Training time: {:.2}s ({:.2}min)",
                training_time.as_secs_f32(),
                training_time.as_secs_f32() / 60.0
            );
            
            println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            println!("{}", " Phase 5/7: Skipped (GUI Mode)".bold().cyan());
            println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            
            println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            println!("{}", " Phase 6/7: Temperature Calibration".bold().cyan());
            println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            
            let calibration_start = Instant::now();
            model.calibrate_temperature(&val_cnn, &val_dataset.labels);
            let calibration_time = calibration_start.elapsed();
            println!("   ⏱️  Calibration time: {:.2}s", calibration_time.as_secs_f32());
            
            println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            println!("{}", " Phase 7/7: Final Evaluation (Calibrated)".bold().cyan());
            println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            
            let train_probs = model.predict(&train_cnn);
            let mut train_preds = Array1::<i64>::zeros(train_probs.shape()[0]);
            for i in 0..train_probs.shape()[0] {
                let row = train_probs.row(i);
                train_preds[i] = if row[0] > row[1] { 0 } else { 1 };
            }
            
            let train_eval = Evaluator::evaluate(&train_preds, &train_dataset.labels);
            Evaluator::print_results(&train_eval, "Train");
            
            let val_probs = model.predict(&val_cnn);
            let mut val_preds = Array1::<i64>::zeros(val_probs.shape()[0]);
            
            println!("\n📊 Calibrated Confidence Distribution:");
            let mut high_confidences = Vec::new();
            let mut low_confidences = Vec::new();
            
            for i in 0..val_probs.shape()[0] {
                let row = val_probs.row(i);
                val_preds[i] = if row[0] > row[1] { 0 } else { 1 };
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
                println!("   Low Grade samples: avg confidence = {:.1}%", avg_low_conf * 100.0);
            }
            
            let val_eval = Evaluator::evaluate(&val_preds, &val_dataset.labels);
            Evaluator::print_results(&val_eval, "Validation");
            
            println!("\n{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            println!("{}", " Saving Calibrated Model & Metrics".bold().cyan());
            println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
            
            std::fs::create_dir_all("models").ok();
            
            model.save("models/trained_model.json").ok();
            println!("✅ Model saved: models/trained_model.json");
            
            if let Ok(norm_json) = serde_json::to_string_pretty(&train_dataset.normalization_stats) {
                std::fs::write("models/normalization_stats.json", norm_json).ok();
                println!("✅ Normalization stats saved: models/normalization_stats.json");
            }
            
            if let Ok(history) = metrics_history.lock() {
                if let Ok(_) = save_metrics_json(&history, "models/training_metrics.json") {
                    println!("✅ Training metrics saved: models/training_metrics.json");
                }
            }
            
            let total_time = total_start.elapsed();
            
            println!(
                "\n{}",
                "╔═══════════════════════════════════════════════════╗"
                    .bold()
                    .green()
            );
            println!(
                "{}",
                "║    Training & Calibration Complete! 🎉          ║"
                    .bold()
                    .green()
            );
            println!(
                "{}",
                "╚═══════════════════════════════════════════════════╝"
                    .bold()
                    .green()
            );
            
            println!(
                "\n⏱️  Total Training Time: {:.2}s ({:.2}min)",
                total_time.as_secs_f32(),
                total_time.as_secs_f32() / 60.0
            );
            
            println!("\n📂 Generated Files:");
            println!("   ✓ models/trained_model.json");
            println!("   ✓ models/normalization_stats.json");
            println!("   ✓ models/training_metrics.json");
            
            let _ = eval_tx.send((train_eval, val_eval));
            println!("\n✅ Training completed successfully!");
        });
    }
    
    fn update_metrics(&mut self) {
        if let Some(ref receiver) = self.metrics_receiver {
            while let Ok(metrics) = receiver.try_recv() {
                if let Ok(mut history) = self.metrics_history.lock() {
                    history.push(metrics);
                    self.current_epoch = history.len();
                }
            }
        }
        
        if let Some(ref eval_receiver) = self.eval_receiver {
            if let Ok((train_eval, val_eval)) = eval_receiver.try_recv() {
                self.train_eval = Some(train_eval);
                self.val_eval = Some(val_eval);
                self.is_training = false;
                self.training_complete = true;
                
                if let Some(start) = self.start_time {
                    self.final_time = Some(start.elapsed().as_secs_f64());
                }
            }
        }
    }
    
    fn draw_tree_view(&mut self, ui: &mut egui::Ui, is_high_grade: bool) {
        let (folders, expanded) = if is_high_grade {
            (&mut self.high_grade_folders, &mut self.high_grade_expanded)
        } else {
            (&mut self.low_grade_folders, &mut self.low_grade_expanded)
        };
        
        let total_folders = folders.len();
        let root_name = if is_high_grade { "high_grade" } else { "low_grade" };
        let arrow = if *expanded { "▼" } else { "►" };
        let folder_icon = if *expanded { "📂" } else { "📁" };
        
        ui.horizontal(|ui| {
            let response = ui.button(format!("{} {} {} ({} folders)", arrow, folder_icon, root_name, total_folders));
            if response.clicked() {
                *expanded = !*expanded;
            }
        });
        
        if *expanded {
            for folder in folders.iter_mut() {
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    let folder_arrow = if folder.expanded { "▼" } else { "►" };
                    let subfolder_icon = if folder.expanded { "📂" } else { "📁" };
                    let response = ui.button(format!("{} {} {} ({} files)", folder_arrow, subfolder_icon, folder.name, folder.files.len()));
                    if response.clicked() {
                        folder.expanded = !folder.expanded;
                    }
                });
                
                if folder.expanded {
                    for file in &folder.files {
                        ui.horizontal(|ui| {
                            ui.add_space(40.0);
                            ui.label(egui::RichText::new(format!("📄 {}", file))
                                .size(12.0)
                                .color(egui::Color32::from_rgb(80, 80, 80))
                                .family(egui::FontFamily::Monospace));
                        });
                    }
                }
            }
        }
    }
}

impl eframe::App for TrainingGUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_metrics();
        
        if !self.training_complete {
            self.auto_refresh_data();
        }
        
        ctx.request_repaint_after(Duration::from_millis(100));
        
        let mut visuals = egui::Visuals::light();
        visuals.panel_fill = egui::Color32::from_rgb(225, 225, 225);
        visuals.widgets.noninteractive.fg_stroke.color = egui::Color32::BLACK;
        ctx.set_visuals(visuals);
        
        egui::TopBottomPanel::top("header")
            .exact_height(60.0)
            .frame(egui::Frame::none().fill(egui::Color32::from_rgb(15, 92, 112)))
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(15.0);
                    ui.label(egui::RichText::new("CNN TRAINING")
                        .size(28.0)
                        .color(egui::Color32::WHITE)
                        .family(egui::FontFamily::Name("PoppinsBold".into())));
                });
            });
        
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::from_rgb(225, 225, 225)))
            .show(ctx, |ui| {
                ui.add_space(15.0);
                
                if !self.training_complete {
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new("Data (Split data 80% Train, 20% Val per folder)")
                            .size(16.0)
                            .color(egui::Color32::BLACK)
                            .family(egui::FontFamily::Name("PoppinsBold".into())));
                    });
                    ui.add_space(10.0);
                    
                    ui.horizontal(|ui| {
                        ui.add_space(20.0);
                        
                        ui.vertical(|ui| {
                            egui::Frame::none()
                                .fill(egui::Color32::WHITE)
                                .inner_margin(15.0)
                                .rounding(5.0)
                                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 200, 200)))
                                .show(ui, |ui| {
                                    ui.set_width(590.0);
                                    ui.set_height(220.0);
                                    
                                    ui.vertical_centered(|ui| {
                                        ui.label(egui::RichText::new("High grade")
                                            .size(18.0)
                                            .color(egui::Color32::BLACK)
                                            .family(egui::FontFamily::Name("PoppinsBold".into())));
                                    });
                                    ui.add_space(8.0);
                                    
                                    egui::ScrollArea::vertical()
                                        .id_source("high_grade_scroll")
                                        .max_height(170.0)
                                        .auto_shrink([false; 2])
                                        .show(ui, |ui| {
                                            self.draw_tree_view(ui, true);
                                        });
                                });
                        });
                        
                        ui.add_space(15.0);
                        
                        ui.vertical(|ui| {
                            egui::Frame::none()
                                .fill(egui::Color32::WHITE)
                                .inner_margin(15.0)
                                .rounding(5.0)
                                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 200, 200)))
                                .show(ui, |ui| {
                                    ui.set_width(590.0);
                                    ui.set_height(220.0);
                                    
                                    ui.vertical_centered(|ui| {
                                        ui.label(egui::RichText::new("Low Grade")
                                            .size(18.0)
                                            .color(egui::Color32::BLACK)
                                            .family(egui::FontFamily::Name("PoppinsBold".into())));
                                    });
                                    ui.add_space(8.0);
                                    
                                    egui::ScrollArea::vertical()
                                        .id_source("low_grade_scroll")
                                        .max_height(170.0)
                                        .auto_shrink([false; 2])
                                        .show(ui, |ui| {
                                            self.draw_tree_view(ui, false);
                                        });
                                });
                        });
                    });
                    ui.add_space(15.0);
                }
                
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    
                    ui.vertical(|ui| {
                        ui.group(|ui| {
                            ui.set_width(590.0);
                            ui.set_height(250.0);
                            ui.vertical_centered(|ui| {
                                ui.label(egui::RichText::new("Train")
                                    .size(20.0)
                                    .family(egui::FontFamily::Name("PoppinsBold".into())));
                            });
                            self.draw_metrics_plot(ui, true);
                        });
                    });
                    
                    ui.add_space(15.0);
                    
                    ui.vertical(|ui| {
                        ui.group(|ui| {
                            ui.set_width(590.0);
                            ui.set_height(250.0);
                            ui.vertical_centered(|ui| {
                                ui.label(egui::RichText::new("Validation")
                                    .size(20.0)
                                    .family(egui::FontFamily::Name("PoppinsBold".into())));
                            });
                            self.draw_metrics_plot(ui, false);
                        });
                    });
                });
                ui.add_space(15.0);
                
                if self.training_complete && self.train_eval.is_some() && self.val_eval.is_some() {
                    let train_eval = self.train_eval.as_ref().unwrap();
                    let val_eval = self.val_eval.as_ref().unwrap();
                    
                    ui.horizontal(|ui| {
                        ui.add_space(20.0);
                        ui.vertical(|ui| {
                            self.draw_eval_tables(ui, "Train", train_eval);
                        });
                        ui.add_space(15.0);
                        ui.vertical(|ui| {
                            self.draw_eval_tables(ui, "Validation", val_eval);
                        });
                    });
                    ui.add_space(15.0);
                }
                
                ui.add_space(10.0);
                
                  ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    ui.add_space(30.0);
                    
                    ui.horizontal(|ui| {
                        ui.add_space(20.0);
                        
                        // START TRAINING Button
                        let button_size = egui::vec2(200.0, 40.0);
                        let (rect, response) = ui.allocate_exact_size(button_size, egui::Sense::click());
                        
                        let (fill_color, icon, text) = if self.is_training {
                            (egui::Color32::from_rgb(255, 140, 0), "⏳", "TRAINING...")
                        } else {
                            (egui::Color32::from_rgb(40, 130, 50), "▶", "START TRAINING")
                        };
                        
                        let fill_color = if response.hovered() && !self.is_training {
                            egui::Color32::from_rgb(50, 140, 60)
                        } else {
                            fill_color
                        };
                        
                        ui.painter().rect_filled(rect, 4.0, fill_color);
                        
                        let icon_galley = ui.painter().layout_no_wrap(
                            icon.to_string(),
                            egui::FontId::proportional(16.0),
                            egui::Color32::WHITE,
                        );
                        let text_galley = ui.painter().layout_no_wrap(
                            format!(" {}", text),
                            egui::FontId::new(15.0, egui::FontFamily::Name("PoppinsBold".into())),
                            egui::Color32::WHITE,
                        );
                        
                        let total_width = icon_galley.size().x + text_galley.size().x;
                        let start_x = rect.center().x - total_width / 2.0;
                        let icon_pos = egui::pos2(
                            start_x,
                            rect.center().y - icon_galley.size().y / 2.0,
                        );
                        let text_pos = egui::pos2(
                            start_x + icon_galley.size().x,
                            rect.center().y - text_galley.size().y / 2.0,
                        );
                        
                        ui.painter().galley(icon_pos, icon_galley, egui::Color32::WHITE);
                        ui.painter().galley(text_pos, text_galley, egui::Color32::WHITE);
                        
                        if response.clicked() && !self.is_training {
                            self.start_training();
                        }
                        
                        ui.add_space(15.0);
                        
                        // PCA Button
                        if !self.training_complete {
                            let pca_button_size = egui::vec2(150.0, 40.0);
                            let (pca_rect, pca_response) = ui.allocate_exact_size(pca_button_size, egui::Sense::click());
                            let pca_color = if pca_response.hovered() {
                                egui::Color32::from_rgb(30, 100, 130)
                            } else {
                                egui::Color32::from_rgb(15, 92, 112)
                            };
                            ui.painter().rect_filled(pca_rect, 4.0, pca_color);
                            let pca_text = ui.painter().layout_no_wrap(
                                "PCA".to_string(),
                                egui::FontId::new(15.0, egui::FontFamily::Name("PoppinsBold".into())),
                                egui::Color32::WHITE,
                            );
                            let pca_text_pos = egui::pos2(
                                pca_rect.center().x - pca_text.size().x / 2.0,
                                pca_rect.center().y - pca_text.size().y / 2.0,
                            );
                            ui.painter().galley(pca_text_pos, pca_text, egui::Color32::WHITE);
                            if pca_response.clicked() {
                                self.generate_pca_visualization();
                            }
                        }
                        
                        // Manage Data Button
                        if !self.training_complete {
                            let manage_button_size = egui::vec2(150.0, 40.0);
                            let (manage_rect, manage_response) = ui.allocate_exact_size(manage_button_size, egui::Sense::click());
                            let manage_color = if manage_response.hovered() {
                                egui::Color32::from_rgb(120, 20, 20)
                            } else {
                                egui::Color32::from_rgb(100, 0, 0)
                            };
                            ui.painter().rect_filled(manage_rect, 4.0, manage_color);
                            let manage_text = ui.painter().layout_no_wrap(
                                "Manage Data".to_string(),
                                egui::FontId::new(15.0, egui::FontFamily::Name("PoppinsBold".into())),
                                egui::Color32::WHITE,
                            );
                            let manage_text_pos = egui::pos2(
                                manage_rect.center().x - manage_text.size().x / 2.0,
                                manage_rect.center().y - manage_text.size().y / 2.0,
                            );
                            ui.painter().galley(manage_text_pos, manage_text, egui::Color32::WHITE);
                            if manage_response.clicked() {
                                Self::open_file_explorer();
                            }
                        }
                    });
                    
                    // Display elapsed time when training is complete
                                       if self.training_complete {
                        if let Some(final_time) = self.final_time {
                            let mins = (final_time / 60.0).floor() as i32;
                            let secs = (final_time % 60.0).floor() as i32;
                            
                            ui.vertical_centered(|ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{}m {}s ({:.2}min)",
                                        mins,
                                        secs,
                                        final_time / 60.0
                                    ))
                                    .size(16.0)
                                    .color(egui::Color32::BLACK)
                                    .family(egui::FontFamily::Name("PoppinsBold".into())),
                                );
                            });
                        }
                    }
                });
            });
    }
}



impl TrainingGUI {
    fn draw_metrics_plot(&self, ui: &mut egui::Ui, is_train: bool) {
        let history = if let Ok(history) = self.metrics_history.lock() {
            history.clone()
        } else {
            Vec::new()
        };
        
        if history.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("Waiting for training data...")
                    .family(egui::FontFamily::Name("Poppins".into())));
            });
            return;
        }
        
        let max_epoch = history.len() as f64;
        
        Plot::new(if is_train { "train_plot" } else { "val_plot" })
            .legend(Legend::default().position(Corner::RightBottom))
            .show_axes([true, true])
            .show_grid([true, true])
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .x_axis_label("Epoch")
            .height(200.0)
            .width(570.0)
            .include_x(0.0)
            .include_x(max_epoch.min(100.0))
            .include_y(0.0)
            .include_y(1.0)
            .show(ui, |plot_ui| {
                let epochs: Vec<f64> = (1..=history.len()).map(|i| i as f64).collect();
                
                let (loss_data, acc_data) = if is_train {
                    (
                        history.iter().map(|m| m.train_loss as f64).collect::<Vec<_>>(),
                        history.iter().map(|m| m.train_accuracy as f64).collect::<Vec<_>>(),
                    )
                } else {
                    (
                        history.iter().map(|m| m.val_loss as f64).collect::<Vec<_>>(),
                        history.iter().map(|m| m.val_accuracy as f64).collect::<Vec<_>>(),
                    )
                };
                
                let loss_points: PlotPoints = epochs.iter()
                    .zip(loss_data.iter())
                    .map(|(x, y)| [*x, *y])
                    .collect();
                
                plot_ui.line(Line::new(loss_points)
                    .color(egui::Color32::from_rgb(51, 102, 255))
                    .width(2.0)
                    .name("loss"));
                
                let acc_points: PlotPoints = epochs.iter()
                    .zip(acc_data.iter())
                    .map(|(x, y)| [*x, *y])
                    .collect();
                
                plot_ui.line(Line::new(acc_points)
                    .color(egui::Color32::from_rgb(255, 140, 0))
                    .width(2.0)
                    .name("accuracy"));
            });
    }
    
    fn draw_eval_tables(&self, ui: &mut egui::Ui, _title: &str, eval: &EvaluationResults) {
        let table_width = 590.0;
        
        ui.vertical(|ui| {
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(240, 235, 220))
                .inner_margin(10.0)
                .rounding(5.0)
                .show(ui, |ui| {
                    ui.set_width(table_width);
                    
                    ui.label(egui::RichText::new("Per-Class Metrics:")
                        .color(egui::Color32::BLACK)
                        .size(14.0)
                        .family(egui::FontFamily::Name("PoppinsBold".into())));
                    ui.add_space(5.0);
                    
                    egui::Grid::new(format!("metrics_{}", eval.accuracy))
                        .striped(false)
                        .spacing([20.0, 5.0])
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("Metric")
                                .color(egui::Color32::BLACK)
                                .strong()
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new("High Grade")
                                .color(egui::Color32::BLACK)
                                .strong()
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new("Low Grade")
                                .color(egui::Color32::BLACK)
                                .strong()
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.end_row();
                            
                            ui.label(egui::RichText::new("Precision")
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{:.3}", eval.high_grade_metrics.precision))
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{:.3}", eval.low_grade_metrics.precision))
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.end_row();
                            
                            ui.label(egui::RichText::new("Recall")
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{:.3}", eval.high_grade_metrics.recall))
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{:.3}", eval.low_grade_metrics.recall))
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.end_row();
                            
                            ui.label(egui::RichText::new("F1-Score")
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{:.3}", eval.high_grade_metrics.f1_score))
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{:.3}", eval.low_grade_metrics.f1_score))
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.end_row();
                            
                            ui.label(egui::RichText::new("Support")
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{}", eval.high_grade_metrics.support))
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{}", eval.low_grade_metrics.support))
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.end_row();
                        });
                });
            
            ui.add_space(5.0);
            
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(240, 235, 220))
                .inner_margin(10.0)
                .rounding(5.0)
                .show(ui, |ui| {
                    ui.set_width(table_width);
                    
                    ui.label(egui::RichText::new("Confusion Matrix:")
                        .color(egui::Color32::BLACK)
                        .size(14.0)
                        .family(egui::FontFamily::Name("PoppinsBold".into())));
                    ui.add_space(5.0);
                    
                    egui::Grid::new(format!("cm_{}", eval.accuracy))
                        .striped(false)
                        .spacing([20.0, 5.0])
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("")
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new("Pred: High")
                                .color(egui::Color32::BLACK)
                                .strong()
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new("Pred: Low")
                                .color(egui::Color32::BLACK)
                                .strong()
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.end_row();
                            
                            ui.label(egui::RichText::new("True: High")
                                .color(egui::Color32::BLACK)
                                .strong()
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{}", eval.confusion_matrix[0][0]))
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{}", eval.confusion_matrix[0][1]))
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.end_row();
                            
                            ui.label(egui::RichText::new("True: Low")
                                .color(egui::Color32::BLACK)
                                .strong()
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{}", eval.confusion_matrix[1][0]))
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.label(egui::RichText::new(format!("{}", eval.confusion_matrix[1][1]))
                                .color(egui::Color32::BLACK)
                                .family(egui::FontFamily::Name("Poppins".into())));
                            ui.end_row();
                        });
                });
            
            ui.add_space(5.0);
            
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(240, 235, 220))
                .inner_margin(10.0)
                .rounding(5.0)
                .show(ui, |ui| {
                    ui.set_width(table_width);
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new(format!("Accuracy: {:.1}%", eval.accuracy * 100.0))
                            .color(egui::Color32::BLACK)
                            .size(16.0)
                            .strong()
                            .family(egui::FontFamily::Name("Poppins".into())));
                    });
                });
        });
    }
}

fn main() -> eframe::Result<()> {
    env_logger::init();
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1250.0, 900.0])
            .with_title("CNN Training"),
        ..Default::default()
    };
    
    eframe::run_native(
        "CNN Training",
        options,
        Box::new(|cc| {
            setup_fonts(&cc.egui_ctx);
            Ok(Box::new(TrainingGUI::new()))
        }),
    )
}

fn setup_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    
    if let Ok(font_data) = std::fs::read("assets/fonts/Poppins-Regular.ttf") {
        fonts.font_data.insert("Poppins".to_owned(), egui::FontData::from_owned(font_data));
        fonts.families.get_mut(&egui::FontFamily::Proportional).unwrap()
            .insert(0, "Poppins".to_owned());
        fonts.families.insert(egui::FontFamily::Name("Poppins".into()), vec!["Poppins".to_owned()]);
    }
    
    if let Ok(font_data) = std::fs::read("assets/fonts/Poppins-Bold.ttf") {
        fonts.font_data.insert("PoppinsBold".to_owned(), egui::FontData::from_owned(font_data));
        fonts.families.get_mut(&egui::FontFamily::Proportional).unwrap()
            .insert(0, "PoppinsBold".to_owned());
        fonts.families.insert(egui::FontFamily::Name("PoppinsBold".into()), vec!["PoppinsBold".to_owned()]);
    }
    
    ctx.set_fonts(fonts);
}
