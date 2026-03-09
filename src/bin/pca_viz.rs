// src/bin/pca_viz.rs
//! PCA Visualization - Pure Rust with Plotters

use csv::ReaderBuilder;
use nalgebra::DMatrix;
use plotters::prelude::*;
use plotters::style::Color as PlottersColor;
use std::process::Command;

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
    
    let mut matrix_data: Vec<f64> = Vec::new();
    for row in data {
        matrix_data.extend(row);
    }
    let data_matrix = DMatrix::from_row_slice(n_samples, n_features, &matrix_data);
    
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        match load_csv(path) {
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
    let standardized_data = standardize(&all_samples);
    
    println!("📊 Computing PCA...");
    let (components, eigenvalues) = compute_pca(&standardized_data);
    
    let total_var: f64 = eigenvalues.iter().sum();
    let var_ratio1 = (eigenvalues[0] / total_var * 100.0 * 100.0).round() / 100.0;
    let var_ratio2 = (eigenvalues[1] / total_var * 100.0 * 100.0).round() / 100.0;
    
    println!("   PC1: {:.2}%", var_ratio1);
    println!("   PC2: {:.2}%", var_ratio2);
    
    println!("\n🔄 Transforming to PC space...");
    let n_samples = standardized_data.len();
    let n_features = standardized_data[0].len();
    
    let mut data_flat: Vec<f64> = Vec::new();
    for row in &standardized_data {
        data_flat.extend(row);
    }
    let data_matrix = DMatrix::from_row_slice(n_samples, n_features, &data_flat);
    
    let transformed: DMatrix<f64> = &data_matrix * &components;
    
    let mut pca_data: Vec<(f64, f64, usize)> = Vec::new();
    for i in 0..n_samples {
        let pc1 = transformed[(i, 0)];
        let pc2 = transformed[(i, 1)];
        pca_data.push((pc1, pc2, labels[i]));
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
            let ellipse_points = generate_ellipse((mean_x, mean_y), cov, 2.0, 200);
            
            let color = PCA_COLORS[label];
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
