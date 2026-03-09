// src/analysis.rs - Pure Rust Data Analysis & Statistics

use anyhow::Result;
use colored::*;
use csv::ReaderBuilder;
use ndarray::Array2;
use statrs::statistics::{Data, Distribution, OrderStatistics, Min, Max};
use std::fs::File;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

struct SensorStatistics {
    sensor_id: usize,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    q25: f64,
    median: f64,
    q75: f64,
}

struct DatasetStatistics {
    high_grade_stats: Vec<SensorStatistics>,
    low_grade_stats: Vec<SensorStatistics>,
    high_grade_samples: usize,
    low_grade_samples: usize,
}

fn load_all_csv_data(grade_path: &Path) -> Result<Array2<f64>> {
    let mut all_data = Vec::new();

    for entry in WalkDir::new(grade_path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("csv") {
            continue;
        }

        let file = File::open(path)?;
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

        for result in reader.records() {
            let record = result?;
            let row: Vec<f64> = record
                .iter()
                .skip(1) // Skip timestamp
                .take(8) // 8 sensors
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
                .collect();
            all_data.extend(row);
        }
    }

    if all_data.is_empty() {
        anyhow::bail!("No data found in {}", grade_path.display());
    }

    let num_rows = all_data.len() / 8;
    let array = Array2::from_shape_vec((num_rows, 8), all_data)?;
    Ok(array)
}

fn compute_sensor_statistics(data: &Array2<f64>) -> Vec<SensorStatistics> {
    let mut stats = Vec::new();

    for sensor_id in 0..8 {
        let sensor_data: Vec<f64> = data.column(sensor_id).to_vec();
        let mut data_stats = Data::new(sensor_data.clone());

        let mean = data_stats.mean().unwrap_or(0.0);
        let std = data_stats.std_dev().unwrap_or(0.0);
        let min = data_stats.min();
        let max = data_stats.max();
        let q25 = data_stats.quantile(0.25);
        let median = data_stats.median();
        let q75 = data_stats.quantile(0.75);

        stats.push(SensorStatistics {
            sensor_id: sensor_id + 1,
            mean,
            std,
            min,
            max,
            q25,
            median,
            q75,
        });
    }

    stats
}

fn analyze_dataset(data_root: &Path) -> Result<DatasetStatistics> {
    println!("\n{}", "📊 Analyzing Dataset...".bold().cyan());

    let high_grade_path = data_root.join("high_grade");
    let high_grade_data = load_all_csv_data(&high_grade_path)?;
    let high_grade_stats = compute_sensor_statistics(&high_grade_data);
    println!("{} High grade: {} samples", "✓".green(), high_grade_data.nrows());

    let low_grade_path = data_root.join("low_grade");
    let low_grade_data = load_all_csv_data(&low_grade_path)?;
    let low_grade_stats = compute_sensor_statistics(&low_grade_data);
    println!("{} Low grade: {} samples", "✓".green(), low_grade_data.nrows());

    Ok(DatasetStatistics {
        high_grade_stats,
        low_grade_stats,
        high_grade_samples: high_grade_data.nrows(),
        low_grade_samples: low_grade_data.nrows(),
    })
}

fn print_statistics_table(stats: &DatasetStatistics) {
    println!("\n{}", "═══════════════════════════════════════════════════════".bold());
    println!("{}", "                HIGH GRADE STATISTICS                  ".bold().green());
    println!("{}", "═══════════════════════════════════════════════════════".bold());
    print_sensor_table(&stats.high_grade_stats);

    println!("\n{}", "═══════════════════════════════════════════════════════".bold());
    println!("{}", "                LOW GRADE STATISTICS                   ".bold().red());
    println!("{}", "═══════════════════════════════════════════════════════".bold());
    print_sensor_table(&stats.low_grade_stats);

    println!("\n{}", "═══════════════════════════════════════════════════════".bold());
    println!("{}", "              COMPARATIVE ANALYSIS                     ".bold().yellow());
    println!("{}", "═══════════════════════════════════════════════════════".bold());
    print_comparative_analysis(stats);
}

fn print_sensor_table(stats: &[SensorStatistics]) {
    println!("┌────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("│ Sensor │   Mean   │   Std    │   Min    │   Q25    │  Median  │   Q75    │   Max    │");
    println!("├────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤");

    for stat in stats {
        println!(
            "│ {:>6} │ {:>8.2} │ {:>8.2} │ {:>8.2} │ {:>8.2} │ {:>8.2} │ {:>8.2} │ {:>8.2} │",
            stat.sensor_id, stat.mean, stat.std, stat.min, stat.q25, stat.median, stat.q75, stat.max
        );
    }

    println!("└────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘");
}

fn print_comparative_analysis(stats: &DatasetStatistics) {
    println!("┌────────┬───────────────────┬───────────────────┬──────────────────────┐");
    println!("│ Sensor │  High Mean±Std    │  Low Mean±Std     │  Mean Difference     │");
    println!("├────────┼───────────────────┼───────────────────┼──────────────────────┤");

    for i in 0..8 {
        let high = &stats.high_grade_stats[i];
        let low = &stats.low_grade_stats[i];
        let diff = high.mean - low.mean;
        let diff_pct = (diff / high.mean * 100.0).abs();

        let diff_str = if diff.abs() > 0.1 {
            format!("{:+.2} ({:.1}%)", diff, diff_pct).yellow().to_string()
        } else {
            format!("{:+.2} ({:.1}%)", diff, diff_pct)
        };

        println!(
            "│ {:>6} │ {:>7.2} ± {:>6.2} │ {:>7.2} ± {:>6.2} │ {:>20} │",
            high.sensor_id, high.mean, high.std, low.mean, low.std, diff_str
        );
    }

    println!("└────────┴───────────────────┴───────────────────┴──────────────────────┘");

    println!("\n{}", "📈 Sample Distribution:".bold());
    println!("  High Grade: {} samples", stats.high_grade_samples.to_string().green());
    println!("  Low Grade: {} samples", stats.low_grade_samples.to_string().red());
    println!("  Total: {} samples", (stats.high_grade_samples + stats.low_grade_samples).to_string().cyan());

    let balance = (stats.high_grade_samples as f64 / stats.low_grade_samples as f64 * 100.0).round();
    println!("  Balance: {:.0}% / {:.0}%", balance, 100.0);
}

fn check_data_quality(stats: &DatasetStatistics) {
    println!("\n{}", "🔍 Data Quality Checks:".bold().cyan());

    let mut issues = Vec::new();

    for stat in &stats.high_grade_stats {
        if stat.std < 0.01 {
            issues.push(format!("High Grade Sensor {}: Very low variance (std={:.6})", stat.sensor_id, stat.std));
        }
    }

    for stat in &stats.low_grade_stats {
        if stat.std < 0.01 {
            issues.push(format!("Low Grade Sensor {}: Very low variance (std={:.6})", stat.sensor_id, stat.std));
        }
    }

    for stat in stats.high_grade_stats.iter().chain(stats.low_grade_stats.iter()) {
        let range = stat.max - stat.min;
        if range > 1000.0 {
            issues.push(format!("Sensor {}: Very large range ({:.2})", stat.sensor_id, range));
        }
    }

    let ratio = stats.high_grade_samples as f64 / stats.low_grade_samples as f64;
    if ratio > 1.5 || ratio < 0.67 {
        issues.push(format!("Class imbalance: {:.2}:1 (High:Low)", ratio));
    }

    if issues.is_empty() {
        println!("  {} No major issues detected!", "✓".green());
    } else {
        println!("  {} {} potential issues:", "⚠".yellow(), issues.len());
        for issue in issues {
            println!("   • {}", issue);
        }
    }
}

#[allow(dead_code)]
fn main() -> Result<()> {
    env_logger::init();

    let data_root = PathBuf::from("data/raw");

    println!("\n{}", "╔═══════════════════════════════════════════╗".bold());
    println!("{}", "║  Coffee Dataset Analyzer (Pure Rust)  ║".bold());
    println!("{}", "╚═══════════════════════════════════════════╝".bold());

    let stats = analyze_dataset(&data_root)?;
    print_statistics_table(&stats);
    check_data_quality(&stats);

    println!("\n{} Analysis complete!", "✓".green().bold());

    Ok(())
}

