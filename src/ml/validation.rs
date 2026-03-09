// src/ml/validation.rs - Pure Rust CSV Validation

use anyhow::Result;
use colored::*;
use csv::ReaderBuilder;
use std::fs::File;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

struct FileValidation {
    #[allow(dead_code)]
    path: PathBuf,
    is_valid: bool,
    num_rows: usize,
    num_cols: usize,
    errors: Vec<String>,
}

struct DatasetValidation {
    total_files: usize,
    valid_files: usize,
    invalid_files: usize,
    files: Vec<FileValidation>,
}

fn validate_csv_file(path: &Path, expected_sensors: usize, expected_rows: Option<usize>) -> FileValidation {
    let mut validation = FileValidation {
        path: path.to_path_buf(),
        is_valid: true,
        num_rows: 0,
        num_cols: 0,
        errors: Vec::new(),
    };

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            validation.is_valid = false;
            validation.errors.push(format!("Cannot open file: {}", e));
            return validation;
        }
    };

    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let headers = match reader.headers() {
        Ok(h) => h,
        Err(e) => {
            validation.is_valid = false;
            validation.errors.push(format!("Cannot read headers: {}", e));
            return validation;
        }
    };

    let expected_cols = expected_sensors + 1;
    validation.num_cols = headers.len();

    if headers.len() != expected_cols {
        validation.is_valid = false;
        validation.errors.push(format!(
            "Expected {} columns (1 timestamp + {} sensors), got {}",
            expected_cols, expected_sensors, headers.len()
        ));
    }

    let mut row_count = 0;
    for (i, result) in reader.records().enumerate() {
        match result {
            Ok(record) => {
                row_count += 1;

                if record.len() != expected_cols {
                    validation.is_valid = false;
                    validation.errors.push(format!("Row {}: expected {} columns, got {}", i + 1, expected_cols, record.len()));
                }

                for (j, field) in record.iter().enumerate().skip(1) {
                    if field.parse::<f64>().is_err() {
                        validation.is_valid = false;
                        validation.errors.push(format!("Row {}, Col {}: '{}' is not a valid number", i + 1, j + 1, field));
                        break;
                    }
                }
            }
            Err(e) => {
                validation.is_valid = false;
                validation.errors.push(format!("Row {}: {}", i + 1, e));
            }
        }

        if validation.errors.len() > 5 {
            validation.errors.push("... (more errors)".to_string());
            break;
        }
    }

    validation.num_rows = row_count;

    if let Some(expected) = expected_rows {
        if row_count != expected {
            validation.is_valid = false;
            validation.errors.push(format!("Expected {} rows, got {}", expected, row_count));
        }
    }

    validation
}

fn validate_dataset(data_root: &Path, expected_sensors: usize, expected_rows: Option<usize>) -> Result<DatasetValidation> {
    let mut dataset_validation = DatasetValidation {
        total_files: 0,
        valid_files: 0,
        invalid_files: 0,
        files: Vec::new(),
    };

    println!("\n{}", "🔍 Validating Dataset...".bold().cyan());
    println!("📂 Root: {}", data_root.display());
    println!("📊 Expected: {} sensors, {:?} rows\n", expected_sensors, expected_rows);

    for entry in WalkDir::new(data_root)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("csv") {
            continue;
        }

        dataset_validation.total_files += 1;

        let validation = validate_csv_file(path, expected_sensors, expected_rows);

        if validation.is_valid {
            dataset_validation.valid_files += 1;
            println!("{} {}", "✓".green(), path.display());
        } else {
            dataset_validation.invalid_files += 1;
            println!("{} {}", "✗".red(), path.display());
            for error in &validation.errors {
                println!("   {} {}", "→".yellow(), error);
            }
        }

        dataset_validation.files.push(validation);
    }

    Ok(dataset_validation)
}

fn print_summary(validation: &DatasetValidation) {
    println!("\n{}", "📋 Validation Summary:".bold().cyan());
    println!("┌─────────────────┬──────────┐");
    println!("│ Total Files     │ {:>8} │", validation.total_files);
    println!("│ Valid Files     │ {:>8} │", validation.valid_files.to_string().green());
    println!("│ Invalid Files   │ {:>8} │", validation.invalid_files.to_string().red());
    println!("└─────────────────┴──────────┘");

    if validation.invalid_files > 0 {
        println!("\n{}", "⚠️  Some files have issues. Please fix them before training.".yellow().bold());
    } else {
        println!("\n{}", "✅ All files are valid! Ready for training.".green().bold());
    }
}

fn check_dataset_structure(data_root: &Path) -> Result<()> {
    println!("\n{}", "🗂️  Checking Dataset Structure...".bold().cyan());

    let high_grade = data_root.join("high_grade");
    let low_grade = data_root.join("low_grade");

    if !high_grade.exists() {
        println!("{} Missing: high_grade/", "✗".red());
    } else {
        println!("{} Found: high_grade/", "✓".green());
        check_grade_structure(&high_grade)?;
    }

    if !low_grade.exists() {
        println!("{} Missing: low_grade/", "✗".red());
    } else {
        println!("{} Found: low_grade/", "✓".green());
        check_grade_structure(&low_grade)?;
    }

    Ok(())
}

fn check_grade_structure(grade_path: &Path) -> Result<()> {
    let samples: Vec<_> = std::fs::read_dir(grade_path)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    for sample in samples {
        let sample_path = sample.path();
        let sample_name = sample_path.file_name().unwrap().to_str().unwrap();

        let mut train_count = 0;
        let mut val_count = 0;
        let mut test_count = 0;

        for entry in WalkDir::new(&sample_path)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("csv") {
                continue;
            }

            let filename = path.file_name().unwrap().to_str().unwrap();
            if filename.starts_with("train_") {
                train_count += 1;
            } else if filename.starts_with("val_") {
                val_count += 1;
            } else if filename.starts_with("test_") {
                test_count += 1;
            }
        }

        let status = if train_count == 8 && val_count == 1 && test_count == 1 {
            format!("{} {} (train: {}, val: {}, test: {})", "✓".green(), sample_name, train_count, val_count, test_count)
        } else {
            format!("{} {} (train: {}, val: {}, test: {}) - Expected: 8 train, 1 val, 1 test", "⚠".yellow(), sample_name, train_count, val_count, test_count)
        };

        println!("   {}", status);
    }

    Ok(())
}

#[allow(dead_code)]
fn main() -> Result<()> {
    env_logger::init();

    let data_root = PathBuf::from("data/raw");

    println!("\n{}", "╔═══════════════════════════════════════════╗".bold());
    println!("{}", "║ Coffee Dataset Validator (Pure Rust)  ║".bold());
    println!("{}", "╚═══════════════════════════════════════════╝".bold());

    check_dataset_structure(&data_root)?;

    let validation = validate_dataset(&data_root, 8, Some(300))?;

    print_summary(&validation);

    if validation.invalid_files > 0 {
        std::process::exit(1);
    }

    Ok(())
}

