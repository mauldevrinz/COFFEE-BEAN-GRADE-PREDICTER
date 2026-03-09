use polars::prelude::*;
use rusqlite::{Connection, Row};
use std::fs::File;
use polars_io::parquet::ParquetWriter;
use polars_io::csv::CsvWriter;

fn main() -> PolarsResult<()> {
    // --- 1. Load data dari sensor_data.db ---
    let conn = Connection::open("data/sensor_data.db").expect("Failed to open sensor_data.db");
    let mut stmt = conn.prepare("SELECT * FROM sensor_readings ORDER BY time_seconds").unwrap();
    let rows = stmt.query_map([], |row| Ok(row_to_vec(row))).unwrap();
    let mut data1: Vec<Vec<f64>> = Vec::new();
    for r in rows { data1.push(r.unwrap()); }

    // --- 2. Load data dari sensor_datatest.db ---
    let conn_test = Connection::open("data/sensor_datatest.db").expect("Failed to open sensor_datatest.db");
    let mut stmt_test = conn_test.prepare("SELECT * FROM sensor_readings ORDER BY time_seconds").unwrap();
    let rows_test = stmt_test.query_map([], |row| Ok(row_to_vec(row))).unwrap();
    let mut data2: Vec<Vec<f64>> = Vec::new();
    for r in rows_test { data2.push(r.unwrap()); }

    // --- 3. Gabungkan kedua dataset menjadi satu ---
    let mut combined_data = data1;
    combined_data.extend(data2);

    // --- 4. Buat DataFrame dari data yang sudah digabung ---
    let df = DataFrame::new(vec![
        Series::new("time_seconds", combined_data.iter().map(|v| v[0]).collect::<Vec<_>>()),
        Series::new("tgs2600", combined_data.iter().map(|v| v[1]).collect::<Vec<_>>()),
        Series::new("mq135", combined_data.iter().map(|v| v[2]).collect::<Vec<_>>()),
        Series::new("mq3", combined_data.iter().map(|v| v[3]).collect::<Vec<_>>()),
        Series::new("mq6", combined_data.iter().map(|v| v[4]).collect::<Vec<_>>()),
        Series::new("mq7", combined_data.iter().map(|v| v[5]).collect::<Vec<_>>()),
        Series::new("tgs2602", combined_data.iter().map(|v| v[6]).collect::<Vec<_>>()),
        Series::new("tgs2611", combined_data.iter().map(|v| v[7]).collect::<Vec<_>>()),
        Series::new("tgs2620", combined_data.iter().map(|v| v[8]).collect::<Vec<_>>()),
    ])?;

    // --- 5. Simpan sebagai training data ke Parquet dan CSV ---
    let mut pq = File::create("ml_data/train_data.parquet").unwrap();
    ParquetWriter::new(&mut pq).finish(&mut df.clone())?;

    let mut csv = File::create("ml_data/train_data.csv").unwrap();
    CsvWriter::new(&mut csv).finish(&mut df.clone())?;

    println!("Combined Training Data: {} rows (sensor_data + sensor_datatest)", df.height());
    println!("Data saved to ml_data/train_data.parquet and ml_data/train_data.csv");

    Ok(())
}

// Helper: konversi Row rusqlite ke Vec<f64>
fn row_to_vec(row: &Row) -> Vec<f64> {
    vec![
        row.get::<_, f64>(0).unwrap_or(0.0),
        row.get::<_, f64>(1).unwrap_or(0.0),
        row.get::<_, f64>(2).unwrap_or(0.0),
        row.get::<_, f64>(3).unwrap_or(0.0),
        row.get::<_, f64>(4).unwrap_or(0.0),
        row.get::<_, f64>(5).unwrap_or(0.0),
        row.get::<_, f64>(6).unwrap_or(0.0),
        row.get::<_, f64>(7).unwrap_or(0.0),
        row.get::<_, f64>(8).unwrap_or(0.0),
    ]
}

