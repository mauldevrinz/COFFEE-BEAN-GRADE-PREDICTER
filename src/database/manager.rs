//! SQLite Database Module untuk Electronic Nose
//! Enhanced logging untuk debugging

use rusqlite::{params, Connection, Result};
use std::path::Path;

/// Sensor reading struct
#[derive(Debug, Clone)]
pub struct SensorReading {
    pub time_seconds: u32,
    pub tgs2600: f32,
    pub mq135: f32,
    pub mq3: f32,
    pub mq6: f32,
    pub mq7: f32,
    pub tgs2602: f32,
    pub tgs2611: f32,
    pub tgs2620: f32,
}

/// Database manager
pub struct DatabaseManager {
    connection: Connection,
}

impl DatabaseManager {
    /// Create database connection
    pub fn new(db_path: &str) -> Result<Self> {
        if db_path != ":memory:" {
            if let Some(parent) = Path::new(db_path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
        }

        let connection = Connection::open(db_path)?;
        println!("📁 Database connected: {}", db_path);

        Ok(Self { connection })
    }

    /// Initialize database tables
    pub fn initialize(&self) -> Result<()> {
        // FORCE DROP old table
        self.connection.execute("DROP TABLE IF EXISTS sensor_readings", [])?;
        println!("🗑️  Old table dropped if exists");

        // CREATE new table with correct structure
        self.connection.execute(
            "CREATE TABLE sensor_readings (
                time_seconds INTEGER PRIMARY KEY,
                tgs2600 REAL NOT NULL,
                mq135 REAL NOT NULL,
                mq3 REAL NOT NULL,
                mq6 REAL NOT NULL,
                mq7 REAL NOT NULL,
                tgs2602 REAL NOT NULL,
                tgs2611 REAL NOT NULL,
                tgs2620 REAL NOT NULL
            )",
            [],
        )?;

        println!("✅ Database table created with correct structure");
        Ok(())
    }

    /// Insert sensor reading with enhanced logging
    pub fn insert_reading(&self, reading: &SensorReading) -> Result<()> {
        // Log every insertion for first 10 seconds, then every 10 seconds
        if reading.time_seconds <= 10 || reading.time_seconds % 10 == 0 {
            println!(
                "💾 Inserting: time_seconds={}, mq3={:.0}, mq6={:.0}",
                reading.time_seconds, reading.mq3, reading.mq6
            );
        }

        self.connection.execute(
            "INSERT OR REPLACE INTO sensor_readings
            (time_seconds, tgs2600, mq135, mq3, mq6, mq7, tgs2602, tgs2611, tgs2620)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                reading.time_seconds,
                reading.tgs2600,
                reading.mq135,
                reading.mq3,
                reading.mq6,
                reading.mq7,
                reading.tgs2602,
                reading.tgs2611,
                reading.tgs2620
            ],
        )?;

        // Confirm insertion for critical first records
        if reading.time_seconds == 1 {
            println!("🎯 FIRST RECORD INSERTED (time_seconds=1)");
        }

        Ok(())
    }

    /// Count total readings
    pub fn count_readings(&self) -> Result<i64> {
        let count: i64 = self.connection.query_row(
            "SELECT COUNT(*) FROM sensor_readings",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Get all readings
    pub fn get_all_readings(&self) -> Result<Vec<SensorReading>> {
        let mut stmt = self.connection.prepare(
            "SELECT time_seconds, tgs2600, mq135, mq3, mq6, mq7, tgs2602, tgs2611, tgs2620
            FROM sensor_readings
            ORDER BY time_seconds ASC",
        )?;

        let reading_iter = stmt.query_map([], |row| {
            Ok(SensorReading {
                time_seconds: row.get(0)?,
                tgs2600: row.get(1)?,
                mq135: row.get(2)?,
                mq3: row.get(3)?,
                mq6: row.get(4)?,
                mq7: row.get(5)?,
                tgs2602: row.get(6)?,
                tgs2611: row.get(7)?,
                tgs2620: row.get(8)?,
            })
        })?;

        let mut readings = Vec::new();
        for reading in reading_iter {
            readings.push(reading?);
        }

        Ok(readings)
    }

    /// Clear all readings
    #[allow(dead_code)]
    pub fn clear_all(&self) -> Result<()> {
        self.connection.execute("DELETE FROM sensor_readings", [])?;
        println!("🗑️  Database cleared");
        Ok(())
    }

    /// Get first and last timestamps for debugging
    #[allow(dead_code)]
    pub fn get_time_range(&self) -> Result<(u32, u32)> {
        let first: u32 = self
            .connection
            .query_row("SELECT MIN(time_seconds) FROM sensor_readings", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);

        let last: u32 = self
            .connection
            .query_row("SELECT MAX(time_seconds) FROM sensor_readings", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);

        Ok((first, last))
    }
}

