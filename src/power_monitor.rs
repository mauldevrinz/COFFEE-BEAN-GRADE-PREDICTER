// src/power_monitor.rs
// Real-time power monitoring via tegrastats (NVIDIA Jetson / Tegra platform)

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

// ─── Data structures ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct PowerSample {
    pub elapsed_secs: f32,
    pub vdd_in_mw: u32,       // Total system power (mW)
    pub vdd_cpu_gpu_mw: u32,  // CPU + GPU + CV power (mW)
    pub vdd_soc_mw: u32,      // SOC power (mW)
    pub cpu_temp_c: f32,      // CPU temperature (°C)
    pub gpu_temp_c: f32,      // GPU temperature (°C)
}

#[derive(Clone, Debug, Default)]
pub struct PowerSummary {
    pub avg_total_w: f32,
    pub peak_total_w: f32,
    pub avg_cpu_gpu_w: f32,
    pub peak_cpu_gpu_w: f32,
    pub avg_cpu_temp: f32,
    pub peak_cpu_temp: f32,
    pub avg_gpu_temp: f32,
    pub peak_gpu_temp: f32,
    /// avg_total_W × duration_secs (Joules)
    pub energy_joules: f32,
    pub duration_secs: f32,
    pub sample_count: usize,
}

// ─── PowerMonitor ────────────────────────────────────────────────────────────

pub struct PowerMonitor {
    pub samples: Arc<Mutex<Vec<PowerSample>>>,
    stop_flag: Arc<AtomicBool>,
    is_stopped: Arc<AtomicBool>,
}

impl PowerMonitor {
    /// Spawn tegrastats in background and start collecting samples.
    pub fn start() -> Self {
        let samples = Arc::new(Mutex::new(Vec::<PowerSample>::new()));
        let stop_flag = Arc::new(AtomicBool::new(false));
        let is_stopped = Arc::new(AtomicBool::new(false));
        let start_time = Instant::now();

        let samples_clone = Arc::clone(&samples);
        let stop_clone = Arc::clone(&stop_flag);
        let stopped_clone = Arc::clone(&is_stopped);

        thread::spawn(move || {
            let mut child = match Command::new("tegrastats")
                .arg("--interval")
                .arg("200")
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
            {
                Ok(c) => c,
                Err(_) => return,
            };

            let stdout = match child.stdout.take() {
                Some(s) => s,
                None => return,
            };

            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if stop_clone.load(Ordering::Relaxed) {
                    let _ = child.kill();
                    break;
                }
                if let Ok(line) = line {
                    let elapsed = start_time.elapsed().as_secs_f32();
                    if let Some(sample) = parse_tegrastats_line(&line, elapsed) {
                        if let Ok(mut v) = samples_clone.lock() {
                            v.push(sample);
                        }
                    }
                }
            }
            stopped_clone.store(true, Ordering::Relaxed);
        });

        Self { samples, stop_flag, is_stopped }
    }

    /// Signal the tegrastats process to stop.
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }

    /// Returns true if the monitor has been stopped.
    pub fn is_stopped(&self) -> bool {
        self.is_stopped.load(Ordering::Relaxed)
    }

    /// Number of samples collected so far.
    pub fn sample_count(&self) -> usize {
        self.samples.lock().map(|v| v.len()).unwrap_or(0)
    }

    /// Clear all collected samples (use when restarting a training run).
    pub fn clear(&self) {
        if let Ok(mut v) = self.samples.lock() {
            v.clear();
        }
    }

    /// Latest sample, or None if no data yet.
    pub fn current_sample(&self) -> Option<PowerSample> {
        self.samples.lock().ok()?.last().cloned()
    }

    /// Running total energy consumed so far (Joules).
    /// Each sample represents ~0.2 seconds (200ms interval) at that wattage level.
    pub fn accumulated_joules(&self) -> f32 {
        self.accumulated_joules_from(0)
    }

    /// Energy consumed from a specific sample index onwards (Joules).
    pub fn accumulated_joules_from(&self, start_idx: usize) -> f32 {
        self.samples
            .lock()
            .map(|v| v[start_idx.min(v.len())..].iter().map(|s| s.vdd_in_mw as f32 / 1000.0 * 0.2).sum::<f32>())
            .unwrap_or(0.0)
    }

    /// Compute summary statistics over all collected samples.
    pub fn summary(&self) -> Option<PowerSummary> {
        self.summary_from(0)
    }

    /// Compute summary statistics only for samples from start_idx onwards (training-period only).
    pub fn summary_from(&self, start_idx: usize) -> Option<PowerSummary> {
        let samples = self.samples.lock().ok()?;
        let slice = &samples[start_idx.min(samples.len())..];
        if slice.is_empty() {
            return None;
        }
        let n = slice.len() as f32;
        let avg_total_w = slice.iter().map(|s| s.vdd_in_mw as f32).sum::<f32>() / n / 1000.0;
        let peak_total_w = slice.iter().map(|s| s.vdd_in_mw as f32).fold(0.0f32, f32::max) / 1000.0;
        let avg_cpu_gpu_w = slice.iter().map(|s| s.vdd_cpu_gpu_mw as f32).sum::<f32>() / n / 1000.0;
        let peak_cpu_gpu_w = slice.iter().map(|s| s.vdd_cpu_gpu_mw as f32).fold(0.0f32, f32::max) / 1000.0;
        let avg_cpu_temp = slice.iter().map(|s| s.cpu_temp_c).sum::<f32>() / n;
        let peak_cpu_temp = slice.iter().map(|s| s.cpu_temp_c).fold(0.0f32, f32::max);
        let avg_gpu_temp = slice.iter().map(|s| s.gpu_temp_c).sum::<f32>() / n;
        let peak_gpu_temp = slice.iter().map(|s| s.gpu_temp_c).fold(0.0f32, f32::max);
        let duration_secs = slice.last().map(|s| s.elapsed_secs).unwrap_or(0.0)
            - slice.first().map(|s| s.elapsed_secs).unwrap_or(0.0);
        let energy_joules: f32 = slice.iter().map(|s| s.vdd_in_mw as f32 / 1000.0 * 0.2).sum();

        Some(PowerSummary {
            avg_total_w,
            peak_total_w,
            avg_cpu_gpu_w,
            peak_cpu_gpu_w,
            avg_cpu_temp,
            peak_cpu_temp,
            avg_gpu_temp,
            peak_gpu_temp,
            energy_joules,
            duration_secs,
            sample_count: slice.len(),
        })
    }
}

// ─── tegrastats parser ────────────────────────────────────────────────────────

fn parse_tegrastats_line(line: &str, elapsed: f32) -> Option<PowerSample> {
    let vdd_in_mw = parse_mw(line, "VDD_IN")?;
    let vdd_cpu_gpu_mw = parse_mw(line, "VDD_CPU_GPU_CV").unwrap_or(0);
    let vdd_soc_mw = parse_mw(line, "VDD_SOC").unwrap_or(0);
    let cpu_temp_c = parse_temp(line, "cpu@").unwrap_or(0.0);
    let gpu_temp_c = parse_temp(line, "gpu@").unwrap_or(0.0);

    Some(PowerSample {
        elapsed_secs: elapsed,
        vdd_in_mw,
        vdd_cpu_gpu_mw,
        vdd_soc_mw,
        cpu_temp_c,
        gpu_temp_c,
    })
}

/// Parse `KEY NNNNmW/...` → NNNNu32.
fn parse_mw(line: &str, key: &str) -> Option<u32> {
    let pos = line.find(key)?;
    let after = line[pos + key.len()..].trim_start();
    let num: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

/// Parse `key@NNN.NNC` → NNN.NN f32.
fn parse_temp(line: &str, key: &str) -> Option<f32> {
    let pos = line.find(key)?;
    let after = &line[pos + key.len()..];
    let num: String = after.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    num.parse().ok()
}
