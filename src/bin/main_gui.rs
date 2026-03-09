//! Electronic Nose - COMPLETE VERSION with ML Integration + Poppins Font
//! FIXED: Graph persists after STOP button, lines remain visible

use coffee_classifier::database::{DatabaseManager, SensorReading};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Sender, Receiver};
use std::{thread, process::Command};
use std::time::Duration;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

const MAX_POINTS: usize = 600;

#[derive(Default, Clone)]
struct SensorData {
    tgs2600: f32,
    mq135: f32,
    mq3: f32,
    mq6: f32,
    mq7: f32,
    tgs2602: f32,
    tgs2611: f32,
    tgs2620: f32,

    inlet_pump_on: bool,
    inlet_valve_on: bool,
    outlet_pump_on: bool,
    outlet_valve_on: bool,

    inlet_time: u32,
    outlet_time: u32,
    elapsed_time: u32,
    connected: bool,

    #[allow(dead_code)]
    timestamp: u64,
}


#[derive(Clone)]
struct SensorHistory {
    timestamps: VecDeque<f64>,
    mq3_data: VecDeque<f64>,
    mq6_data: VecDeque<f64>,
    mq7_data: VecDeque<f64>,
    mq135_data: VecDeque<f64>,
    tgs2600_data: VecDeque<f64>,
    tgs2602_data: VecDeque<f64>,
    tgs2611_data: VecDeque<f64>,
    tgs2620_data: VecDeque<f64>,
}

impl Default for SensorHistory {
    fn default() -> Self {
        Self {
            timestamps: VecDeque::with_capacity(MAX_POINTS),
            mq3_data: VecDeque::with_capacity(MAX_POINTS),
            mq6_data: VecDeque::with_capacity(MAX_POINTS),
            mq7_data: VecDeque::with_capacity(MAX_POINTS),
            mq135_data: VecDeque::with_capacity(MAX_POINTS),
            tgs2600_data: VecDeque::with_capacity(MAX_POINTS),
            tgs2602_data: VecDeque::with_capacity(MAX_POINTS),
            tgs2611_data: VecDeque::with_capacity(MAX_POINTS),
            tgs2620_data: VecDeque::with_capacity(MAX_POINTS),
        }
    }
}

impl SensorHistory {
    fn add_data(&mut self, data: &SensorData, current_second: u32) {
        let timestamp = current_second as f64;
        self.timestamps.push_back(timestamp);
        self.mq3_data.push_back(data.mq3 as f64);
        self.mq6_data.push_back(data.mq6 as f64);
        self.mq7_data.push_back(data.mq7 as f64);
        self.mq135_data.push_back(data.mq135 as f64);
        self.tgs2600_data.push_back(data.tgs2600 as f64);
        self.tgs2602_data.push_back(data.tgs2602 as f64);
        self.tgs2611_data.push_back(data.tgs2611 as f64);
        self.tgs2620_data.push_back(data.tgs2620 as f64);

        if self.timestamps.len() > MAX_POINTS {
            self.timestamps.pop_front();
            self.mq3_data.pop_front();
            self.mq6_data.pop_front();
            self.mq7_data.pop_front();
            self.mq135_data.pop_front();
            self.tgs2600_data.pop_front();
            self.tgs2602_data.pop_front();
            self.tgs2611_data.pop_front();
            self.tgs2620_data.pop_front();
        }
    }

    fn reset(&mut self) {
        self.timestamps.clear();
        self.mq3_data.clear();
        self.mq6_data.clear();
        self.mq7_data.clear();
        self.mq135_data.clear();
        self.tgs2600_data.clear();
        self.tgs2602_data.clear();
        self.tgs2611_data.clear();
        self.tgs2620_data.clear();
    }
}

struct CoffeeQualityApp {
    sensor_data: Arc<Mutex<SensorData>>,
    sensor_history: Arc<Mutex<SensorHistory>>,
    database: Arc<Mutex<Option<DatabaseManager>>>,
    db_status: Arc<Mutex<String>>,
    data_counter: Arc<Mutex<u32>>,
    sample_name: String,
    command_sender: Sender<String>,
    system_running: Arc<Mutex<bool>>,
    button_disabled: bool,
    uploaded_file: Option<String>,
    db_created: bool,
}

fn setup_custom_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    // Load Regular font
    let regular_font_path = std::path::Path::new("assets/fonts/Poppins-Regular.ttf");
    if regular_font_path.exists() {
        fonts.font_data.insert(
            "Poppins".to_owned(),  // [OK] Nama kunci: "Poppins"
            egui::FontData::from_static(include_bytes!("../../assets/fonts/Poppins-Regular.ttf")),
        );
        println!("[OK] Poppins Regular loaded");
    }

    // Load Bold font
    let bold_font_path = std::path::Path::new("assets/fonts/Poppins-Bold.ttf");
    if bold_font_path.exists() {
        fonts.font_data.insert(
            "PoppinsBold".to_owned(),  // [OK] Nama kunci: "PoppinsBold"
            egui::FontData::from_static(include_bytes!("../../assets/fonts/Poppins-Bold.ttf")),
        );
        println!("[OK] Poppins Bold loaded");
    }

    // [OK] Register custom font families dengan NAMA YANG SAMA
    fonts
        .families
        .insert(
            egui::FontFamily::Name("Poppins".into()),
            vec!["Poppins".to_owned()]
        );

    fonts
        .families
        .insert(
            egui::FontFamily::Name("PoppinsBold".into()),
            vec!["PoppinsBold".to_owned()]
        );

    // [OK] Set Poppins sebagai default font
    fonts
        .families
        .get_mut(&egui::FontFamily::Proportional)
        .unwrap()
        .insert(0, "Poppins".to_owned());

    ctx.set_fonts(fonts);
}


fn parse_json_data(json_str: &str) -> Result<SensorData, Box<dyn std::error::Error>> {
    let v: Value = serde_json::from_str(json_str)?;
    let sensors = &v["sensors"];
    let status = &v["status"];

    Ok(SensorData {
        tgs2600: sensors["tgs2600"].as_f64().unwrap_or(0.0) as f32,
        mq135: sensors["mq135"].as_f64().unwrap_or(0.0) as f32,
        mq3: sensors["mq3"].as_f64().unwrap_or(0.0) as f32,
        mq6: sensors["mq6"].as_f64().unwrap_or(0.0) as f32,
        mq7: sensors["mq7"].as_f64().unwrap_or(0.0) as f32,
        tgs2602: sensors["tgs2602"].as_f64().unwrap_or(0.0) as f32,
        tgs2611: sensors["tgs2611"].as_f64().unwrap_or(0.0) as f32,
        tgs2620: sensors["tgs2620"].as_f64().unwrap_or(0.0) as f32,
        inlet_pump_on: status["inlet_pump"].as_bool().unwrap_or(false),
inlet_valve_on: status["inlet_valve"].as_bool().unwrap_or(false),
outlet_pump_on: status["outlet_pump"].as_bool().unwrap_or(false),
outlet_valve_on: status["outlet_valve"].as_bool().unwrap_or(false),
inlet_time: status["inlet_time"].as_u64().unwrap_or(0) as u32,
outlet_time: status["outlet_time"].as_u64().unwrap_or(0) as u32,
elapsed_time: status["elapsed_time"].as_u64().unwrap_or(0) as u32,
        connected: true,
        timestamp: v["timestamp"].as_u64().unwrap_or(0),
    })
}

impl CoffeeQualityApp {
    fn new() -> Self {
        println!("🔧 Initializing app...");
        
        let sensor_data = Arc::new(Mutex::new(SensorData::default()));
        let sensor_history = Arc::new(Mutex::new(SensorHistory::default()));
        let database = Arc::new(Mutex::new(None));
        let db_status = Arc::new(Mutex::new("[ ] No Database".to_string()));
        let data_counter = Arc::new(Mutex::new(0u32));
        let system_running = Arc::new(Mutex::new(false));
        
        let (tx, rx) = channel();

        let app = Self {
            sensor_data: Arc::clone(&sensor_data),
            sensor_history: Arc::clone(&sensor_history),
            database,
            db_status,
            data_counter,
            sample_name: String::new(),
            command_sender: tx,
            system_running,
            button_disabled: false,
            uploaded_file: None,
            db_created: false,
        };

        app.start_serial_thread(rx);
        app
    }

    fn start_serial_thread(&self, command_rx: Receiver<String>) {
        let sensor_data = Arc::clone(&self.sensor_data);
        let sensor_history = Arc::clone(&self.sensor_history);
        let database = Arc::clone(&self.database);
        let db_status = Arc::clone(&self.db_status);
        let data_counter = Arc::clone(&self.data_counter);
        let system_running = Arc::clone(&self.system_running);

        thread::spawn(move || {
            println!("🔌 Serial thread started");
            
         let port_priorities = vec!["COM11"];  // [OK] HARDCODED ke COM11 saja

            let mut connection_reset = true;

            loop {
                if connection_reset {
                    if let Ok(mut counter) = data_counter.lock() {
                        *counter = 0;
                    }
                    if let Ok(mut running) = system_running.lock() {
                        *running = false;
                    }
                    connection_reset = false;
                }

                let mut port_name = String::new();
                let mut found = false;

                for &priority_port in &port_priorities {
                    if serialport::new(priority_port, 115200).open().is_ok() {
                        port_name = priority_port.to_string();
                        found = true;
                        println!("[OK] Using port: {}", port_name);
                        break;
                    }
                }

                if !found {
                    if let Ok(ports) = serialport::available_ports() {
                        for port in ports {
                            if port.port_name.contains("COM")
                                || port.port_name.contains("ttyACM")
                                || port.port_name.contains("ttyUSB")
                            {
                                port_name = port.port_name.clone();
                                found = true;
                                println!("🔍 Auto-detected: {}", port_name);
                                break;
                            }
                        }
                    }
                }

                if !found || port_name.is_empty() {
                    if let Ok(mut data) = sensor_data.lock() {
                        data.connected = false;
                    }
                    if let Ok(mut status) = db_status.lock() {
                        if !status.contains("Database") {
                            *status = "[-] No Serial Port".to_string();
                        }
                    }
                    thread::sleep(Duration::from_secs(2));
                    continue;
                }

                match serialport::new(&port_name, 115200)
                    .timeout(Duration::from_millis(3000))
                    .open()
                {
                    Ok(mut port) => {
                        println!("[+] Connected to {}", port_name);
                        
                        if let Ok(mut data) = sensor_data.lock() {
                            data.connected = true;
                        }

                        let mut reader = BufReader::new(port.try_clone().unwrap());

                        loop {
                            if let Ok(command) = command_rx.try_recv() {
                                println!("📤 Sending: {}", command);
                                if let Err(e) = port.write_all(format!("{}\n", command).as_bytes()) {
                                    eprintln!("[X] Write error: {}", e);
                                    connection_reset = true;
                                    break;
                                }
                            }

                            let mut line_buffer = String::new();
                            match reader.read_line(&mut line_buffer) {
                                Ok(0) => {
                                    println!("[-] Port disconnected");
                                    connection_reset = true;
                                    break;
                                }
                                Ok(_) => {
                                    let trimmed = line_buffer.trim();
                                    if !trimmed.is_empty()
                                        && trimmed.starts_with('{')
                                        && trimmed.ends_with('}')
                                    {
                                        match parse_json_data(trimmed) {
                                            Ok(parsed_data) => {
                                                if let Ok(mut data) = sensor_data.lock() {
                                                    *data = parsed_data.clone();
                                                    data.connected = true;
                                                }

                                                let should_record = if let Ok(running) = system_running.lock() {
                                                    *running
                                                } else {
                                                    false
                                                };

                                                if should_record && parsed_data.elapsed_time >= 1 {
                                                    let current_count = parsed_data.elapsed_time;
                                                    
                                                    if let Ok(mut counter) = data_counter.lock() {
                                                        *counter = current_count;
                                                    }

                                                    if current_count <= 600 {
                                                        if let Ok(mut history) = sensor_history.lock() {
                                                            history.add_data(&parsed_data, current_count);
                                                        }
                                                    }

                                                    if current_count <= 600 {
                                                        if let Ok(db_opt) = database.lock() {
                                                            if let Some(ref db) = *db_opt {
                                                                let reading = SensorReading {
                                                                    time_seconds: current_count,
                                                                    tgs2600: parsed_data.tgs2600,
                                                                    mq135: parsed_data.mq135,
                                                                    mq3: parsed_data.mq3,
                                                                    mq6: parsed_data.mq6,
                                                                    mq7: parsed_data.mq7,
                                                                    tgs2602: parsed_data.tgs2602,
                                                                    tgs2611: parsed_data.tgs2611,
                                                                    tgs2620: parsed_data.tgs2620,
                                                                };

                                                                match db.insert_reading(&reading) {
                                                                    Ok(_) => {
                                                                        if current_count <= 10 || current_count % 10 == 0 {
                                                                            println!(
                                                                                "📝 {}s | DB Saved | MQ3:{:.0} MQ6:{:.0} MQ7:{:.0} MQ135:{:.0}",
                                                                                current_count,
                                                                                parsed_data.mq3,
                                                                                parsed_data.mq6,
                                                                                parsed_data.mq7,
                                                                                parsed_data.mq135
                                                                            );
                                                                        }
                                                                        if let Ok(mut status) = db_status.lock() {
                                                                            *status = format!("[+] Recording {}/600s", current_count);
                                                                        }
                                                                    }
                                                                    Err(e) => {
                                                                        eprintln!("[X] DB Error: {}", e);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }

                                                    if current_count >= 600 {
                                                        println!("[OK] REACHED 600s!");
                                                        if let Ok(mut status) = db_status.lock() {
                                                            *status = "[OK] Complete! 600s recorded".to_string();
                                                        }
                                                        if let Ok(mut running) = system_running.lock() {
                                                            *running = false;
                                                        }
                                                    }
                                                }
                                            }
                                            Err(_) => {}
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("[X] Read Error: {}", e);
                                    connection_reset = true;
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[X] Failed: {}", e);
                        if let Ok(mut data) = sensor_data.lock() {
                            data.connected = false;
                        }
                        thread::sleep(Duration::from_secs(2));
                    }
                }
            }
        });
    }
}

impl eframe::App for CoffeeQualityApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let current_data = if let Ok(data) = self.sensor_data.lock() {
            data.clone()
        } else {
            SensorData::default()
        };

        let current_history = if let Ok(history) = self.sensor_history.lock() {
            history.clone()
        } else {
            SensorHistory::default()
        };

        let db_status_text = if let Ok(status) = self.db_status.lock() {
            status.clone()
        } else {
            "Unknown".to_string()
        };

        if let Ok(running) = self.system_running.lock() {
            self.button_disabled = *running;
        }

        let mut visuals = egui::Visuals::light();
        visuals.window_fill = egui::Color32::WHITE;
        ctx.set_visuals(visuals);

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                let viewport_size = ctx.screen_rect().size();
                draw_header(ui, viewport_size, current_data.connected, &db_status_text);

                let content_rect = egui::Rect::from_min_size(
                    egui::pos2(0.0, 70.0),
                    egui::vec2(viewport_size.x, viewport_size.y - 70.0),
                );
                ui.painter().rect_filled(
                    content_rect,
                    egui::Rounding::ZERO,
                    egui::Color32::from_rgb(220, 220, 220),
                );

                let margins = 15.0;
                let uniform_spacing = 15.0;
                let left_column_width = 270.0;
                let gas_panel_height = 170.0;
                let gas_panel_spacing = 12.0;
                let gas_panel_x_offset = 120.0;
                let gas_panel_x = margins + gas_panel_x_offset;

                let inlet_rect = egui::Rect::from_min_size(
                    egui::pos2(gas_panel_x, 82.0),
                    egui::vec2(left_column_width, gas_panel_height),
                );
                ui.allocate_ui_at_rect(inlet_rect, |ui| {
                    draw_gas_panel(
                        ui,
                        "INLET GAS STATUS",
                        true,
                        current_data.inlet_pump_on,
                        current_data.inlet_valve_on,
                        current_data.inlet_time,
                    );
                });

                let outlet_spacing_extra = 45.0;
                let outlet_rect = egui::Rect::from_min_size(
                    egui::pos2(gas_panel_x, 82.0 + gas_panel_height + gas_panel_spacing + outlet_spacing_extra),
                    egui::vec2(left_column_width, gas_panel_height),
                );
                ui.allocate_ui_at_rect(outlet_rect, |ui| {
                    draw_gas_panel(
                        ui,
                        "OUTLET GAS STATUS",
                        false,
                        current_data.outlet_pump_on,
                        current_data.outlet_valve_on,
                        current_data.outlet_time,
                    );
                });

                let right_column_x = gas_panel_x + left_column_width + uniform_spacing;
                let adc_width = 780.0;
                let adc_height = 170.0;
                let adc_rect = egui::Rect::from_min_size(
                    egui::pos2(right_column_x, 82.0),
                    egui::vec2(adc_width, adc_height),
                );
                ui.allocate_ui_at_rect(adc_rect, |ui| {
                    draw_adc_panel(ui, &current_data);
                });

                let graph_y = 82.0 + adc_height + 8.0;
                let graph_width = 780.0;
                let graph_height = viewport_size.y - graph_y - margins;
                let graph_rect = egui::Rect::from_min_size(
                    egui::pos2(right_column_x, graph_y),
                    egui::vec2(graph_width, graph_height),
                );
                ui.allocate_ui_at_rect(graph_rect, |ui| {
                    draw_graph_panel(ui, &current_history);
                });

                let sample_panel_width = 200.0;
                let sample_panel_x = right_column_x + adc_width + uniform_spacing;
                let sample_panel_rect = egui::Rect::from_min_size(
                    egui::pos2(sample_panel_x, 82.0),
                    egui::vec2(sample_panel_width, 170.0),
                );
                ui.allocate_ui_at_rect(sample_panel_rect, |ui| {
                    draw_sample_panel(
                        ui,
                        &mut self.sample_name,
                        &self.command_sender,
                        &self.system_running,
                        &self.data_counter,
                        &self.sensor_history,
                        self.button_disabled,
                    );
                });

                let classification_panel_y = 82.0 + 170.0 + 15.0;
                let classification_panel_rect = egui::Rect::from_min_size(
                    egui::pos2(sample_panel_x, classification_panel_y),
                    egui::vec2(sample_panel_width, 340.0),
                );
                ui.allocate_ui_at_rect(classification_panel_rect, |ui| {
                    draw_classification_panel_inline(
                        ui,
                        &mut self.uploaded_file,
                        &self.database,
                        &self.db_status,
                        &mut self.db_created,
                    );
                });
            });

        ctx.request_repaint_after(Duration::from_millis(100));
    }
}

fn draw_header(ui: &mut egui::Ui, viewport_size: egui::Vec2, connected: bool, db_status: &str) {
    let header_height = 70.0;
    let header_rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(viewport_size.x, header_height));
    ui.painter().rect_filled(header_rect, egui::Rounding::ZERO, egui::Color32::from_rgb(28, 94, 124));

    ui.painter().text(
        egui::pos2(header_rect.center().x, header_rect.center().y + 2.0),
        egui::Align2::CENTER_CENTER,
        "ELECTRONIC NOSE FOR CLASSIFICATION ARABICA COFFEE QUALITY",
        egui::FontId::new(20.0, egui::FontFamily::Name("PoppinsBold".into())),
        egui::Color32::WHITE,
    );

    ui.allocate_ui_at_rect(header_rect, |ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
            ui.add_space(20.0);

            let db_color = if db_status.contains("Error") {
                egui::Color32::RED
            } else if db_status.contains("Complete") {
                egui::Color32::from_rgb(0, 200, 255)
            } else if db_status.contains("Recording") {
                egui::Color32::from_rgb(0, 255, 0)
            } else if db_status.contains("Created") {
                egui::Color32::YELLOW
            } else {
                egui::Color32::LIGHT_GRAY
            };

            ui.label(
                egui::RichText::new(format!("[DB] {}", db_status))
                    .size(12.0)
                    .family(egui::FontFamily::Name("Poppins".into()))
                    .color(db_color),
            );

            ui.add_space(20.0);

            let status_color = if connected {
                egui::Color32::from_rgb(0, 255, 0)
            } else {
                egui::Color32::RED
            };
            let status_text = if connected { "[+] CONNECTED" } else { "[-] DISCONNECTED" };

            ui.label(
                egui::RichText::new(status_text)
                    .size(14.0)
                    .family(egui::FontFamily::Name("Poppins".into()))
                    .color(status_color),
            );
        });
    });
}

fn draw_gas_panel(
    ui: &mut egui::Ui,
    title: &str,
    is_inlet: bool,
    pump_on: bool,
    valve_on: bool,
    elapsed_time: u32,
) {
    let panel_width = 270.0;
    let panel_height = 170.0;
    let panel_rect = egui::Rect::from_min_size(ui.cursor().min, egui::vec2(panel_width, panel_height));

    ui.painter().rect_stroke(
        panel_rect,
        egui::Rounding::ZERO,
        egui::Stroke::new(2.0, egui::Color32::BLACK),
    );

    let header_color = if is_inlet {
        egui::Color32::from_rgb(34, 100, 34)
    } else {
        egui::Color32::from_rgb(139, 69, 69)
    };

    let header_rect = egui::Rect::from_min_size(panel_rect.min, egui::vec2(panel_width, 32.0));
    ui.painter().rect_filled(header_rect, egui::Rounding::ZERO, header_color);

    let content_background_rect = egui::Rect::from_min_size(
        egui::pos2(panel_rect.min.x + 2.0, panel_rect.min.y + 32.0),
        egui::vec2(panel_width - 4.0, panel_height - 34.0),
    );
    ui.painter().rect_filled(content_background_rect, egui::Rounding::ZERO, egui::Color32::WHITE);

    ui.painter().text(
        egui::pos2(header_rect.center().x, header_rect.center().y),
        egui::Align2::CENTER_CENTER,
        title,
        egui::FontId::new(18.0, egui::FontFamily::Name("PoppinsBold".into())),
        egui::Color32::WHITE,
    );

    ui.allocate_ui_at_rect(
        egui::Rect::from_min_size(
            egui::pos2(panel_rect.min.x + 18.0, panel_rect.min.y + 45.0),
            egui::vec2(panel_width - 36.0, panel_height - 60.0),
        ),
        |ui| {
            ui.vertical(|ui| {
                ui.add_space(20.0);

                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("PUMP")
                            .size(16.0)
                            .family(egui::FontFamily::Name("PoppinsBold".into()))
                            .strong(),
                    );
                    ui.add_space(30.0);
                    draw_status_indicator(ui, pump_on);
                    ui.add_space(18.0);
                    ui.label(
                        egui::RichText::new(format!("{} S", elapsed_time))
                            .family(egui::FontFamily::Name("Poppins".into()))
                            .color(egui::Color32::BLACK),
                    );
                });

                ui.add_space(25.0);

                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("VALVE")
                            .size(16.0)
                            .family(egui::FontFamily::Name("PoppinsBold".into()))
                            .strong(),
                    );
                    ui.add_space(22.0);
                    draw_status_indicator(ui, valve_on);
                    ui.add_space(18.0);
                    ui.label(
                        egui::RichText::new(format!("{} S", elapsed_time))
                            .family(egui::FontFamily::Name("Poppins".into()))
                            .color(egui::Color32::BLACK),
                    );
                });
            });
        },
    );

    ui.allocate_space(egui::vec2(panel_width, panel_height));
}

fn draw_classification_panel_inline(
    ui: &mut egui::Ui,
    uploadedfile: &mut Option<String>,
    database: &Arc<Mutex<Option<DatabaseManager>>>,
    dbstatus: &Arc<Mutex<String>>,
    dbcreated: &mut bool,
) {
    let panelwidth = 200.0;
    let panelheight = 340.0;
    let panelrect = egui::Rect::from_min_size(ui.cursor().min, egui::vec2(panelwidth, panelheight));
    
    // Panel border
    ui.painter().rect_stroke(
        panelrect,
        egui::Rounding::ZERO,
        egui::Stroke::new(2.0, egui::Color32::BLACK),
    );

    // Header "CLASSIFICATION"
    let headercolor = egui::Color32::from_rgb(204, 153, 0);
    let headerrect = egui::Rect::from_min_size(panelrect.min, egui::vec2(panelwidth, 32.0));
    ui.painter().rect_filled(headerrect, egui::Rounding::ZERO, headercolor);
    ui.painter().text(
        egui::pos2(headerrect.center().x, headerrect.center().y),
        egui::Align2::CENTER_CENTER,
        "CLASSIFICATION",
        egui::FontId::new(16.0, egui::FontFamily::Name("PoppinsBold".into())),
        egui::Color32::WHITE,
    );

    // Content background (white)
    let contentbackgroundrect = egui::Rect::from_min_size(
        egui::pos2(panelrect.min.x + 2.0, panelrect.min.y + 32.0),
        egui::vec2(panelwidth - 4.0, panelheight - 34.0),
    );
    ui.painter().rect_filled(contentbackgroundrect, egui::Rounding::ZERO, egui::Color32::WHITE);

    let contentpadding = 10.0;
    ui.allocate_ui_at_rect(
        egui::Rect::from_min_size(
            egui::pos2(panelrect.min.x + contentpadding, panelrect.min.y + 32.0 + contentpadding),
            egui::vec2(panelwidth - contentpadding * 2.0, panelheight - 32.0 - contentpadding * 2.0),
        ),
        |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(3.0);

                let uploadwidth = panelwidth - contentpadding * 2.0 - 4.0;

                // ===== TEXT INPUT: Create Database =====
                let mut temppath = uploadedfile.clone().unwrap_or_default();
                let textedit = egui::TextEdit::singleline(&mut temppath)
                    .font(egui::FontId::new(12.0, egui::FontFamily::Name("Poppins".into())))
                    .hint_text("Create Database")
                    .desired_width(uploadwidth - 8.0)
                    .frame(true);

                if ui.add(textedit).changed() {
                    if !temppath.is_empty() {
                        *uploadedfile = Some(temppath);
                    } else {
                        *uploadedfile = None;
                    }
                }

                ui.add_space(5.0);

                // ===== BUTTON 1: [DB] CREATE =====
                let createbutton = egui::Button::new(
                    egui::RichText::new("[DB] CREATE")
                        .size(12.0)
                        .family(egui::FontFamily::Name("PoppinsBold".into()))
                        .color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(108, 117, 125))
                .min_size(egui::vec2(uploadwidth, 28.0));

                if ui.add(createbutton).clicked() {
                    if let Some(dbname) = uploadedfile.clone() {
                        let dbfile = if dbname.ends_with(".db") {
                            dbname
                        } else {
                            format!("{}.db", dbname)
                        };

                        match DatabaseManager::new(&dbfile) {
                            Ok(dbmanager) => {
                                match dbmanager.initialize() {
                                    Ok(_) => {
                                        println!("[OK] Database created: {}", dbfile);
                                        *uploadedfile = Some(dbfile.clone());
                                        *dbcreated = true;

                                        if let Ok(mut dbopt) = database.lock() {
                                            *dbopt = Some(dbmanager);
                                        }

                                        if let Ok(mut status) = dbstatus.lock() {
                                            *status = format!("Database Created: {}", dbfile);
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("[X] Failed to initialize database: {}", e);
                                        if let Ok(mut status) = dbstatus.lock() {
                                            *status = format!("Error: {}", e);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("[X] Failed to create database: {}", e);
                                if let Ok(mut status) = dbstatus.lock() {
                                    *status = format!("Error: {}", e);
                                }
                            }
                        }
                    }
                }

                ui.add_space(5.0);

                // ===== BUTTON 2: CNN TRAINING (FULL WIDTH) =====
                              // ===== BUTTON 2: CNN TRAINING (FULL WIDTH) =====
                let cnn_training_button = egui::Button::new(
                    egui::RichText::new("CNN TRAINING")
                        .size(12.0)
                        .family(egui::FontFamily::Name("PoppinsBold".into()))
                        .color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(23, 162, 184))
                .min_size(egui::vec2(uploadwidth, 30.0));

              if ui.add(cnn_training_button).clicked() {
                    println!("🚀 CNN TRAINING GUI clicked!");
                    
                    // [OK] Force clean build untuk memastikan binary terbaru
                    std::thread::spawn(|| {
                        println!("🔧 Checking for latest train_gui binary...");
                        
                        // [OK] Prioritize RELEASE build (most likely to be updated)
                        let possible_paths = vec![
                            "./target/release/train_gui.exe",    // Windows release (PRIORITY)
                            "./target/release/train_gui",        // Linux/Mac release (PRIORITY)
                            "target/release/train_gui.exe",
                            "target/release/train_gui",
                            "./target/debug/train_gui.exe",      // Windows debug (fallback)
                            "./target/debug/train_gui",          // Linux/Mac debug (fallback)
                            "target/debug/train_gui.exe",
                            "target/debug/train_gui",
                        ];
                        
                        let mut binary_path = None;
                        let mut binary_timestamp = None;
                        
                        // Find the NEWEST binary
                        for path in &possible_paths {
                            if let Ok(metadata) = std::fs::metadata(path) {
                                if let Ok(modified) = metadata.modified() {
                                    if binary_timestamp.is_none() || Some(modified) > binary_timestamp {
                                        binary_path = Some(path.to_string());
                                        binary_timestamp = Some(modified);
                                        println!("📁 Found binary: {} (modified: {:?})", path, modified);
                                    }
                                }
                            }
                        }
                        
                        let binary_path = match binary_path {
                            Some(path) => {
                                println!("[OK] Using newest binary: {}", path);
                                path
                            },
                            None => {
                                eprintln!("[X] Error: train_gui binary not found!");
                                eprintln!("\n📋 Build instructions:");
                                eprintln!("   1. Clean old builds:");
                                eprintln!("      cargo clean");
                                eprintln!("   2. Build release version:");
                                eprintln!("      cargo build --bin train_gui --release");
                                eprintln!("   3. Or build debug version:");
                                eprintln!("      cargo build --bin train_gui");
                                return;
                            }
                        };
                        
                        // Launch the GUI
                        println!("🚀 Launching CNN Training GUI...");
                        match Command::new(&binary_path).spawn() {
                            Ok(mut child) => {
                                println!("[OK] CNN Training GUI launched successfully!");
                                println!("   Binary: {}", binary_path);
                                println!("   PID: {:?}", child.id());
                                
                                // Wait for completion in background
                                std::thread::spawn(move || {
                                    match child.wait() {
                                        Ok(status) => {
                                            println!("📊 Training GUI closed (status: {})", status);
                                        }
                                        Err(e) => {
                                            eprintln!("[X] Error waiting for GUI: {}", e);
                                        }
                                    }
                                });
                            }
                            Err(e) => {
                                eprintln!("[X] Failed to launch training GUI!");
                                eprintln!("   Binary: {}", binary_path);
                                eprintln!("   Error: {}", e);
                                eprintln!("\n💡 Troubleshooting:");
                                eprintln!("   1. Rebuild: cargo build --bin train_gui --release");
                                eprintln!("   2. Check permissions (Linux/Mac): chmod +x {}", binary_path);
                            }
                        }
                    });
                }

                ui.add_space(3.0);

                // ===== BUTTON 3: RF TRAINING (FULL WIDTH) =====
                let rf_training_button = egui::Button::new(
                    egui::RichText::new("RF TRAINING")
                        .size(12.0)
                        .family(egui::FontFamily::Name("PoppinsBold".into()))
                        .color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(34, 139, 34))
                .min_size(egui::vec2(uploadwidth, 30.0));

                if ui.add(rf_training_button).clicked() {
                    println!("🌲 RF TRAINING GUI clicked!");

                    std::thread::spawn(|| {
                        println!("🔧 Checking for latest train_rf_gui binary...");

                        let possible_paths = vec![
                            "./target/release/train_rf_gui.exe",
                            "./target/release/train_rf_gui",
                            "target/release/train_rf_gui.exe",
                            "target/release/train_rf_gui",
                            "./target/debug/train_rf_gui.exe",
                            "./target/debug/train_rf_gui",
                            "target/debug/train_rf_gui.exe",
                            "target/debug/train_rf_gui",
                        ];

                        let mut binary_path = None;
                        let mut binary_timestamp = None;

                        for path in &possible_paths {
                            if let Ok(metadata) = std::fs::metadata(path) {
                                if let Ok(modified) = metadata.modified() {
                                    if binary_timestamp.is_none() || Some(modified) > binary_timestamp {
                                        binary_path = Some(path.to_string());
                                        binary_timestamp = Some(modified);
                                        println!("📁 Found binary: {} (modified: {:?})", path, modified);
                                    }
                                }
                            }
                        }

                        let binary_path = match binary_path {
                            Some(path) => {
                                println!("[OK] Using newest binary: {}", path);
                                path
                            }
                            None => {
                                eprintln!("[X] Error: train_rf_gui binary not found!");
                                eprintln!("\n📋 Build instructions:");
                                eprintln!("   cargo build --bin train_rf_gui --release");
                                return;
                            }
                        };

                        println!("🚀 Launching RF Training GUI...");
                        match Command::new(&binary_path).spawn() {
                            Ok(mut child) => {
                                println!("[OK] RF Training GUI launched successfully!");
                                println!("   Binary: {}", binary_path);
                                println!("   PID: {:?}", child.id());

                                std::thread::spawn(move || {
                                    match child.wait() {
                                        Ok(status) => println!("📊 RF Training GUI closed (status: {})", status),
                                        Err(e) => eprintln!("[X] Error waiting for RF GUI: {}", e),
                                    }
                                });
                            }
                            Err(e) => {
                                eprintln!("[X] Failed to launch RF Training GUI!");
                                eprintln!("   Binary: {}", binary_path);
                                eprintln!("   Error: {}", e);
                                eprintln!("\n💡 Rebuild: cargo build --bin train_rf_gui --release");
                            }
                        }
                    });
                }

                ui.add_space(3.0);

                // ===== BUTTON 4: SVM TRAINING (FULL WIDTH) =====
                let svm_training_button = egui::Button::new(
                    egui::RichText::new("SVM TRAINING")
                        .size(12.0)
                        .family(egui::FontFamily::Name("PoppinsBold".into()))
                        .color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(0, 90, 160))
                .min_size(egui::vec2(uploadwidth, 30.0));

                if ui.add(svm_training_button).clicked() {
                    println!("⚡ SVM TRAINING GUI clicked!");

                    std::thread::spawn(|| {
                        println!("🔧 Checking for latest train_svm_gui binary...");

                        let possible_paths = vec![
                            "./target/release/train_svm_gui.exe",
                            "./target/release/train_svm_gui",
                            "target/release/train_svm_gui.exe",
                            "target/release/train_svm_gui",
                            "./target/debug/train_svm_gui.exe",
                            "./target/debug/train_svm_gui",
                            "target/debug/train_svm_gui.exe",
                            "target/debug/train_svm_gui",
                        ];

                        let mut binary_path = None;
                        let mut binary_timestamp = None;

                        for path in &possible_paths {
                            if let Ok(metadata) = std::fs::metadata(path) {
                                if let Ok(modified) = metadata.modified() {
                                    if binary_timestamp.is_none() || Some(modified) > binary_timestamp {
                                        binary_path = Some(path.to_string());
                                        binary_timestamp = Some(modified);
                                        println!("📁 Found binary: {} (modified: {:?})", path, modified);
                                    }
                                }
                            }
                        }

                        let binary_path = match binary_path {
                            Some(path) => {
                                println!("[OK] Using newest binary: {}", path);
                                path
                            }
                            None => {
                                eprintln!("[X] Error: train_svm_gui binary not found!");
                                eprintln!("\n📋 Build instructions:");
                                eprintln!("   cargo build --bin train_svm_gui --release");
                                return;
                            }
                        };

                        println!("🚀 Launching SVM Training GUI...");
                        match Command::new(&binary_path).spawn() {
                            Ok(mut child) => {
                                println!("[OK] SVM Training GUI launched successfully!");
                                println!("   Binary: {}", binary_path);
                                println!("   PID: {:?}", child.id());

                                std::thread::spawn(move || {
                                    match child.wait() {
                                        Ok(status) => println!("📊 SVM Training GUI closed (status: {})", status),
                                        Err(e) => eprintln!("[X] Error waiting for SVM GUI: {}", e),
                                    }
                                });
                            }
                            Err(e) => {
                                eprintln!("[X] Failed to launch SVM Training GUI!");
                                eprintln!("   Binary: {}", binary_path);
                                eprintln!("   Error: {}", e);
                                eprintln!("\n💡 Rebuild: cargo build --bin train_svm_gui --release");
                            }
                        }
                    });
                }

                ui.add_space(3.0);

                // ===== BUTTON 5: LSTM TRAINING (FULL WIDTH) =====
                let lstm_training_button = egui::Button::new(
                    egui::RichText::new("LSTM TRAINING")
                        .size(12.0)
                        .family(egui::FontFamily::Name("PoppinsBold".into()))
                        .color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(100, 20, 180))
                .min_size(egui::vec2(uploadwidth, 30.0));

                if ui.add(lstm_training_button).clicked() {
                    println!("🔁 LSTM TRAINING GUI clicked!");

                    std::thread::spawn(|| {
                        println!("🔧 Checking for latest train_lstm_gui binary...");

                        let possible_paths = vec![
                            "./target/release/train_lstm_gui.exe",
                            "./target/release/train_lstm_gui",
                            "target/release/train_lstm_gui.exe",
                            "target/release/train_lstm_gui",
                            "./target/debug/train_lstm_gui.exe",
                            "./target/debug/train_lstm_gui",
                            "target/debug/train_lstm_gui.exe",
                            "target/debug/train_lstm_gui",
                        ];

                        let mut binary_path = None;
                        let mut binary_timestamp = None;

                        for path in &possible_paths {
                            if let Ok(metadata) = std::fs::metadata(path) {
                                if let Ok(modified) = metadata.modified() {
                                    if binary_timestamp.is_none() || Some(modified) > binary_timestamp {
                                        binary_path = Some(path.to_string());
                                        binary_timestamp = Some(modified);
                                        println!("📁 Found binary: {} (modified: {:?})", path, modified);
                                    }
                                }
                            }
                        }

                        let binary_path = match binary_path {
                            Some(path) => {
                                println!("[OK] Using newest binary: {}", path);
                                path
                            }
                            None => {
                                eprintln!("[X] Error: train_lstm_gui binary not found!");
                                eprintln!("\n📋 Build instructions:");
                                eprintln!("   cargo build --bin train_lstm_gui --release");
                                return;
                            }
                        };

                        println!("🚀 Launching LSTM Training GUI...");
                        match Command::new(&binary_path).spawn() {
                            Ok(mut child) => {
                                println!("[OK] LSTM Training GUI launched successfully!");
                                println!("   Binary: {}", binary_path);
                                println!("   PID: {:?}", child.id());

                                std::thread::spawn(move || {
                                    match child.wait() {
                                        Ok(status) => println!("📊 LSTM Training GUI closed (status: {})", status),
                                        Err(e) => eprintln!("[X] Error waiting for LSTM GUI: {}", e),
                                    }
                                });
                            }
                            Err(e) => {
                                eprintln!("[X] Failed to launch LSTM Training GUI!");
                                eprintln!("   Binary: {}", binary_path);
                                eprintln!("   Error: {}", e);
                                eprintln!("\n💡 Rebuild: cargo build --bin train_lstm_gui --release");
                            }
                        }
                    });
                }

                ui.add_space(3.0);

                // ===== BUTTON 6: MLP TRAINING (FULL WIDTH) =====
                let mlp_training_button = egui::Button::new(
                    egui::RichText::new("MLP TRAINING")
                        .size(12.0)
                        .family(egui::FontFamily::Name("PoppinsBold".into()))
                        .color(egui::Color32::WHITE),
                )
                .fill(egui::Color32::from_rgb(0, 130, 110))
                .min_size(egui::vec2(uploadwidth, 30.0));

                if ui.add(mlp_training_button).clicked() {
                    println!("🧠 MLP TRAINING GUI clicked!");

                    std::thread::spawn(|| {
                        println!("🔧 Checking for latest train_mlp_gui binary...");

                        let possible_paths = vec![
                            "./target/release/train_mlp_gui.exe",
                            "./target/release/train_mlp_gui",
                            "target/release/train_mlp_gui.exe",
                            "target/release/train_mlp_gui",
                            "./target/debug/train_mlp_gui.exe",
                            "./target/debug/train_mlp_gui",
                            "target/debug/train_mlp_gui.exe",
                            "target/debug/train_mlp_gui",
                        ];

                        let mut binary_path = None;
                        let mut binary_timestamp = None;

                        for path in &possible_paths {
                            if let Ok(metadata) = std::fs::metadata(path) {
                                if let Ok(modified) = metadata.modified() {
                                    if binary_timestamp.is_none() || Some(modified) > binary_timestamp {
                                        binary_path = Some(path.to_string());
                                        binary_timestamp = Some(modified);
                                        println!("📁 Found binary: {} (modified: {:?})", path, modified);
                                    }
                                }
                            }
                        }

                        let binary_path = match binary_path {
                            Some(path) => {
                                println!("[OK] Using newest binary: {}", path);
                                path
                            }
                            None => {
                                eprintln!("[X] Error: train_mlp_gui binary not found!");
                                eprintln!("\n📋 Build instructions:");
                                eprintln!("   cargo build --bin train_mlp_gui --release");
                                return;
                            }
                        };

                        println!("🚀 Launching MLP Training GUI...");
                        match Command::new(&binary_path).spawn() {
                            Ok(mut child) => {
                                println!("[OK] MLP Training GUI launched successfully!");
                                println!("   Binary: {}", binary_path);
                                println!("   PID: {:?}", child.id());
                                std::thread::spawn(move || {
                                    match child.wait() {
                                        Ok(status) => println!("📊 MLP Training GUI closed (status: {})", status),
                                        Err(e) => eprintln!("[X] Error waiting for MLP GUI: {}", e),
                                    }
                                });
                            }
                            Err(e) => {
                                eprintln!("[X] Failed to launch MLP Training GUI!");
                                eprintln!("   Binary: {}", binary_path);
                                eprintln!("   Error: {}", e);
                                eprintln!("\n💡 Rebuild: cargo build --bin train_mlp_gui --release");
                            }
                        }
                    });
                }

                ui.add_space(3.0);

                // ===== BUTTON 7 & 8: EXPORT dan RESULT (Side by Side) =====
                ui.horizontal(|ui| {
                    ui.add_space(2.0);
                    let buttonspacing = 8.0;
                    let buttonwidth = (uploadwidth - buttonspacing) / 2.0;

                    // EXPORT BUTTON (Red)
                    let trainingbutton = egui::Button::new(
                        egui::RichText::new("EXPORT")
                            .size(14.0)
                            .family(egui::FontFamily::Name("PoppinsBold".into()))
                            .color(egui::Color32::WHITE),
                    )
                    .fill(egui::Color32::from_rgb(220, 20, 60))
                    .min_size(egui::vec2(buttonwidth, 36.0));

                    if ui.add(trainingbutton).clicked() {
                        println!("📤 EXPORT clicked - Select database file...");
                        
                        let dbstatus_clone = Arc::clone(dbstatus);
                        
                        std::thread::spawn(move || {
                            if let Some(dbpath) = rfd::FileDialog::new()
                                .add_filter("Database", &["db"])
                                .set_title("Select Database File to Export")
                                .pick_file()
                            {
                                println!("Selected database: {:?}", dbpath);
                                
                                // Export database to CSV
                                match export_database_to_csv(&dbpath) {
                                    Ok(csvpath) => {
                                        println!("[OK] Exported to: {:?}", csvpath);
                                        if let Ok(mut status) = dbstatus_clone.lock() {
                                            *status = format!("Exported to CSV: {}", csvpath.display());
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("[X] Export failed: {}", e);
                                        if let Ok(mut status) = dbstatus_clone.lock() {
                                            *status = format!("Export error: {}", e);
                                        }
                                    }
                                }
                            } else {
                                println!("No file selected");
                            }
                        });
                    }

                    ui.add_space(buttonspacing);

                    // RESULT BUTTON (Blue)
                    let resultbutton = egui::Button::new(
                        egui::RichText::new("RESULT")
                            .size(14.0)
                            .strong()
                            .color(egui::Color32::WHITE),
                    )
                    .fill(egui::Color32::from_rgb(70, 130, 180))
                    .min_size(egui::vec2(buttonwidth, 36.0));

                    if ui.add(resultbutton).clicked() {
                        println!("📊 RESULT clicked - Opening result window...");
                        
                        std::thread::spawn(|| {
                            use std::process::Command;
                            
                            let display = std::env::var("DISPLAY").unwrap_or("0".to_string());
                            let binary_path = "./target/release/predict_gui";
                            
                            if let Err(e) = Command::new(binary_path)
                                .env("DISPLAY", display)
                                .env("LIBGL_ALWAYS_SOFTWARE", "1")
                                .env("GDK_BACKEND", "x11")
                                .spawn()
                            {
                                eprintln!("[X] Failed to launch result GUI: {}", e);
                            }
                        });
                    }
                });
            });
        },
    );

    ui.allocate_space(egui::vec2(panelwidth, panelheight));
}


fn draw_status_indicator(ui: &mut egui::Ui, is_on: bool) {
    ui.horizontal(|ui| {
        let on_color = if is_on {
            egui::Color32::from_rgb(180, 180, 180)
        } else {
            egui::Color32::WHITE
        };

        ui.painter().rect_filled(
            egui::Rect::from_min_size(ui.cursor().min, egui::vec2(35.0, 24.0)),
            egui::Rounding::same(2.0),
            on_color,
        );
        ui.painter().rect_stroke(
            egui::Rect::from_min_size(ui.cursor().min, egui::vec2(35.0, 24.0)),
            egui::Rounding::same(2.0),
            egui::Stroke::new(if is_on { 2.0 } else { 1.0 }, egui::Color32::DARK_GRAY),
        );
        ui.painter().text(
            egui::pos2(ui.cursor().min.x + 17.5, ui.cursor().min.y + 12.0),
            egui::Align2::CENTER_CENTER,
            "ON",
            egui::FontId::new(11.0, egui::FontFamily::Name("Poppins".into())),
            egui::Color32::BLACK,
        );
        ui.add_space(35.0);

        let off_color = if !is_on {
            egui::Color32::from_rgb(180, 180, 180)
        } else {
            egui::Color32::WHITE
        };

        ui.painter().rect_filled(
            egui::Rect::from_min_size(ui.cursor().min, egui::vec2(35.0, 24.0)),
            egui::Rounding::same(2.0),
            off_color,
        );
        ui.painter().rect_stroke(
            egui::Rect::from_min_size(ui.cursor().min, egui::vec2(35.0, 24.0)),
            egui::Rounding::same(2.0),
            egui::Stroke::new(if !is_on { 2.0 } else { 1.0 }, egui::Color32::DARK_GRAY),
        );
        ui.painter().text(
            egui::pos2(ui.cursor().min.x + 17.5, ui.cursor().min.y + 12.0),
            egui::Align2::CENTER_CENTER,
            "OFF",
            egui::FontId::new(11.0, egui::FontFamily::Name("Poppins".into())),
            egui::Color32::BLACK,
        );
        ui.add_space(35.0);
    });
}

fn draw_adc_panel(ui: &mut egui::Ui, data: &SensorData) {
    let panel_width = 780.0;
    let panel_height = 170.0;
    let panel_rect = egui::Rect::from_min_size(ui.cursor().min, egui::vec2(panel_width, panel_height));

    ui.painter().rect_stroke(
        panel_rect,
        egui::Rounding::ZERO,
        egui::Stroke::new(2.0, egui::Color32::BLACK),
    );

    let header_color = egui::Color32::from_rgb(139, 119, 53);
    let header_rect = egui::Rect::from_min_size(panel_rect.min, egui::vec2(panel_width, 32.0));
    ui.painter().rect_filled(header_rect, egui::Rounding::ZERO, header_color);

    let content_background_rect = egui::Rect::from_min_size(
        egui::pos2(panel_rect.min.x + 2.0, panel_rect.min.y + 32.0),
        egui::vec2(panel_width - 4.0, panel_height - 34.0),
    );
    ui.painter().rect_filled(content_background_rect, egui::Rounding::ZERO, egui::Color32::WHITE);

    ui.painter().text(
        egui::pos2(header_rect.center().x, header_rect.center().y),
        egui::Align2::CENTER_CENTER,
        "MILIVOLT VALUE",
        egui::FontId::new(18.0, egui::FontFamily::Name("PoppinsBold".into())),
        egui::Color32::WHITE,
    );

    ui.allocate_ui_at_rect(
        egui::Rect::from_min_size(
            egui::pos2(panel_rect.min.x + 50.0, panel_rect.min.y + 40.0),
            egui::vec2(panel_width - 100.0, panel_height - 60.0),
        ),
        |ui| {
            ui.vertical(|ui| {
                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    ui.add_space(20.0);

                    ui.vertical(|ui| {
                        draw_legend_line(ui, "MQ - 3", egui::Color32::from_rgb(255, 102, 51));
                        ui.add_space(5.0);
                        ui.horizontal(|ui| {
                            ui.add_space(25.0);
                            ui.label(
                                egui::RichText::new(format!("{:.0}", data.mq3))
                                    .family(egui::FontFamily::Name("Poppins".into()))
                                    .color(egui::Color32::BLACK),
                            );
                        });
                    });

                    ui.add_space(110.0);

                    ui.vertical(|ui| {
                        draw_legend_line(ui, "MQ - 6", egui::Color32::BLACK);
                        ui.add_space(5.0);
                        ui.horizontal(|ui| {
                            ui.add_space(25.0);
                            ui.label(
                                egui::RichText::new(format!("{:.0}", data.mq6))
                                    .family(egui::FontFamily::Name("Poppins".into()))
                                    .color(egui::Color32::BLACK),
                            );
                        });
                    });

                    ui.add_space(110.0);

                    ui.vertical(|ui| {
                        draw_legend_line(ui, "MQ - 7", egui::Color32::from_rgb(51, 102, 255));
                        ui.add_space(5.0);
                        ui.horizontal(|ui| {
                            ui.add_space(25.0);
                            ui.label(
                                egui::RichText::new(format!("{:.0}", data.mq7))
                                    .family(egui::FontFamily::Name("Poppins".into()))
                                    .color(egui::Color32::BLACK),
                            );
                        });
                    });

                    ui.add_space(110.0);

                    ui.vertical(|ui| {
                        draw_legend_line(ui, "MQ - 135", egui::Color32::from_rgb(153, 51, 255));
                        ui.add_space(5.0);
                        ui.horizontal(|ui| {
                            ui.add_space(25.0);
                            ui.label(
                                egui::RichText::new(format!("{:.0}", data.mq135))
                                    .family(egui::FontFamily::Name("Poppins".into()))
                                    .color(egui::Color32::BLACK),
                            );
                        });
                    });
                });

                ui.add_space(20.0);

                ui.horizontal(|ui| {
                    ui.add_space(20.0);

                    ui.vertical(|ui| {
                        draw_legend_line(ui, "TGS2600", egui::Color32::from_rgb(102, 102, 51));
                        ui.add_space(5.0);
                        ui.horizontal(|ui| {
                            ui.add_space(30.0);
                            ui.label(
                                egui::RichText::new(format!("{:.0}", data.tgs2600))
                                    .family(egui::FontFamily::Name("Poppins".into()))
                                    .color(egui::Color32::BLACK),
                            );
                        });
                    });

                    ui.add_space(90.0);

                    ui.vertical(|ui| {
                        draw_legend_line(ui, "TGS2602", egui::Color32::from_rgb(51, 153, 51));
                        ui.add_space(5.0);
                        ui.horizontal(|ui| {
                            ui.add_space(30.0);
                            ui.label(
                                egui::RichText::new(format!("{:.0}", data.tgs2602))
                                    .family(egui::FontFamily::Name("Poppins".into()))
                                    .color(egui::Color32::BLACK),
                            );
                        });
                    });

                    ui.add_space(90.0);

                    ui.vertical(|ui| {
                        draw_legend_line(ui, "TGS2611", egui::Color32::from_rgb(51, 204, 255));
                        ui.add_space(5.0);
                        ui.horizontal(|ui| {
                            ui.add_space(30.0);
                            ui.label(
                                egui::RichText::new(format!("{:.0}", data.tgs2611))
                                    .family(egui::FontFamily::Name("Poppins".into()))
                                    .color(egui::Color32::BLACK),
                            );
                        });
                    });

                    ui.add_space(90.0);

                    ui.vertical(|ui| {
                        draw_legend_line(ui, "TGS2620", egui::Color32::from_rgb(102, 102, 102));
                        ui.add_space(5.0);
                        ui.horizontal(|ui| {
                            ui.add_space(30.0);
                            ui.label(
                                egui::RichText::new(format!("{:.0}", data.tgs2620))
                                    .family(egui::FontFamily::Name("Poppins".into()))
                                    .color(egui::Color32::BLACK),
                            );
                        });
                    });
                });
            });
        },
    );

    ui.allocate_space(egui::vec2(panel_width, panel_height));
}

fn draw_legend_line(ui: &mut egui::Ui, text: &str, color: egui::Color32) {
    ui.horizontal(|ui| {
        ui.painter().line_segment(
            [
                egui::pos2(ui.cursor().min.x, ui.cursor().min.y + 7.0),
                egui::pos2(ui.cursor().min.x + 20.0, ui.cursor().min.y + 7.0),
            ],
            egui::Stroke::new(3.0, color),
        );
        ui.add_space(25.0);
        ui.label(
            egui::RichText::new(text)
                .size(12.0)
                .family(egui::FontFamily::Name("PoppinsBold".into()))
                .strong(),
        );
    });
}

fn draw_graph_panel(ui: &mut egui::Ui, history: &SensorHistory) {
    let available_rect = ui.available_rect_before_wrap();
    
    ui.painter().rect_stroke(
        available_rect,
        egui::Rounding::ZERO,
        egui::Stroke::new(2.0, egui::Color32::BLACK),
    );

    ui.painter().rect_filled(
        egui::Rect::from_min_size(
            egui::pos2(available_rect.min.x + 2.0, available_rect.min.y + 2.0),
            egui::vec2(available_rect.width() - 4.0, available_rect.height() - 4.0),
        ),
        egui::Rounding::ZERO,
        egui::Color32::WHITE,
    );

    ui.allocate_ui_at_rect(
        egui::Rect::from_min_size(
            egui::pos2(available_rect.min.x + 8.0, available_rect.min.y + 8.0),
            egui::vec2(available_rect.width() - 16.0, available_rect.height() - 16.0),
        ),
        |ui| {
            Plot::new("sensor_plot")
                .width(ui.available_width())
                .height(ui.available_height())
                .show_axes([true, true])
                .show_grid(true)
                .x_axis_formatter(|mark, _range| format!("{:.0}s", mark.value))
                .show(ui, |plot_ui| {
                    if !history.timestamps.is_empty() {
                        let mq3_points: PlotPoints = history
                            .timestamps
                            .iter()
                            .zip(&history.mq3_data)
                            .map(|(t, v)| [*t, *v])
                            .collect();

                        let mq6_points: PlotPoints = history
                            .timestamps
                            .iter()
                            .zip(&history.mq6_data)
                            .map(|(t, v)| [*t, *v])
                            .collect();

                        let mq7_points: PlotPoints = history
                            .timestamps
                            .iter()
                            .zip(&history.mq7_data)
                            .map(|(t, v)| [*t, *v])
                            .collect();

                        let mq135_points: PlotPoints = history
                            .timestamps
                            .iter()
                            .zip(&history.mq135_data)
                            .map(|(t, v)| [*t, *v])
                            .collect();

                        let tgs2600_points: PlotPoints = history
                            .timestamps
                            .iter()
                            .zip(&history.tgs2600_data)
                            .map(|(t, v)| [*t, *v])
                            .collect();

                        let tgs2602_points: PlotPoints = history
                            .timestamps
                            .iter()
                            .zip(&history.tgs2602_data)
                            .map(|(t, v)| [*t, *v])
                            .collect();

                        let tgs2611_points: PlotPoints = history
                            .timestamps
                            .iter()
                            .zip(&history.tgs2611_data)
                            .map(|(t, v)| [*t, *v])
                            .collect();

                        let tgs2620_points: PlotPoints = history
                            .timestamps
                            .iter()
                            .zip(&history.tgs2620_data)
                            .map(|(t, v)| [*t, *v])
                            .collect();

                        plot_ui.line(Line::new(mq3_points).color(egui::Color32::from_rgb(255, 102, 51)).name("MQ-3"));
                        plot_ui.line(Line::new(mq6_points).color(egui::Color32::BLACK).name("MQ-6"));
                        plot_ui.line(Line::new(mq7_points).color(egui::Color32::from_rgb(51, 102, 255)).name("MQ-7"));
                        plot_ui.line(Line::new(mq135_points).color(egui::Color32::from_rgb(153, 51, 255)).name("MQ-135"));
                        plot_ui.line(Line::new(tgs2600_points).color(egui::Color32::from_rgb(102, 102, 51)).name("TGS2600"));
                        plot_ui.line(Line::new(tgs2602_points).color(egui::Color32::from_rgb(51, 153, 51)).name("TGS2602"));
                        plot_ui.line(Line::new(tgs2611_points).color(egui::Color32::from_rgb(51, 204, 255)).name("TGS2611"));
                        plot_ui.line(Line::new(tgs2620_points).color(egui::Color32::from_rgb(102, 102, 102)).name("TGS2620"));
                    }
                });
        },
    );
}

fn draw_sample_panel(
    ui: &mut egui::Ui,
    sample_name: &mut String,
    command_sender: &Sender<String>,
    system_running: &Arc<Mutex<bool>>,
    data_counter: &Arc<Mutex<u32>>,
    sensor_history: &Arc<Mutex<SensorHistory>>,
    button_disabled: bool,
) {
    let panel_width = 200.0;
    let panel_height = 170.0;
    let panel_rect = egui::Rect::from_min_size(ui.cursor().min, egui::vec2(panel_width, panel_height));

    ui.painter().rect_stroke(
        panel_rect,
        egui::Rounding::ZERO,
        egui::Stroke::new(2.0, egui::Color32::BLACK),
    );

    let header_color = egui::Color32::from_rgb(31, 78, 121);
    let header_rect = egui::Rect::from_min_size(panel_rect.min, egui::vec2(panel_width, 32.0));
    ui.painter().rect_filled(header_rect, egui::Rounding::ZERO, header_color);

    ui.painter().text(
        egui::pos2(header_rect.center().x, header_rect.center().y),
        egui::Align2::CENTER_CENTER,
        "ARABICA SAMPLE",
        egui::FontId::new(16.0, egui::FontFamily::Name("PoppinsBold".into())),
        egui::Color32::WHITE,
    );

    let content_background_rect = egui::Rect::from_min_size(
        egui::pos2(panel_rect.min.x + 2.0, panel_rect.min.y + 32.0),
        egui::vec2(panel_width - 4.0, panel_height - 34.0),
    );
    ui.painter().rect_filled(content_background_rect, egui::Rounding::ZERO, egui::Color32::WHITE);

    let content_padding = 10.0;
    ui.allocate_ui_at_rect(
        egui::Rect::from_min_size(
            egui::pos2(panel_rect.min.x + content_padding, panel_rect.min.y + 32.0 + content_padding),
            egui::vec2(panel_width - content_padding * 2.0, panel_height - 32.0 - content_padding * 2.0),
        ),
        |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(3.0);

                let inner_width = panel_width - content_padding * 2.0;
                let textarea_width = inner_width - 4.0;
                let textarea_height = 68.0;

                ui.horizontal(|ui| {
                    ui.add_space(2.0);
                    ui.vertical(|ui| {
                        let text_edit_rect = egui::Rect::from_min_size(
                            ui.cursor().min,
                            egui::vec2(textarea_width, textarea_height),
                        );

                        ui.painter().rect_stroke(
                            text_edit_rect,
                            egui::Rounding::same(4.0),
                            egui::Stroke::new(1.5, egui::Color32::from_gray(150)),
                        );

                        let text_edit = egui::TextEdit::multiline(sample_name)
                            .font(egui::FontId::new(15.0, egui::FontFamily::Name("Poppins".into())))
                            .text_color(egui::Color32::BLACK)
                            .desired_width(textarea_width - 8.0)
                            .desired_rows(3)
                            .frame(false);

                        ui.add(text_edit);
                    });
                });

                ui.add_space(5.0);

                ui.horizontal(|ui| {
                    ui.add_space(2.0);

                    let button_spacing = 8.0;
                    let button_width = (textarea_width - button_spacing) / 2.0;

                    let start_button = egui::Button::new(
                        egui::RichText::new("START")
                            .size(14.0)
                            .family(egui::FontFamily::Name("PoppinsBold".into()))
                            .color(egui::Color32::WHITE),
                    )
                    .fill(if button_disabled {
                        egui::Color32::from_rgb(100, 100, 100)
                    } else {
                        egui::Color32::from_rgb(34, 139, 34)
                    })
                    .min_size(egui::vec2(button_width, 36.0));

                    let start_response = ui.add_enabled(!button_disabled, start_button);

                    if start_response.clicked() {
                        println!("▶️ START button clicked!");

                        if let Ok(mut running) = system_running.lock() {
                            *running = true;
                            println!("[OK] system_running = TRUE");
                        }

                        if let Ok(mut counter) = data_counter.lock() {
                            *counter = 0;
                            println!("🔄 data_counter = 0");
                        }

                        if let Ok(mut history) = sensor_history.lock() {
                            history.reset();
                            println!("🧹 sensor_history reset");
                        }

                        match command_sender.send("START".to_string()) {
                            Ok(_) => {
                                println!("📤 START command sent to Arduino");
                                println!("⏳ Waiting for first data (elapsed_time>=1)...");
                            }
                            Err(e) => {
                                eprintln!("[X] Failed to send START: {}", e);
                                if let Ok(mut running) = system_running.lock() {
                                    *running = false;
                                }
                            }
                        }
                    }

                    ui.add_space(button_spacing);

                    let stop_button = egui::Button::new(
                        egui::RichText::new("STOP")
                            .size(14.0)
                            .family(egui::FontFamily::Name("PoppinsBold".into()))
                            .color(egui::Color32::WHITE),
                    )
                    .fill(if button_disabled {
                        egui::Color32::from_rgb(180, 0, 0)
                    } else {
                        egui::Color32::from_rgb(120, 120, 120)
                    })
                    .min_size(egui::vec2(button_width, 36.0));

                    if ui.add(stop_button).clicked() && button_disabled {
                        println!("⏹️ STOP!");

                        if let Ok(mut running) = system_running.lock() {
                            *running = false;
                        }

                        if let Ok(mut counter) = data_counter.lock() {
                            *counter = 0;
                        }

                        match command_sender.send("STOP".to_string()) {
                            Ok(_) => println!("[OK] STOP sent - Graph preserved"),
                            Err(e) => eprintln!("[X] Failed: {}", e),
                        }
                    }
                });
            });
        },
    );

    ui.allocate_space(egui::vec2(panel_width, panel_height));
}

// ═══════════════════════════════════════════════════════
// DATABASE TO CSV EXPORT
// ═══════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════
// DATABASE TO CSV EXPORT
// ═══════════════════════════════════════════════════════

fn export_database_to_csv(db_path: &PathBuf) -> Result<PathBuf, String> {
    // Open database
    let db = DatabaseManager::new(db_path.to_str().unwrap())
        .map_err(|e| format!("Failed to open database: {}", e))?;
    
    // Get all readings
    let readings = db.get_all_readings()
        .map_err(|e| format!("Failed to read data: {}", e))?;
    
    if readings.is_empty() {
        return Err("Database is empty!".to_string());
    }
    
    println!("📊 Found {} readings in database", readings.len());
    
    // Create CSV path (same folder as .db file)
    let csv_path = db_path.with_extension("csv");
    
    // Write CSV
    let mut writer = csv::Writer::from_path(&csv_path)
        .map_err(|e| format!("Failed to create CSV: {}", e))?;
    
    // Write header
    writer.write_record(&[
        "time_seconds",
        "tgs2600",
        "mq135",
        "mq3",
        "mq6",
        "mq7",
        "tgs2602",
        "tgs2611",
        "tgs2620",
    ]).map_err(|e| format!("Failed to write header: {}", e))?;
    
    // [OK] FIX: Iterate dengan reference (&readings)
    for reading in &readings {
        writer.write_record(&[
            reading.time_seconds.to_string(),
            reading.tgs2600.to_string(),
            reading.mq135.to_string(),
            reading.mq3.to_string(),
            reading.mq6.to_string(),
            reading.mq7.to_string(),
            reading.tgs2602.to_string(),
            reading.tgs2611.to_string(),
            reading.tgs2620.to_string(),
        ]).map_err(|e| format!("Failed to write row: {}", e))?;
    }
    
    writer.flush()
        .map_err(|e| format!("Failed to flush CSV: {}", e))?;
    
    // [OK] Sekarang readings masih bisa dipakai
    println!("[OK] CSV exported: {} rows written", readings.len());
    Ok(csv_path)
}

// ═══════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════

fn main() -> eframe::Result<()> {
    println!("🚀 Starting Electronic Nose with ML Integration...");
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([1200.0, 700.0])
            .with_title("Electronic Nose for Classification Arabica Coffee Quality")
            .with_maximized(true),
        ..Default::default()
    };

    eframe::run_native(
        "Electronic Nose Coffee Quality",
        options,
        Box::new(|cc| {
            setup_custom_fonts(&cc.egui_ctx);
            Ok(Box::new(CoffeeQualityApp::new()))
        }),
    )
}
