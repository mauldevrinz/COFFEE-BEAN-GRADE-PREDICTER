// src/model.rs - Simple model configuration

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub num_classes: usize,
    pub input_channels: usize,
    pub input_length: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            num_classes: 2,
            input_channels: 8,
            input_length: 300,
        }
    }
}
