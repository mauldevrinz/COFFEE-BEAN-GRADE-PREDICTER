//! Machine Learning Module
//! CNN + Random Forest coffee quality classifier

pub mod models;
pub use models::cnn;
pub mod data_loader;
pub mod preprocessing;
pub mod training;
pub mod evaluation;
pub mod validation;
pub mod predict;
pub mod analysis;
pub mod model;
pub mod visualization;

// Re-export model types
pub use models::CoffeeCNN;
pub use models::{CoffeeRandomForest, RandomForestConfig, RFTrainingMetrics, extract_features};
pub use models::{CoffeeSVM, SVMConfig, SVMTrainingMetrics, extract_features_svm};
pub use models::{CoffeeLSTM, LSTMConfig, LSTMTrainingMetrics};
pub use models::{CoffeeMLP, MLPConfig, MLPTrainingMetrics, extract_features_mlp};

// Re-export commonly used items
pub use data_loader::{CoffeeDataLoader, NormalizationStats, CoffeeDataset, DataLoaderConfig};
pub use preprocessing::{DataPreprocessor, PreprocessingConfig, MissingStrategy};
pub use training::{Trainer, TrainingConfig, TrainingMetrics};
pub use evaluation::Evaluator;
