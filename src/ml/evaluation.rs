// src/evaluation.rs - Evaluation Metrics

use colored::*;
use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub support: usize,
}

#[derive(Debug, Clone)]
pub struct EvaluationResults {
    pub accuracy: f64,
    pub high_grade_metrics: ClassMetrics,
    pub low_grade_metrics: ClassMetrics,
    pub confusion_matrix: [[usize; 2]; 2],
}

pub struct Evaluator;

impl Evaluator {
    pub fn evaluate(predictions: &Array1<i64>, labels: &Array1<i64>) -> EvaluationResults {
        assert_eq!(predictions.len(), labels.len(), "Predictions and labels must have same length");
        
        let mut confusion_matrix = [[0; 2]; 2];
        
        for (pred, true_label) in predictions.iter().zip(labels.iter()) {
            let pred_idx = *pred as usize;
            let true_idx = *true_label as usize;
            confusion_matrix[true_idx][pred_idx] += 1;
        }
        
        let high_tp = confusion_matrix[0][0];
        let high_fp = confusion_matrix[1][0];
        let high_fn = confusion_matrix[0][1];
        
        let high_precision = if high_tp + high_fp > 0 {
            high_tp as f64 / (high_tp + high_fp) as f64
        } else {
            0.0
        };
        
        let high_recall = if high_tp + high_fn > 0 {
            high_tp as f64 / (high_tp + high_fn) as f64
        } else {
            0.0
        };
        
        let high_f1 = if high_precision + high_recall > 0.0 {
            2.0 * (high_precision * high_recall) / (high_precision + high_recall)
        } else {
            0.0
        };
        
        let high_support = high_tp + high_fn;
        
        let low_tp = confusion_matrix[1][1];
        let low_fp = confusion_matrix[0][1];
        let low_fn = confusion_matrix[1][0];
        
        let low_precision = if low_tp + low_fp > 0 {
            low_tp as f64 / (low_tp + low_fp) as f64
        } else {
            0.0
        };
        
        let low_recall = if low_tp + low_fn > 0 {
            low_tp as f64 / (low_tp + low_fn) as f64
        } else {
            0.0
        };
        
        let low_f1 = if low_precision + low_recall > 0.0 {
            2.0 * (low_precision * low_recall) / (low_precision + low_recall)
        } else {
            0.0
        };
        
        let low_support = low_tp + low_fn;
        
        let correct = high_tp + low_tp;
        let total = labels.len();
        let accuracy = correct as f64 / total as f64;
        
        EvaluationResults {
            accuracy,
            high_grade_metrics: ClassMetrics {
                precision: high_precision,
                recall: high_recall,
                f1_score: high_f1,
                support: high_support,
            },
            low_grade_metrics: ClassMetrics {
                precision: low_precision,
                recall: low_recall,
                f1_score: low_f1,
                support: low_support,
            },
            confusion_matrix,
        }
    }
    
    pub fn print_results(results: &EvaluationResults, dataset_name: &str) {
        println!("\n{}", format!("═══════════════════════════════════════════════════").cyan().bold());
        println!("{}", format!("  {} Set Evaluation Results", dataset_name).cyan().bold());
        println!("{}", format!("═══════════════════════════════════════════════════").cyan().bold());
        
        println!("\n{}", "📊 Overall Metrics:".bold());
        println!("┌─────────────────────┬──────────────┐");
        println!("│ Metric              │ Value        │");
        println!("├─────────────────────┼──────────────┤");
        println!("│ Overall Accuracy    │ {:>11.2}% │", results.accuracy * 100.0);
        println!("└─────────────────────┴──────────────┘");
        
        println!("\n{}", "📈 Per-Class Metrics:".bold());
        println!("┌─────────────────────┬──────────────┬──────────────┐");
        println!("│ Metric              │ High Grade   │ Low Grade    │");
        println!("├─────────────────────┼──────────────┼──────────────┤");
        println!("│ Precision           │ {:>12.3} │ {:>12.3} │", 
            results.high_grade_metrics.precision,
            results.low_grade_metrics.precision
        );
        println!("│ Recall              │ {:>12.3} │ {:>12.3} │",
            results.high_grade_metrics.recall,
            results.low_grade_metrics.recall
        );
        println!("│ F1-Score            │ {:>12.3} │ {:>12.3} │",
            results.high_grade_metrics.f1_score,
            results.low_grade_metrics.f1_score
        );
        println!("│ Support             │ {:>12} │ {:>12} │",
            results.high_grade_metrics.support,
            results.low_grade_metrics.support
        );
        println!("└─────────────────────┴──────────────┴──────────────┘");
        
        println!("\n{}", "🎯 Confusion Matrix:".bold());
        println!("┌────────────────┬──────────────┬──────────────┐");
        println!("│                │ Pred: High   │ Pred: Low    │");
        println!("├────────────────┼──────────────┼──────────────┤");
        println!("│ True: High     │ {:>12} │ {:>12} │",
            results.confusion_matrix[0][0],
            results.confusion_matrix[0][1]
        );
        println!("│ True: Low      │ {:>12} │ {:>12} │",
            results.confusion_matrix[1][0],
            results.confusion_matrix[1][1]
        );
        println!("└────────────────┴──────────────┴──────────────┘");
        
        Self::print_performance_analysis(results);
    }
    
    fn print_performance_analysis(results: &EvaluationResults) {
        println!("\n{}", "💡 Performance Analysis:".bold());
        
        let mut messages = Vec::new();
        
        if results.accuracy < 0.7 {
            messages.push("⚠️  Low overall accuracy (<70%). Consider improving model.".yellow().to_string());
        } else if results.accuracy > 0.95 {
            messages.push("⚠️  Very high accuracy (>95%). Check for overfitting.".yellow().to_string());
        } else {
            messages.push("✅ Overall accuracy is in good range (70-95%)".green().to_string());
        }
        
        let precision_diff = (results.high_grade_metrics.precision - results.low_grade_metrics.precision).abs();
        
        if precision_diff > 0.2 {
            messages.push(format!("⚠️  Large precision difference ({:.1}%) between classes", precision_diff * 100.0).yellow().to_string());
        }
        
        if results.high_grade_metrics.recall < 0.7 {
            messages.push("⚠️  Low High Grade recall. Many High samples misclassified.".yellow().to_string());
        }
        
        if results.low_grade_metrics.recall < 0.7 {
            messages.push("⚠️  Low Low Grade recall. Many Low samples misclassified.".yellow().to_string());
        }
        
        for msg in messages {
            println!("  {}", msg);
        }
        
        println!("\n{}", "💡 Recommendations:".bold());
        if results.accuracy >= 0.85 {
            println!("  ✅ Model performance is good!");
            println!("  • Consider deploying to production");
            println!("  • Monitor performance on new data");
        } else if results.accuracy >= 0.70 {
            println!("  🔄 Model performance is acceptable but could improve:");
            println!("  • Try data augmentation");
            println!("  • Tune hyperparameters");
            println!("  • Collect more diverse samples");
        } else {
            println!("  ⚠️  Model needs significant improvement:");
            println!("  • Review data quality");
            println!("  • Try different model architecture");
            println!("  • Increase dataset size");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_perfect_prediction() {
        let predictions = array![0, 0, 1, 1, 0, 1];
        let labels = array![0, 0, 1, 1, 0, 1];
        
        let results = Evaluator::evaluate(&predictions, &labels);
        
        assert_eq!(results.accuracy, 1.0);
        assert_eq!(results.high_grade_metrics.precision, 1.0);
        assert_eq!(results.high_grade_metrics.recall, 1.0);
    }
}