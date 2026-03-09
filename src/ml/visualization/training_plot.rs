// src/ml/visualization/training_plot.rs
// Training Metrics Visualization - FIXED VERSION

use plotters::prelude::*;
use anyhow::Result;

use crate::ml::training::TrainingMetrics;

pub struct TrainingVisualizer {
    output_dir: String,
}

impl TrainingVisualizer {
    pub fn new(output_dir: &str) -> Self {
        std::fs::create_dir_all(output_dir).ok();
        Self {
            output_dir: output_dir.to_string(),
        }
    }

    /// Generate all training plots
    pub fn plot_all(&self, metrics: &[TrainingMetrics]) -> Result<()> {
        println!("\n📊 Generating training visualizations...");
        
        self.plot_loss(metrics)?;
        self.plot_accuracy(metrics)?;
        self.plot_combined(metrics)?;
        self.plot_loss_diff(metrics)?;
        
        println!("✅ All plots saved to: {}", self.output_dir);
        Ok(())
    }

    /// Plot training and validation loss
    pub fn plot_loss(&self, metrics: &[TrainingMetrics]) -> Result<()> {
        let path = format!("{}/loss_curve.png", self.output_dir);
        let root = BitMapBackend::new(&path, (1200, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let epochs: Vec<usize> = (1..=metrics.len()).collect();
        let train_loss: Vec<f32> = metrics.iter().map(|m| m.train_loss).collect();
        let val_loss: Vec<f32> = metrics.iter().map(|m| m.val_loss).collect();

        let max_loss = train_loss.iter()
            .chain(val_loss.iter())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_loss = train_loss.iter()
            .chain(val_loss.iter())
            .cloned()
            .fold(f32::INFINITY, f32::min);

        let mut chart = ChartBuilder::on(&root)
            .caption("Training and Validation Loss", ("sans-serif", 50).into_font())
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(
                1usize..metrics.len(),
                (min_loss * 0.9)..(max_loss * 1.1)
            )?;

        chart
            .configure_mesh()
            .x_desc("Epoch")
            .y_desc("Loss")
            .x_label_style(("sans-serif", 20))
            .y_label_style(("sans-serif", 20))
            .draw()?;

        // Train loss line
        chart
            .draw_series(LineSeries::new(
                epochs.iter().zip(train_loss.iter()).map(|(e, l)| (*e, *l)),
                &RED.mix(0.8),
            ))?
            .label("Train Loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        // Val loss line
        chart
            .draw_series(LineSeries::new(
                epochs.iter().zip(val_loss.iter()).map(|(e, l)| (*e, *l)),
                &BLUE.mix(0.8),
            ))?
            .label("Val Loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        // Add points
        chart.draw_series(
            epochs.iter()
                .zip(train_loss.iter())
                .map(|(e, l)| Circle::new((*e, *l), 3, RED.filled()))
        )?;

        chart.draw_series(
            epochs.iter()
                .zip(val_loss.iter())
                .map(|(e, l)| Circle::new((*e, *l), 3, BLUE.filled()))
        )?;

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .label_font(("sans-serif", 25))
            .draw()?;

        root.present()?;
        println!("  ✓ Loss curve: {}", path);
        Ok(())
    }

    /// Plot training and validation accuracy
    pub fn plot_accuracy(&self, metrics: &[TrainingMetrics]) -> Result<()> {
        let path = format!("{}/accuracy_curve.png", self.output_dir);
        let root = BitMapBackend::new(&path, (1200, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let epochs: Vec<usize> = (1..=metrics.len()).collect();
        let train_acc: Vec<f32> = metrics.iter().map(|m| m.train_accuracy * 100.0).collect();
        let val_acc: Vec<f32> = metrics.iter().map(|m| m.val_accuracy * 100.0).collect();

        let max_acc = train_acc.iter()
            .chain(val_acc.iter())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_acc = train_acc.iter()
            .chain(val_acc.iter())
            .cloned()
            .fold(f32::INFINITY, f32::min);

        let mut chart = ChartBuilder::on(&root)
            .caption("Training and Validation Accuracy", ("sans-serif", 50).into_font())
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(
                1usize..metrics.len(),
                (min_acc - 5.0).max(0.0)..(max_acc + 5.0).min(100.0)
            )?;

        chart
            .configure_mesh()
            .x_desc("Epoch")
            .y_desc("Accuracy (%)")
            .x_label_style(("sans-serif", 20))
            .y_label_style(("sans-serif", 20))
            .draw()?;

        // Train accuracy line
        chart
            .draw_series(LineSeries::new(
                epochs.iter().zip(train_acc.iter()).map(|(e, a)| (*e, *a)),
                &GREEN.mix(0.8),
            ))?
            .label("Train Accuracy")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        // Val accuracy line
        chart
            .draw_series(LineSeries::new(
                epochs.iter().zip(val_acc.iter()).map(|(e, a)| (*e, *a)),
                &CYAN.mix(0.8),
            ))?
            .label("Val Accuracy")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &CYAN));

        // Add points
        chart.draw_series(
            epochs.iter()
                .zip(train_acc.iter())
                .map(|(e, a)| Circle::new((*e, *a), 3, GREEN.filled()))
        )?;

        chart.draw_series(
            epochs.iter()
                .zip(val_acc.iter())
                .map(|(e, a)| Circle::new((*e, *a), 3, CYAN.filled()))
        )?;

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .label_font(("sans-serif", 25))
            .draw()?;

        root.present()?;
        println!("  ✓ Accuracy curve: {}", path);
        Ok(())
    }

    /// Plot combined loss and accuracy (2x2 grid)
    pub fn plot_combined(&self, metrics: &[TrainingMetrics]) -> Result<()> {
        let path = format!("{}/training_summary.png", self.output_dir);
        let root = BitMapBackend::new(&path, (1600, 1200)).into_drawing_area();
        root.fill(&WHITE)?;

        let areas = root.split_evenly((2, 2));

        // Top-left: Loss comparison
        self.draw_subplot_loss(&areas[0], metrics, "Loss Comparison")?;

        // Top-right: Accuracy comparison
        self.draw_subplot_accuracy(&areas[1], metrics, "Accuracy Comparison")?;

        // Bottom-left: Overfitting indicator (loss gap)
        self.draw_overfitting_indicator(&areas[2], metrics)?;

        // Bottom-right: Final metrics summary
        self.draw_metrics_summary(&areas[3], metrics)?;

        root.present()?;
        println!("  ✓ Combined summary: {}", path);
        Ok(())
    }

    fn draw_subplot_loss(
        &self,
        area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        metrics: &[TrainingMetrics],
        title: &str,
    ) -> Result<()> {
        let epochs: Vec<usize> = (1..=metrics.len()).collect();
        let train_loss: Vec<f32> = metrics.iter().map(|m| m.train_loss).collect();
        let val_loss: Vec<f32> = metrics.iter().map(|m| m.val_loss).collect();

        let max_loss = train_loss.iter()
            .chain(val_loss.iter())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(1usize..metrics.len(), 0.0f32..max_loss * 1.1)?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(
            epochs.iter().zip(train_loss.iter()).map(|(e, l)| (*e, *l)),
            &RED,
        ))?;

        chart.draw_series(LineSeries::new(
            epochs.iter().zip(val_loss.iter()).map(|(e, l)| (*e, *l)),
            &BLUE,
        ))?;

        Ok(())
    }

    fn draw_subplot_accuracy(
        &self,
        area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        metrics: &[TrainingMetrics],
        title: &str,
    ) -> Result<()> {
        let epochs: Vec<usize> = (1..=metrics.len()).collect();
        let train_acc: Vec<f32> = metrics.iter().map(|m| m.train_accuracy * 100.0).collect();
        let val_acc: Vec<f32> = metrics.iter().map(|m| m.val_accuracy * 100.0).collect();

        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(1usize..metrics.len(), 0.0f32..100.0f32)?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(
            epochs.iter().zip(train_acc.iter()).map(|(e, a)| (*e, *a)),
            &GREEN,
        ))?;

        chart.draw_series(LineSeries::new(
            epochs.iter().zip(val_acc.iter()).map(|(e, a)| (*e, *a)),
            &CYAN,
        ))?;

        Ok(())
    }

    fn draw_overfitting_indicator(
        &self,
        area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        metrics: &[TrainingMetrics],
    ) -> Result<()> {
        let epochs: Vec<usize> = (1..=metrics.len()).collect();
        let loss_gap: Vec<f32> = metrics.iter()
            .map(|m| (m.val_loss - m.train_loss).abs())
            .collect();

        let max_gap = loss_gap.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut chart = ChartBuilder::on(area)
            .caption("Overfitting Indicator (Val-Train Loss Gap)", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(1usize..metrics.len(), 0.0f32..max_gap * 1.2)?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(
            epochs.iter().zip(loss_gap.iter()).map(|(e, g)| (*e, *g)),
            &MAGENTA,
        ))?;

        // Draw warning threshold
        let threshold = 0.1f32;
        chart.draw_series(LineSeries::new(
            vec![(1, threshold), (metrics.len(), threshold)],
            ShapeStyle::from(&RED).stroke_width(2),
        ))?;

        Ok(())
    }

    fn draw_metrics_summary(
        &self,
        area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        metrics: &[TrainingMetrics],
    ) -> Result<()> {
        area.fill(&WHITE)?;

        let last = metrics.last().unwrap();
        let best_val_acc = metrics.iter()
            .map(|m| m.val_accuracy)
            .fold(f32::NEG_INFINITY, f32::max);

        let text = format!(
            "Final Metrics\n\n\
             Train Loss: {:.4}\n\
             Val Loss: {:.4}\n\n\
             Train Acc: {:.2}%\n\
             Val Acc: {:.2}%\n\n\
             Best Val Acc: {:.2}%\n\
             Total Epochs: {}",
            last.train_loss,
            last.val_loss,
            last.train_accuracy * 100.0,
            last.val_accuracy * 100.0,
            best_val_acc * 100.0,
            metrics.len()
        );

        area.draw_text(
            &text,
            &TextStyle::from(("sans-serif", 25).into_font()).color(&BLACK),
            (50, 50),
        )?;

        Ok(())
    }

    /// Plot loss difference (overfitting detection)
    pub fn plot_loss_diff(&self, metrics: &[TrainingMetrics]) -> Result<()> {
        let path = format!("{}/overfitting_analysis.png", self.output_dir);
        let root = BitMapBackend::new(&path, (1200, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let epochs: Vec<usize> = (1..=metrics.len()).collect();
        let loss_diff: Vec<f32> = metrics.iter()
            .map(|m| m.val_loss - m.train_loss)
            .collect();

        let max_diff = loss_diff.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_diff = loss_diff.iter().cloned().fold(f32::INFINITY, f32::min);

        let mut chart = ChartBuilder::on(&root)
            .caption("Overfitting Analysis (Val Loss - Train Loss)", ("sans-serif", 50).into_font())
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(
                1usize..metrics.len(),
                (min_diff - 0.05)..(max_diff + 0.05)
            )?;

        chart
            .configure_mesh()
            .x_desc("Epoch")
            .y_desc("Loss Difference")
            .x_label_style(("sans-serif", 20))
            .y_label_style(("sans-serif", 20))
            .draw()?;

        // Zero line
        chart.draw_series(LineSeries::new(
            vec![(1, 0.0), (metrics.len(), 0.0)],
            &BLACK.mix(0.3),
        ))?;

        // Loss difference line
        chart.draw_series(LineSeries::new(
            epochs.iter().zip(loss_diff.iter()).map(|(e, d)| (*e, *d)),
            &MAGENTA.mix(0.8),
        ))?;

        // Fill area
        chart.draw_series(
            AreaSeries::new(
                epochs.iter().zip(loss_diff.iter()).map(|(e, d)| (*e, *d)),
                0.0,
                &MAGENTA.mix(0.2),
            )
        )?;

        root.present()?;
        println!("  ✓ Overfitting analysis: {}", path);
        Ok(())
    }
}

// ================================================
// USAGE IN TRAINING
// ================================================

pub fn visualize_training(metrics: &[TrainingMetrics], output_dir: &str) -> Result<()> {
    let visualizer = TrainingVisualizer::new(output_dir);
    visualizer.plot_all(metrics)?;
    Ok(())
}