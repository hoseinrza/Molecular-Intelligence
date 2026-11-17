//! The benchmark result report: JSON for tooling, Markdown for humans.

use crate::json::Json;

/// One training epoch's traced numbers.
#[derive(Debug, Clone)]
pub struct EpochTrace {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Mean masked-MSE training loss (standardized target space).
    pub train_mean_loss: f32,
    /// Pooled validation MAE in eV (checkpoint-selection metric).
    pub validation_mae_ev: f64,
    /// Whether this epoch held the best validation MAE so far.
    pub is_best: bool,
}

/// Test metrics for one target slot.
#[derive(Debug, Clone)]
pub struct SlotMetrics {
    /// Slot name (`homo`, `lumo`, `gap`).
    pub name: String,
    /// Number of scored observations.
    pub n: usize,
    /// Mean absolute error (eV).
    pub mae_ev: f64,
    /// Root-mean-square error (eV).
    pub rmse_ev: f64,
    /// Coefficient of determination; `None` when the references have zero
    /// variance (R² is undefined, never fudged).
    pub r2: Option<f64>,
}

/// The complete, self-describing benchmark report.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// SDF records read (before exclusions).
    pub records: usize,
    /// Records excluded via `uncharacterized.txt`.
    pub excluded_uncharacterized: usize,
    /// Records skipped by the loader, with reasons carried separately.
    pub skipped: usize,
    /// Title/position disagreements counted by the loader.
    pub title_mismatches: usize,
    /// Entries that entered the benchmark.
    pub loaded_entries: usize,
    /// Optional pilot subset applied before splitting.
    pub subset: Option<usize>,
    /// The split seed.
    pub split_seed: u64,
    /// Training subset size.
    pub train_size: usize,
    /// Validation subset size.
    pub validation_size: usize,
    /// Test subset size.
    pub test_size: usize,
    /// Hidden layer widths in order.
    pub hidden: Vec<usize>,
    /// Trained epochs.
    pub epochs: usize,
    /// Base learning rate.
    pub lr: f32,
    /// Glorot initialization seed.
    pub init_seed: u64,
    /// The optimizer and its hyperparameters.
    pub optimizer: String,
    /// The learning-rate schedule.
    pub schedule: String,
    /// Seconds spent loading and parsing.
    pub load_seconds: f64,
    /// Seconds spent featurizing and standardizing.
    pub featurize_seconds: f64,
    /// Seconds spent training.
    pub train_seconds: f64,
    /// Seconds spent on the final test evaluation.
    pub evaluate_seconds: f64,
    /// The per-epoch trace.
    pub trace: Vec<EpochTrace>,
    /// The checkpoint-selected epoch (best validation MAE).
    pub best_epoch: usize,
    /// Per-slot test metrics, in `[homo, lumo, gap]` order.
    pub slots: Vec<SlotMetrics>,
    /// Pooled test MAE (eV).
    pub pooled_mae_ev: f64,
    /// Pooled test RMSE (eV).
    pub pooled_rmse_ev: f64,
}

impl BenchmarkReport {
    /// The JSON representation of the report.
    pub fn to_json(&self) -> Json {
        let slot_json = |slot: &SlotMetrics| {
            Json::Object(vec![
                ("name".into(), Json::String(slot.name.clone())),
                ("n".into(), Json::Number(slot.n as f64)),
                ("mae_ev".into(), Json::Number(slot.mae_ev)),
                ("rmse_ev".into(), Json::Number(slot.rmse_ev)),
                ("r2".into(), slot.r2.map(Json::Number).unwrap_or(Json::Null)),
            ])
        };
        Json::Object(vec![
            ("benchmark".into(), Json::String("qm9-baseline-v1".into())),
            (
                "descriptor_layout".into(),
                Json::String("v1: 18 slots".into()),
            ),
            (
                "target_layout".into(),
                Json::String("[homo, lumo, gap] eV".into()),
            ),
            (
                "dataset".into(),
                Json::Object(vec![
                    ("records".into(), Json::Number(self.records as f64)),
                    (
                        "excluded_uncharacterized".into(),
                        Json::Number(self.excluded_uncharacterized as f64),
                    ),
                    ("skipped".into(), Json::Number(self.skipped as f64)),
                    (
                        "title_mismatches".into(),
                        Json::Number(self.title_mismatches as f64),
                    ),
                    (
                        "loaded_entries".into(),
                        Json::Number(self.loaded_entries as f64),
                    ),
                    (
                        "subset".into(),
                        self.subset
                            .map(|n| Json::Number(n as f64))
                            .unwrap_or(Json::Null),
                    ),
                ]),
            ),
            (
                "split".into(),
                Json::Object(vec![
                    ("seed".into(), Json::Number(self.split_seed as f64)),
                    ("fractions".into(), Json::String("80/10/10".into())),
                    ("train".into(), Json::Number(self.train_size as f64)),
                    (
                        "validation".into(),
                        Json::Number(self.validation_size as f64),
                    ),
                    ("test".into(), Json::Number(self.test_size as f64)),
                ]),
            ),
            (
                "hyperparameters".into(),
                Json::Object(vec![
                    (
                        "hidden".into(),
                        Json::Array(
                            self.hidden
                                .iter()
                                .map(|width| Json::Number(*width as f64))
                                .collect(),
                        ),
                    ),
                    ("epochs".into(), Json::Number(self.epochs as f64)),
                    ("lr".into(), Json::Number(f64::from(self.lr))),
                    ("init_seed".into(), Json::Number(self.init_seed as f64)),
                    ("batch_size".into(), Json::Number(1.0)),
                    ("optimizer".into(), Json::String(self.optimizer.clone())),
                    ("schedule".into(), Json::String(self.schedule.clone())),
                ]),
            ),
            (
                "timing_seconds".into(),
                Json::Object(vec![
                    ("load".into(), Json::Number(self.load_seconds)),
                    ("featurize".into(), Json::Number(self.featurize_seconds)),
                    ("train".into(), Json::Number(self.train_seconds)),
                    ("evaluate".into(), Json::Number(self.evaluate_seconds)),
                ]),
            ),
            (
                "trace".into(),
                Json::Array(
                    self.trace
                        .iter()
                        .map(|entry| {
                            Json::Object(vec![
                                ("epoch".into(), Json::Number(entry.epoch as f64)),
                                (
                                    "train_mean_loss".into(),
                                    Json::Number(f64::from(entry.train_mean_loss)),
                                ),
                                (
                                    "validation_mae_ev".into(),
                                    Json::Number(entry.validation_mae_ev),
                                ),
                                ("is_best".into(), Json::Bool(entry.is_best)),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("best_epoch".into(), Json::Number(self.best_epoch as f64)),
            (
                "test_slots".into(),
                Json::Array(self.slots.iter().map(slot_json).collect()),
            ),
            (
                "test_pooled".into(),
                Json::Object(vec![
                    (
                        "n".into(),
                        Json::Number(self.slots.iter().map(|slot| slot.n).sum::<usize>() as f64),
                    ),
                    ("mae_ev".into(), Json::Number(self.pooled_mae_ev)),
                    ("rmse_ev".into(), Json::Number(self.pooled_rmse_ev)),
                ]),
            ),
        ])
    }

    /// The human-readable Markdown summary.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# QM9 baseline benchmark (Experiment 01)\n\n");
        out.push_str("Descriptor-based MLP, per-sample SGD on the Burn Flex CPU backend.\n\n");
        out.push_str("## Dataset\n\n");
        out.push_str(&format!(
            "- SDF records: {} (excluded uncharacterized: {}, skipped: {}, title mismatches: {})\n",
            self.records, self.excluded_uncharacterized, self.skipped, self.title_mismatches
        ));
        out.push_str(&format!("- Loaded entries: {}\n", self.loaded_entries));
        if let Some(subset) = self.subset {
            out.push_str(&format!("- Pilot subset: first {subset} entries\n"));
        }
        out.push_str("\n## Split and hyperparameters\n\n");
        out.push_str(&format!(
            "- Split: 80/10/10, seed {} (train {}, validation {}, test {})\n",
            self.split_seed, self.train_size, self.validation_size, self.test_size
        ));
        out.push_str(&format!(
            "- Network: 18 → {:?} → 3; {} epochs; base lr {}; init seed {}; batch size 1\n",
            self.hidden, self.epochs, self.lr, self.init_seed
        ));
        out.push_str(&format!(
            "- Optimizer: {} ; schedule: {}\n",
            self.optimizer, self.schedule
        ));
        out.push_str(
            "- Preprocessing: z-score inputs and targets over the training subset only\n\n",
        );
        out.push_str("## Timing\n\n");
        out.push_str(&format!(
            "| load | featurize | train | evaluate |\n|---|---|---|---|\n| {:.1}s | {:.1}s | {:.1}s | {:.1}s |\n\n",
            self.load_seconds, self.featurize_seconds, self.train_seconds, self.evaluate_seconds
        ));
        out.push_str("## Training trace\n\n");
        out.push_str("| epoch | train loss | val MAE (eV) | best |\n|---|---|---|---|\n");
        for entry in &self.trace {
            out.push_str(&format!(
                "| {} | {:.6} | {:.4} | {} |\n",
                entry.epoch,
                entry.train_mean_loss,
                entry.validation_mae_ev,
                if entry.is_best { "✓" } else { "" }
            ));
        }
        out.push_str(&format!(
            "\nCheckpoint-selected epoch: {}\n\n",
            self.best_epoch
        ));
        out.push_str("## Test metrics (eV)\n\n");
        out.push_str("| slot | n | MAE | RMSE | R² |\n|---|---|---|---|---|\n");
        for slot in &self.slots {
            out.push_str(&format!(
                "| {} | {} | {:.4} | {:.4} | {} |\n",
                slot.name,
                slot.n,
                slot.mae_ev,
                slot.rmse_ev,
                slot.r2
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "undefined".into()),
            ));
        }
        out.push_str(&format!(
            "\nPooled: n = {}, MAE = {:.4} eV, RMSE = {:.4} eV\n",
            self.slots.iter().map(|slot| slot.n).sum::<usize>(),
            self.pooled_mae_ev,
            self.pooled_rmse_ev
        ));
        out
    }
}
