//! The QM9 baseline benchmark (Phase 8, Experiment 01).
//!
//! Executes the documented procedure from
//! `docs/architecture/baseline-model.md`: load `qm9.sdf` through `mi-io`,
//! split 80/10/10 under a seed, featurize descriptors and targets, fit
//! z-score scalers **on the training subset only**, train the MLP with
//! per-sample SGD, select the checkpoint by validation MAE, and report
//! MAE/RMSE/R² per target slot on the test subset — everything in
//! electronvolts, everything reproducible from the recorded seed and
//! hyperparameters.

mod json;
mod report;
mod uncharacterized;

use mi_data::{MolecularDataset, SplitFractions, random_split};
use mi_evaluation::{mae, r2, rmse};
use mi_featurization::{
    ColumnScalers, DESCRIPTOR_WIDTH, StandardScaler, Targets, descriptor_vector,
};
use mi_io::{Qm9Options, load_qm9_sdf};
use mi_models::{Mlp, MlpConfig, RegressionSample, predict};
use mi_tensor::{Cpu, Tensor, TensorShape};
use mi_training::{Adam, LrSchedule, Optimizer, Sgd, TrainConfig, train};
use report::{BenchmarkReport, EpochTrace, SlotMetrics};
use std::collections::BTreeSet;
use std::time::Instant;

/// Reference values for one evaluation entry: the raw descriptor and the
/// present targets in eV.
struct EvaluationEntry {
    descriptor: Vec<f32>,
    references: [Option<f32>; Targets::WIDTH],
}

/// Parsed command-line configuration.
struct Args {
    dataset: String,
    uncharacterized: String,
    out_dir: String,
    epochs: usize,
    lr: f32,
    hidden: Vec<usize>,
    seed: u64,
    subset: Option<usize>,
    validate_every: usize,
    batch_size: usize,
    optimizer: OptimizerChoice,
    schedule: ScheduleChoice,
}

/// Which update rule the run uses.
#[derive(Clone, Copy)]
enum OptimizerChoice {
    /// Plain SGD or SGD with classic momentum.
    Sgd { momentum: f32 },
    /// Adam with standard betas and epsilon.
    Adam,
}

/// How the learning rate evolves.
#[derive(Clone, Copy)]
enum ScheduleChoice {
    /// Constant at the base rate.
    Constant,
    /// Exponential decay with the given factor.
    Exponential { decay: f32 },
    /// Cosine annealing to zero across all epochs.
    Cosine,
}

impl Args {
    fn parse(argv: &[String]) -> Result<Self, String> {
        let mut dataset = "datasets/qm9.sdf".into();
        let mut uncharacterized = "datasets/uncharacterized.txt".into();
        let mut out_dir = "benchmarks/qm9-baseline".into();
        let mut epochs = 30usize;
        let mut lr = 0.01f32;
        let mut hidden = vec![64usize, 32];
        let mut seed = 42u64;
        let mut subset = None;
        let mut validate_every = 1usize;
        let mut batch_size = 1usize;
        let mut optimizer = OptimizerChoice::Sgd { momentum: 0.0 };
        let mut schedule = ScheduleChoice::Constant;
        let mut momentum = 0.0f32;
        let mut decay = 0.9f32;
        let mut position = 0usize;
        while position < argv.len() {
            let flag = argv[position].as_str();
            let value = |position: &mut usize| -> Result<String, String> {
                *position += 1;
                argv.get(*position)
                    .cloned()
                    .ok_or(format!("{flag} needs a value"))
            };
            match flag {
                "--dataset" => dataset = value(&mut position)?,
                "--uncharacterized" => uncharacterized = value(&mut position)?,
                "--out" => out_dir = value(&mut position)?,
                "--epochs" => {
                    epochs = value(&mut position)?
                        .parse()
                        .map_err(|error| format!("--epochs: {error}"))?
                }
                "--lr" => {
                    lr = value(&mut position)?
                        .parse()
                        .map_err(|error| format!("--lr: {error}"))?
                }
                "--hidden" => {
                    let widths = value(&mut position)?;
                    hidden = widths
                        .split(',')
                        .map(|width| width.trim().parse::<usize>())
                        .collect::<Result<_, _>>()
                        .map_err(|error| format!("--hidden: {error}"))?;
                }
                "--seed" => {
                    seed = value(&mut position)?
                        .parse()
                        .map_err(|error| format!("--seed: {error}"))?
                }
                "--subset" => {
                    subset = Some(
                        value(&mut position)?
                            .parse()
                            .map_err(|error| format!("--subset: {error}"))?,
                    )
                }
                "--validate-every" => {
                    validate_every = value(&mut position)?
                        .parse()
                        .map_err(|error| format!("--validate-every: {error}"))?
                }
                "--batch-size" => {
                    batch_size = value(&mut position)?
                        .parse()
                        .map_err(|error| format!("--batch-size: {error}"))?;
                    if batch_size == 0 {
                        return Err("--batch-size must be positive".into());
                    }
                }
                "--optimizer" => {
                    optimizer = match value(&mut position)?.as_str() {
                        "sgd" => OptimizerChoice::Sgd { momentum },
                        "adam" => OptimizerChoice::Adam,
                        other => return Err(format!("unknown optimizer {other:?} (sgd | adam)")),
                    }
                }
                "--momentum" => {
                    momentum = value(&mut position)?
                        .parse()
                        .map_err(|error| format!("--momentum: {error}"))?;
                    if let OptimizerChoice::Sgd { momentum: slot } = &mut optimizer {
                        *slot = momentum;
                    }
                }
                "--schedule" => {
                    schedule = match value(&mut position)?.as_str() {
                        "constant" => ScheduleChoice::Constant,
                        "exponential" => ScheduleChoice::Exponential { decay },
                        "cosine" => ScheduleChoice::Cosine,
                        other => {
                            return Err(format!(
                                "unknown schedule {other:?} (constant | exponential | cosine)"
                            ));
                        }
                    }
                }
                "--decay" => {
                    decay = value(&mut position)?
                        .parse()
                        .map_err(|error| format!("--decay: {error}"))?;
                    if let ScheduleChoice::Exponential { decay: slot } = &mut schedule {
                        *slot = decay;
                    }
                }
                other => return Err(format!("unknown flag {other:?}")),
            }
            position += 1;
        }
        if epochs == 0 {
            return Err("--epochs must be positive".into());
        }
        if !lr.is_finite() || lr <= 0.0 {
            return Err("--lr must be finite and positive".into());
        }
        Ok(Self {
            dataset,
            uncharacterized,
            out_dir,
            epochs,
            lr,
            hidden,
            seed,
            subset,
            validate_every,
            batch_size,
            optimizer,
            schedule,
        })
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match Args::parse(&argv) {
        Ok(args) => args,
        Err(error) => {
            eprintln!(
                "error: {error}\n\nusage: qm9-baseline [--dataset PATH] [--uncharacterized PATH] [--out DIR] [--epochs N] [--lr F] [--hidden W,W,...] [--seed N] [--subset N] [--validate-every N] [--batch-size N] [--optimizer sgd|adam] [--momentum F] [--schedule constant|exponential|cosine] [--decay F]"
            );
            std::process::exit(2);
        }
    };
    if let Err(error) = run(args) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    let total_start = Instant::now();

    // 1. Load: exclusions first, then the streaming SDF load.
    let exclusions: BTreeSet<usize> =
        uncharacterized::parse_uncharacterized(&args.uncharacterized)?;
    println!(
        "uncharacterized exclusions: {} indices from {}",
        exclusions.len(),
        args.uncharacterized
    );
    let load_start = Instant::now();
    let loaded = load_qm9_sdf(&args.dataset, &Qm9Options::new(), &exclusions)
        .map_err(|error| format!("loading {}: {error}", args.dataset))?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let mut dataset = loaded.dataset().clone();
    println!(
        "loaded: {} records read, {} excluded, {} skipped, {} title mismatches, {} entries kept ({:.1}s)",
        loaded.record_count(),
        loaded.excluded_uncharacterized(),
        loaded.skipped().len(),
        loaded.title_mismatches(),
        dataset.len(),
        load_seconds
    );
    if let Some(subset) = args.subset {
        let mut trimmed = MolecularDataset::new().with_name("qm9-pilot");
        for entry in dataset.iter().take(subset) {
            trimmed.push(entry.clone());
        }
        dataset = trimmed;
        println!("pilot subset: first {subset} entries");
    }
    if dataset.len() < 10 {
        return Err(format!(
            "only {} entries available — refusing to benchmark a dataset this small",
            dataset.len()
        ));
    }

    // 2. Split: canonical 80/10/10 under the recorded seed.
    let split = random_split(&dataset, SplitFractions::DEFAULT, args.seed)
        .map_err(|error| format!("splitting: {error}"))?;
    println!(
        "split: train {} / validation {} / test {}",
        split.train().len(),
        split.validation().len(),
        split.test().len()
    );

    // 3. Prepare: descriptors, targets, and training-subset-only scalers.
    let featurize_start = Instant::now();
    let training_device = Cpu::training();
    let inference_device = Cpu::inference();
    let train_entries: Vec<_> = split.train().iter().cloned().collect();
    let validation_entries: Vec<EvaluationEntry> =
        split.validation().iter().map(entry_to_evaluation).collect();
    let test_entries: Vec<EvaluationEntry> = split.test().iter().map(entry_to_evaluation).collect();

    // Raw training descriptors (host) and per-slot present targets (eV).
    let train_descriptors: Vec<Vec<f32>> = train_entries
        .iter()
        .map(|entry| {
            let tensor = descriptor_vector(entry.molecule(), &inference_device)
                .map_err(|error| format!("descriptors: {error}"))?;
            tensor
                .to_vec()
                .map_err(|error| format!("descriptor readback: {error}"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut target_columns = vec![Vec::new(); Targets::WIDTH];
    for entry in &train_entries {
        let targets = Targets::extract(entry);
        for (slot, column) in target_columns.iter_mut().enumerate() {
            if let Some(value) = targets.present_value(slot) {
                column.push(value);
            }
        }
    }
    let input_scalers = ColumnScalers::fit_rows(&train_descriptors)
        .map_err(|error| format!("input scalers: {error}"))?;
    let target_scalers = target_columns
        .iter()
        .map(|column| StandardScaler::fit(column))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("target scalers: {error}"))?;

    // Training samples on the training device: standardized input rows and
    // standardized values with the raw presence mask, all `[1, width]`.
    let input_shape = TensorShape::new(vec![1, DESCRIPTOR_WIDTH])
        .map_err(|error| format!("input shape: {error}"))?;
    let row_shape = TensorShape::new(vec![1, Targets::WIDTH])
        .map_err(|error| format!("target shape: {error}"))?;
    let samples: Vec<RegressionSample> = train_entries
        .iter()
        .zip(&train_descriptors)
        .map(|(entry, descriptor)| -> Result<RegressionSample, String> {
            let scaled = input_scalers
                .scale_row(descriptor)
                .map_err(|error| format!("input scalers: {error}"))?;
            let input = Tensor::<2>::from_data(scaled, input_shape.clone(), &training_device)
                .map_err(|error| format!("training input: {error}"))?;
            let targets = Targets::extract(entry);
            let mut values = Vec::with_capacity(Targets::WIDTH);
            let mut mask = Vec::with_capacity(Targets::WIDTH);
            for (slot, scaler) in target_scalers.iter().enumerate() {
                match targets.present_value(slot) {
                    Some(value) => {
                        values.push(scaler.scale(value));
                        mask.push(1.0);
                    }
                    None => {
                        values.push(0.0);
                        mask.push(0.0);
                    }
                }
            }
            let mask_sum: f32 = mask.iter().sum();
            let values = Tensor::<2>::from_data(values, row_shape.clone(), &training_device)
                .map_err(|error| format!("target tensor: {error}"))?;
            let mask = Tensor::<2>::from_data(mask, row_shape.clone(), &training_device)
                .map_err(|error| format!("mask tensor: {error}"))?;
            Ok(RegressionSample {
                input,
                values,
                mask,
                mask_sum,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let featurize_seconds = featurize_start.elapsed().as_secs_f64();
    println!(
        "featurized: {} training samples, {} validation, {} test ({:.1}s)",
        samples.len(),
        validation_entries.len(),
        test_entries.len(),
        featurize_seconds
    );

    // 4. Train: per-sample updates, checkpoint selection on validation MAE.
    let config = MlpConfig {
        input_dim: DESCRIPTOR_WIDTH,
        hidden: args.hidden.clone(),
        output_dim: Targets::WIDTH,
    };
    let mut optimizer: Box<dyn Optimizer> = match args.optimizer {
        OptimizerChoice::Sgd { momentum } => Sgd::new(momentum)
            .map(Box::new)
            .map_err(|error| format!("optimizer: {error}"))?,
        OptimizerChoice::Adam => Adam::new()
            .map(Box::new)
            .map_err(|error| format!("optimizer: {error}"))?,
    };
    let schedule = match args.schedule {
        ScheduleChoice::Constant => LrSchedule::Constant,
        ScheduleChoice::Exponential { decay } => LrSchedule::Exponential { decay },
        ScheduleChoice::Cosine => LrSchedule::Cosine {
            total_epochs: args.epochs,
        },
    };
    let mut model = Mlp::new_with_seed(config.clone(), &training_device, args.seed)
        .map_err(|error| format!("model: {error}"))?;
    let train_start = Instant::now();
    let mut trace = Vec::with_capacity(args.epochs);
    let mut best_validation_mae = f64::INFINITY;
    let mut best_epoch = 0usize;
    let mut best_parameters = model
        .parameter_values()
        .map_err(|error| format!("snapshot: {error}"))?;
    for epoch in 0..args.epochs {
        let lr = schedule
            .lr(epoch, args.lr)
            .map_err(|error| format!("schedule at epoch {epoch}: {error}"))?;
        let history = train(
            &mut model,
            &samples,
            optimizer.as_mut(),
            &TrainConfig {
                epochs: 1,
                base_lr: lr,
                schedule: LrSchedule::Constant,
                batch_size: args.batch_size,
            },
            &training_device,
        )
        .map_err(|error| format!("training epoch {epoch}: {error}"))?;
        let train_mean_loss = history[0].mean_loss;
        let validate = epoch % args.validate_every == 0 || epoch + 1 == args.epochs;
        let validation_mae_ev = if validate {
            validation_mae_ev(
                &model,
                &validation_entries,
                &input_scalers,
                &target_scalers,
                args.batch_size,
            )?
        } else {
            f64::NAN
        };
        let is_best = validation_mae_ev < best_validation_mae;
        if is_best {
            best_validation_mae = validation_mae_ev;
            best_epoch = epoch;
            best_parameters = model
                .parameter_values()
                .map_err(|error| format!("snapshot: {error}"))?;
        }
        println!(
            "epoch {:>4}/{:>4}  train_loss {:.6}  val_mae {} eV{}",
            epoch + 1,
            args.epochs,
            train_mean_loss,
            if validation_mae_ev.is_finite() {
                format!("{validation_mae_ev:.4}")
            } else {
                "—".into()
            },
            if is_best { "  *checkpoint*" } else { "" }
        );
        trace.push(EpochTrace {
            epoch,
            train_mean_loss,
            validation_mae_ev,
            is_best,
        });
    }
    let train_seconds = train_start.elapsed().as_secs_f64();

    // 5. Report: rebuild the checkpoint on the inference device, score the
    //    test subset per slot.
    let evaluate_start = Instant::now();
    let inference_model = Mlp::with_parameters(config, best_parameters, &inference_device)
        .map_err(|error| format!("checkpoint rebuild: {error}"))?;
    let mut slot_predictions: Vec<Vec<f32>> = vec![Vec::new(); Targets::WIDTH];
    let mut slot_references: Vec<Vec<f32>> = vec![Vec::new(); Targets::WIDTH];
    for entry in &test_entries {
        let scaled = input_scalers
            .scale_row(&entry.descriptor)
            .map_err(|error| format!("test scalers: {error}"))?;
        let input = Tensor::<1>::from_data(
            scaled,
            TensorShape::new(vec![DESCRIPTOR_WIDTH])
                .map_err(|error| format!("test input shape: {error}"))?,
            &inference_device,
        )
        .map_err(|error| format!("test input: {error}"))?;
        let output = predict(&inference_model, &input)
            .map_err(|error| format!("test predict: {error}"))?
            .to_vec()
            .map_err(|error| format!("test readback: {error}"))?;
        for slot in 0..Targets::WIDTH {
            if let Some(reference) = entry.references[slot] {
                let prediction = target_scalers[slot].unscale(output[slot]);
                slot_predictions[slot].push(prediction);
                slot_references[slot].push(reference);
            }
        }
    }
    let evaluate_seconds = evaluate_start.elapsed().as_secs_f64();
    let slot_names = ["homo", "lumo", "gap"];
    let mut slots = Vec::with_capacity(Targets::WIDTH);
    let mut pooled_pairs: Vec<(f32, f32)> = Vec::new();
    for slot in 0..Targets::WIDTH {
        let predictions = &slot_predictions[slot];
        let references = &slot_references[slot];
        pooled_pairs.extend(predictions.iter().zip(references).map(|(p, r)| (*p, *r)));
        slots.push(SlotMetrics {
            name: slot_names[slot].into(),
            n: references.len(),
            mae_ev: mae(predictions, references).map_err(|error| format!("test MAE: {error}"))?,
            rmse_ev: rmse(predictions, references)
                .map_err(|error| format!("test RMSE: {error}"))?,
            r2: r2(predictions, references).ok(),
        });
    }
    let pooled_mae_ev = mae(
        &pooled_pairs.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
        &pooled_pairs.iter().map(|(_, r)| *r).collect::<Vec<_>>(),
    )
    .map_err(|error| format!("pooled MAE: {error}"))?;
    let pooled_rmse_ev = rmse(
        &pooled_pairs.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
        &pooled_pairs.iter().map(|(_, r)| *r).collect::<Vec<_>>(),
    )
    .map_err(|error| format!("pooled RMSE: {error}"))?;

    let report = BenchmarkReport {
        records: loaded.record_count(),
        excluded_uncharacterized: loaded.excluded_uncharacterized(),
        skipped: loaded.skipped().len(),
        title_mismatches: loaded.title_mismatches(),
        loaded_entries: dataset.len(),
        subset: args.subset,
        split_seed: args.seed,
        train_size: split.train().len(),
        validation_size: split.validation().len(),
        test_size: split.test().len(),
        hidden: args.hidden.clone(),
        epochs: args.epochs,
        lr: args.lr,
        init_seed: args.seed,
        optimizer: optimizer.name(),
        schedule: schedule.name(),
        load_seconds,
        featurize_seconds,
        train_seconds,
        evaluate_seconds,
        trace,
        best_epoch,
        slots,
        pooled_mae_ev,
        pooled_rmse_ev,
    };

    // 6. Persist: JSON for tooling, Markdown for humans.
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|error| format!("creating {}: {error}", args.out_dir))?;
    let json_path = format!("{}/result.json", args.out_dir);
    let markdown_path = format!("{}/result.md", args.out_dir);
    std::fs::write(&json_path, report.to_json().to_json() + "\n")
        .map_err(|error| format!("writing {json_path}: {error}"))?;
    std::fs::write(&markdown_path, report.to_markdown())
        .map_err(|error| format!("writing {markdown_path}: {error}"))?;

    println!(
        "\n=== test results (checkpoint epoch {}) ===",
        report.best_epoch
    );
    for slot in &report.slots {
        println!(
            "  {:>4}: n {:>6}  MAE {:.4} eV  RMSE {:.4} eV  R² {}",
            slot.name,
            slot.n,
            slot.mae_ev,
            slot.rmse_ev,
            slot.r2
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "undefined".into())
        );
    }
    println!(
        "  pooled: MAE {:.4} eV  RMSE {:.4} eV",
        report.pooled_mae_ev, report.pooled_rmse_ev
    );
    println!(
        "total {:.1}s (load {:.1}, featurize {:.1}, train {:.1}, eval {:.1})",
        total_start.elapsed().as_secs_f64(),
        load_seconds,
        featurize_seconds,
        train_seconds,
        evaluate_seconds
    );
    println!("reports: {json_path} , {markdown_path}");
    Ok(())
}

/// Extracts the raw descriptor and present targets of one entry.
fn entry_to_evaluation(entry: &mi_data::MolecularEntry) -> EvaluationEntry {
    let descriptor = descriptor_vector(entry.molecule(), &Cpu::inference())
        .map_err(|error| format!("descriptor: {error}"))
        .and_then(|tensor| {
            tensor
                .to_vec()
                .map_err(|error| format!("descriptor readback: {error}"))
        })
        .expect("descriptors are infallible for loader-validated molecules");
    let targets = Targets::extract(entry);
    let mut references = [None; Targets::WIDTH];
    for (slot, reference) in references.iter_mut().enumerate() {
        *reference = targets.present_value(slot);
    }
    EvaluationEntry {
        descriptor,
        references,
    }
}

/// Pooled validation MAE (eV) across all present slots.
fn validation_mae_ev(
    model: &Mlp,
    entries: &[EvaluationEntry],
    input_scalers: &ColumnScalers,
    target_scalers: &[StandardScaler],
    batch_size: usize,
) -> Result<f64, String> {
    let device = Cpu::inference();
    let mut predictions = Vec::new();
    let mut references = Vec::new();
    for chunk in entries.chunks(batch_size.max(1)) {
        let mut rows = Vec::with_capacity(chunk.len() * DESCRIPTOR_WIDTH);
        for entry in chunk {
            rows.extend(
                input_scalers
                    .scale_row(&entry.descriptor)
                    .map_err(|error| format!("validation scalers: {error}"))?,
            );
        }
        let input = Tensor::<2>::from_data(
            rows,
            TensorShape::new(vec![chunk.len(), DESCRIPTOR_WIDTH])
                .map_err(|error| format!("validation input shape: {error}"))?,
            &device,
        )
        .map_err(|error| format!("validation input: {error}"))?;
        let outputs = model
            .forward(&input)
            .map_err(|error| format!("validation predict: {error}"))?
            .to_vec()
            .map_err(|error| format!("validation readback: {error}"))?;
        for (row, entry) in outputs.chunks(Targets::WIDTH).zip(chunk) {
            for slot in 0..Targets::WIDTH {
                if let Some(reference) = entry.references[slot] {
                    predictions.push(target_scalers[slot].unscale(row[slot]));
                    references.push(reference);
                }
            }
        }
    }
    mae(&predictions, &references).map_err(|error| format!("validation MAE: {error}"))
}
