//! End-to-end baseline proof: dataset → split → descriptor+targets →
//! MLP training → inference snapshot → metrics. This is the executable
//! demonstration that featurization, autodiff, parameter updates, and
//! evaluation cooperate on the Flex CPU backend — the Phase 7 result
//! that Phase 8's benchmark reuses.
//!
//! Targets are synthetic (fixed eV values per molecule); the test proves
//! the *machinery* learns, not chemistry — the real QM9 run is the
//! documented procedure in `docs/architecture/baseline-model.md`.
//!
//! # Preprocessing (part of the documented recipe)
//!
//! Descriptor columns live on very different scales (an atomic-number sum
//! reaches ~30 while one-hot flags are 0/1), and the raw targets span
//! tens of eV. Feeding such raw values into per-sample SGD kills hidden
//! ReLU units in the first update — an instability this test initially
//! hit and a textbook motivation for preprocessing. The recipe therefore
//! z-scores inputs **and** targets over the training samples, trains in
//! the standardized space, and un-scales predictions before computing
//! the reported metrics (which stay in electronvolts).

use mi_chemistry::{Atom, BondType, ChemicalElement, Molecule, Position};
use mi_data::{
    ElectronicProperties, Energy, MolecularDataset, MolecularEntry, SplitFractions, random_split,
};
use mi_evaluation::{mae, r2, rmse};
use mi_featurization::{DESCRIPTOR_WIDTH, Targets, descriptor_vector};
use mi_models::{Mlp, MlpConfig, RegressionSample, predict};
use mi_tensor::{Cpu, Tensor, TensorShape};
use mi_training::{LrSchedule, Sgd, TrainConfig, train};

/// Six small molecules with fixed synthetic targets in eV; descriptors
/// differ by construction (different elements, bond orders, sizes).
fn fixture_dataset() -> MolecularDataset {
    let mut dataset = MolecularDataset::new().with_name("synthetic-baseline");

    let mut push = |name: &str, build: fn() -> Molecule, homo: f64, lumo: f64| {
        let _ = name;
        let properties = ElectronicProperties::new()
            .with_homo(Energy::from_ev(homo).unwrap())
            .with_lumo(Energy::from_ev(lumo).unwrap());
        dataset.push(
            MolecularEntry::new(build())
                .unwrap()
                .with_properties(properties),
        );
    };

    push("water", build_water, -12.6, 1.5);
    push("methane", build_methane, -13.6, 2.0);
    push("ethene", build_ethene, -10.0, 3.0);
    push("helium", build_helium, -24.5, 20.0);
    push("hf", build_hf, -15.5, 0.5);
    push("hydroxide", build_hydroxide, -6.0, 4.0);
    dataset
}

fn build_water() -> Molecule {
    let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
    let mut molecule = Molecule::new().with_name("water");
    molecule.add_atom(Atom::new(element("O")).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
    molecule
        .add_atom(Atom::new(element("H")).with_position(Position::new(0.0, 0.95, 0.0).unwrap()));
    molecule
        .add_atom(Atom::new(element("H")).with_position(Position::new(0.0, -0.95, 0.0).unwrap()));
    molecule.add_bond_between(0, 1, BondType::Single).unwrap();
    molecule.add_bond_between(0, 2, BondType::Single).unwrap();
    molecule
}

fn build_methane() -> Molecule {
    let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
    let mut molecule = Molecule::new().with_name("methane");
    molecule.add_atom(Atom::new(element("C")).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
    molecule.add_atom(Atom::new(element("H")).with_position(Position::new(1.1, 0.0, 0.0).unwrap()));
    molecule
        .add_atom(Atom::new(element("H")).with_position(Position::new(-1.1, 0.0, 0.0).unwrap()));
    molecule.add_atom(Atom::new(element("H")).with_position(Position::new(0.0, 1.1, 0.0).unwrap()));
    molecule
        .add_atom(Atom::new(element("H")).with_position(Position::new(0.0, -1.1, 0.0).unwrap()));
    for neighbor in 1..5 {
        molecule
            .add_bond_between(0, neighbor, BondType::Single)
            .unwrap();
    }
    molecule
}

fn build_ethene() -> Molecule {
    let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
    let mut molecule = Molecule::new().with_name("ethene");
    molecule.add_atom(Atom::new(element("C")).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
    molecule
        .add_atom(Atom::new(element("C")).with_position(Position::new(1.34, 0.0, 0.0).unwrap()));
    molecule.add_atom(Atom::new(element("H")).with_position(Position::new(0.0, 0.9, 0.0).unwrap()));
    molecule
        .add_atom(Atom::new(element("H")).with_position(Position::new(1.34, -0.9, 0.0).unwrap()));
    molecule.add_bond_between(0, 1, BondType::Double).unwrap();
    molecule.add_bond_between(0, 2, BondType::Single).unwrap();
    molecule.add_bond_between(1, 3, BondType::Single).unwrap();
    molecule
}

fn build_helium() -> Molecule {
    let mut molecule = Molecule::new().with_name("helium");
    molecule.add_atom(
        Atom::new(ChemicalElement::from_symbol("He").unwrap())
            .with_position(Position::new(0.0, 0.0, 0.0).unwrap()),
    );
    molecule
}

fn build_hf() -> Molecule {
    let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
    let mut molecule = Molecule::new().with_name("hf");
    molecule.add_atom(Atom::new(element("F")).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
    molecule
        .add_atom(Atom::new(element("H")).with_position(Position::new(0.0, 0.92, 0.0).unwrap()));
    molecule.add_bond_between(0, 1, BondType::Single).unwrap();
    molecule
}

fn build_hydroxide() -> Molecule {
    let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
    let mut molecule = Molecule::new().with_name("hydroxide");
    molecule.add_atom(
        Atom::new(element("O"))
            .with_formal_charge(-1)
            .with_position(Position::new(0.0, 0.0, 0.0).unwrap()),
    );
    molecule
        .add_atom(Atom::new(element("H")).with_position(Position::new(0.0, 0.97, 0.0).unwrap()));
    molecule.add_bond_between(0, 1, BondType::Single).unwrap();
    molecule
}

/// A column statistics record for z-score standardization and its inverse.
#[derive(Clone, Copy)]
struct Scaler {
    mean: f32,
    deviation: f32,
}

impl Scaler {
    fn fit(values: &[f32]) -> Self {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / values.len() as f32;
        Self {
            mean,
            deviation: variance.sqrt().max(1e-6),
        }
    }

    fn scale(&self, value: f32) -> f32 {
        (value - self.mean) / self.deviation
    }

    fn unscale(&self, value: f32) -> f32 {
        value * self.deviation + self.mean
    }
}

/// The standardized training set plus the scalars needed to report
/// metrics in electronvolts.
struct PreparedTraining {
    samples: Vec<RegressionSample>,
    input_scalers: Vec<Scaler>,
    target_scalers: Vec<Scaler>,
}

/// Builds standardized training samples on one device.
fn prepare_training(dataset: &MolecularDataset, device: &mi_tensor::Device) -> PreparedTraining {
    let raw_descriptors: Vec<Vec<f32>> = dataset
        .iter()
        .map(|entry| {
            descriptor_vector(entry.molecule(), device)
                .unwrap()
                .to_vec()
                .unwrap()
        })
        .collect();

    // Raw target values per slot, over present entries only.
    let target_scalers: Vec<Scaler> = (0..Targets::WIDTH)
        .map(|slot| {
            let column: Vec<f32> = dataset
                .iter()
                .filter_map(|entry| Targets::extract(entry).present_value(slot))
                .collect();
            Scaler::fit(&column)
        })
        .collect();

    let input_scalers: Vec<Scaler> = (0..DESCRIPTOR_WIDTH)
        .map(|slot| {
            let column: Vec<f32> = raw_descriptors.iter().map(|row| row[slot]).collect();
            Scaler::fit(&column)
        })
        .collect();

    let samples = dataset
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let input = Tensor::<1>::from_data(
                (0..DESCRIPTOR_WIDTH)
                    .map(|slot| input_scalers[slot].scale(raw_descriptors[index][slot]))
                    .collect(),
                TensorShape::new(vec![DESCRIPTOR_WIDTH]).unwrap(),
                device,
            )
            .unwrap();
            let targets = Targets::extract(entry);
            let (values, mask) = targets.to_tensors(device).unwrap();
            let value_slots = values.to_vec().unwrap();
            let mask_slots = mask.to_vec().unwrap();
            let scaled_values: Vec<f32> = (0..Targets::WIDTH)
                .map(|slot| {
                    if mask_slots[slot] == 1.0 {
                        target_scalers[slot].scale(value_slots[slot])
                    } else {
                        0.0
                    }
                })
                .collect();
            let values = Tensor::<1>::from_data(
                scaled_values,
                TensorShape::new(vec![Targets::WIDTH]).unwrap(),
                device,
            )
            .unwrap();
            RegressionSample {
                input,
                values,
                mask,
            }
        })
        .collect();

    PreparedTraining {
        samples,
        input_scalers,
        target_scalers,
    }
}

#[test]
fn training_reduces_loss_and_reaches_high_train_r2() {
    let dataset = fixture_dataset();
    let split = random_split(&dataset, SplitFractions::new(0.5, 0.25, 0.25).unwrap(), 42).unwrap();
    assert_eq!(split.train().len(), 3);
    assert_eq!(split.validation().len(), 1);
    assert_eq!(split.test().len(), 2);

    // The overfit proof trains on the whole fixture (the split above is
    // exercised for its sizes); the real benchmark trains on the split's
    // training subset with the same recipe.
    let training_device = Cpu::training();
    let prepared = prepare_training(&dataset, &training_device);
    assert_eq!(prepared.samples.len(), 6);

    let config = MlpConfig {
        input_dim: DESCRIPTOR_WIDTH,
        hidden: vec![32, 16],
        output_dim: Targets::WIDTH,
    };
    let mut model = Mlp::new_with_seed(config.clone(), &training_device, 7).unwrap();

    let epochs = 800;
    let history = train(
        &mut model,
        &prepared.samples,
        &mut Sgd::new(0.0).unwrap(),
        &TrainConfig {
            epochs,
            base_lr: 0.01,
            schedule: LrSchedule::Constant,
        },
        &training_device,
    )
    .unwrap();
    assert_eq!(history.len(), epochs);
    assert_eq!(history[0].epoch, 0);
    // The machinery learns: the final epoch loss is a small fraction of
    // the first, and no sample was skipped.
    assert!(
        history[epochs - 1].mean_loss < history[0].mean_loss * 0.01,
        "loss did not fall: {} → {}",
        history[0].mean_loss,
        history[epochs - 1].mean_loss
    );
    assert_eq!(history[epochs - 1].skipped_samples, 0);

    // Inference snapshot: parameters move to the inference device, then
    // predictions are un-scaled and scored with the evaluation metrics.
    let values = model.parameter_values().unwrap();
    let inference_model = Mlp::with_parameters(config, values, &Cpu::inference()).unwrap();
    let inference_device = Cpu::inference();

    let mut predictions = Vec::new();
    let mut references = Vec::new();
    for entry in dataset.iter() {
        // The model consumes standardized inputs — the raw descriptor must
        // pass through the same input scalers the training data saw.
        let raw_descriptor = descriptor_vector(entry.molecule(), &inference_device)
            .unwrap()
            .to_vec()
            .unwrap();
        let descriptor = Tensor::<1>::from_data(
            (0..DESCRIPTOR_WIDTH)
                .map(|slot| prepared.input_scalers[slot].scale(raw_descriptor[slot]))
                .collect(),
            TensorShape::new(vec![DESCRIPTOR_WIDTH]).unwrap(),
            &inference_device,
        )
        .unwrap();
        let prediction = predict(&inference_model, &descriptor)
            .unwrap()
            .to_vec()
            .unwrap();
        let targets = Targets::extract(entry);
        let (_, mask) = targets.to_tensors(&inference_device).unwrap();
        for (slot, &present) in mask.to_vec().unwrap().iter().enumerate() {
            if present == 1.0 {
                let ev_prediction = prepared.target_scalers[slot].unscale(prediction[slot]);
                let reference = targets
                    .present_value(slot)
                    .expect("a present slot always carries its value");
                predictions.push(ev_prediction);
                references.push(reference);
            }
        }
    }
    let (train_mae, train_rmse, train_r2) = evaluation_metrics(&predictions, &references);
    assert!(
        train_mae < 0.3,
        "overfit MAE too high: {train_mae} (rmse {train_rmse}, r2 {train_r2})"
    );
    assert!(
        train_r2 > 0.99,
        "overfit R² too low: {train_r2} (mae {train_mae}, rmse {train_rmse})"
    );
}

#[test]
fn same_seed_produces_the_same_loss_curve() {
    let dataset = fixture_dataset();
    let training_device = Cpu::training();
    let prepared = prepare_training(&dataset, &training_device);

    let config = MlpConfig {
        input_dim: DESCRIPTOR_WIDTH,
        hidden: vec![32, 16],
        output_dim: Targets::WIDTH,
    };
    let mut first = Mlp::new_with_seed(config.clone(), &training_device, 7).unwrap();
    let mut second = Mlp::new_with_seed(config, &training_device, 7).unwrap();

    let config = TrainConfig {
        epochs: 40,
        base_lr: 0.01,
        schedule: LrSchedule::Constant,
    };
    let first_history = train(
        &mut first,
        &prepared.samples,
        &mut Sgd::new(0.0).unwrap(),
        &config,
        &training_device,
    )
    .unwrap();
    let second_history = train(
        &mut second,
        &prepared.samples,
        &mut Sgd::new(0.0).unwrap(),
        &config,
        &training_device,
    )
    .unwrap();
    assert_eq!(first_history, second_history);
}

#[test]
fn fully_masked_samples_are_skipped_and_counted() {
    // An entry without any properties produces an all-zero mask; the loop
    // must skip it per epoch and say so.
    let mut dataset = MolecularDataset::new().with_name("skip-count");
    let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
    let mut water = Molecule::new();
    water.add_atom(Atom::new(element("O")));
    water.add_atom(Atom::new(element("H")));
    water.add_bond_between(0, 1, BondType::Single).unwrap();
    dataset.push(MolecularEntry::new(water).unwrap());

    let training_device = Cpu::training();
    let prepared = prepare_training(&dataset, &training_device);
    assert_eq!(prepared.samples.len(), 1);

    let config = MlpConfig {
        input_dim: DESCRIPTOR_WIDTH,
        hidden: vec![8],
        output_dim: Targets::WIDTH,
    };
    let mut model = Mlp::new_with_seed(config, &training_device, 3).unwrap();
    let history = train(
        &mut model,
        &prepared.samples,
        &mut Sgd::new(0.0).unwrap(),
        &TrainConfig {
            epochs: 5,
            base_lr: 0.01,
            schedule: LrSchedule::Constant,
        },
        &training_device,
    )
    .unwrap();
    for metrics in history {
        assert_eq!(metrics.skipped_samples, 1);
        assert_eq!(metrics.mean_loss, 0.0, "skipped epochs carry no loss");
    }
}

fn evaluation_metrics(predictions: &[f32], references: &[f32]) -> (f64, f64, f64) {
    (
        mae(predictions, references).unwrap(),
        rmse(predictions, references).unwrap(),
        r2(predictions, references).unwrap(),
    )
}
