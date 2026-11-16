//! Fixed-size molecular descriptors for non-graph models.
//!
//! Message-passing models consume [`GraphTensors`]; simpler models (the
//! Phase 7 MLP baseline, tree-based baselines) need one fixed-size vector
//! per molecule. This module aggregates the *same versioned feature
//! schemas* into that vector — nothing is invented: every column is an
//! aggregate of what the schemas already define, plus explicit counts and
//! geometry.
//!
//! # Descriptor layout v1 ([`DESCRIPTOR_WIDTH`] = 18)
//!
//! | Slots | Content | Source |
//! | ----- | ------- | ------ |
//! | 0–5   | per-column **sums** of `NodeFeatures::to_vector` (total atomic number, net formal charge, isotope sums, degree sums) | node schema v1 |
//! | 6–12  | per-column **sums** of `EdgeFeatures::to_vector` over chemical (undirected) bonds (bond-type counts, total bond order) | edge schema v1 |
//! | 13–14 | atom count, bond count | topology |
//! | 15    | `has_positions` flag (1.0 when geometry was included) | mask convention |
//! | 16–17 | mean and maximum bond distance in Ångström (0.0 when positions are absent) | [`geometry`](crate::geometry) |
//!
//! Aggregating a categorical column (slot 0 sums atomic numbers) is
//! physically meaningful here: the sum of atomic numbers is the total
//! electron count. A molecule without positions still gets a descriptor —
//! the flag flips to 0 and the geometric slots are 0, exactly the
//! presence-mask convention of the targets; no position is invented.

use crate::error::FeaturizationResult;
use crate::geometry;
use mi_chemistry::Molecule;
use mi_graph::MolecularGraph;
use mi_tensor::{Device, Tensor, TensorShape};

/// Width of the fixed-size descriptor vector (layout v1).
pub const DESCRIPTOR_WIDTH: usize = 18;

/// Builds the fixed-size descriptor vector of one molecule on `device`.
///
/// # Errors
///
/// Returns [`crate::FeaturizationError`] if the molecule fails graph
/// construction or tensor ingress fails.
pub fn descriptor_vector(molecule: &Molecule, device: &Device) -> FeaturizationResult<Tensor<1>> {
    let graph = MolecularGraph::from_molecule(molecule)?;

    let mut descriptor = vec![0.0f64; DESCRIPTOR_WIDTH];

    // Slots 0–5: node-schema column sums.
    for node in graph.nodes() {
        for (slot, value) in node.features().to_vector().into_iter().enumerate() {
            descriptor[slot] += value;
        }
    }

    // Slots 6–12: edge-schema column sums over chemical (undirected) bonds.
    for edge in graph.edges() {
        for (slot, value) in edge.features().to_vector().into_iter().enumerate() {
            descriptor[6 + slot] += value;
        }
    }

    // Slots 13–14: counts.
    descriptor[13] = graph.node_count() as f64;
    descriptor[14] = graph.edge_count() as f64;

    // Slots 15–17: geometry presence flag and bond-distance aggregates.
    match geometry::bond_distances(molecule) {
        Ok(distances) if !distances.is_empty() => {
            descriptor[15] = 1.0;
            let count = distances.len() as f64;
            let mean = distances.iter().map(|&d| f64::from(d)).sum::<f64>() / count;
            let max = distances.iter().map(|&d| f64::from(d)).fold(0.0, f64::max);
            descriptor[16] = mean;
            descriptor[17] = max;
        }
        // No bonds at all, or no positions: flag stays 0, slots stay 0.
        _ => {}
    }

    let shape = TensorShape::new(vec![DESCRIPTOR_WIDTH])?;
    let data = descriptor.into_iter().map(|value| value as f32).collect();
    Ok(Tensor::from_data(data, shape, device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mi_chemistry::{Atom, BondType, ChemicalElement, Position};
    use mi_tensor::Cpu;

    fn device() -> Device {
        Cpu::inference()
    }

    fn water() -> Molecule {
        let mut molecule = Molecule::new().with_name("water");
        let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
        molecule.add_atom(Atom::new(element("O")));
        molecule.add_atom(Atom::new(element("H")));
        molecule.add_atom(Atom::new(element("H")));
        molecule.add_bond_between(0, 1, BondType::Single).unwrap();
        molecule.add_bond_between(0, 2, BondType::Single).unwrap();
        molecule
    }

    #[test]
    fn layout_width_is_eighteen() {
        assert_eq!(DESCRIPTOR_WIDTH, 18);
        let vector = descriptor_vector(&water(), &device()).unwrap();
        assert_eq!(vector.shape().dims(), &[DESCRIPTOR_WIDTH]);
    }

    #[test]
    fn water_descriptor_is_exact() {
        let values = descriptor_vector(&water(), &device())
            .unwrap()
            .to_vec()
            .unwrap();
        // Node sums: Z 8+1+1 = 10; charges 0; isotope flags 0; isotope
        // numbers 0; degrees 2+1+1 = 4; bond counts 2+1+1 = 4.
        assert_eq!(&values[0..6], &[10.0, 0.0, 0.0, 0.0, 4.0, 4.0]);
        // Edge sums: two single bonds → is_single 2, others 0,
        // has_bond_order 2, bond order 1+1 = 2.
        assert_eq!(&values[6..13], &[2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0]);
        // Counts and geometry (no positions → flag 0, geometry slots 0).
        assert_eq!(&values[13..18], &[3.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn geometry_slots_activate_with_positions() {
        let mut molecule = Molecule::new();
        let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
        molecule
            .add_atom(Atom::new(element("O")).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
        molecule
            .add_atom(Atom::new(element("H")).with_position(Position::new(0.0, 1.5, 0.0).unwrap()));
        molecule.add_bond_between(0, 1, BondType::Single).unwrap();
        let values = descriptor_vector(&molecule, &device())
            .unwrap()
            .to_vec()
            .unwrap();
        assert_eq!(values[15], 1.0);
        assert_eq!(values[16], 1.5);
        assert_eq!(values[17], 1.5);
    }

    #[test]
    fn monoatomic_molecule_descriptor_has_zero_edges() {
        let mut molecule = Molecule::new();
        molecule.add_atom(Atom::new(ChemicalElement::from_symbol("He").unwrap()));
        let values = descriptor_vector(&molecule, &device())
            .unwrap()
            .to_vec()
            .unwrap();
        assert_eq!(&values[0..6], &[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&values[6..13], &[0.0; 7]);
        assert_eq!(&values[13..18], &[1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn formal_charges_sum_to_net_charge() {
        // Hydroxide: O(−1)–H → net charge −1.
        let mut molecule = Molecule::new();
        let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
        molecule.add_atom(Atom::new(element("O")).with_formal_charge(-1));
        molecule.add_atom(Atom::new(element("H")));
        molecule.add_bond_between(0, 1, BondType::Single).unwrap();
        let values = descriptor_vector(&molecule, &device())
            .unwrap()
            .to_vec()
            .unwrap();
        assert_eq!(values[1], -1.0);
    }
}
