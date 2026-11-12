//! Geometric features from Cartesian positions.
//!
//! `mi-graph`'s feature schemas are deliberately topological (v1); three
//! dimensional structure is a separate, physically grounded quantity: it
//! comes from [`Atom::position`](mi_chemistry::Atom::position) values
//! entered through the I/O layer, never from invented data. This module
//! turns those positions into the distance features message-passing
//! models typically consume.
//!
//! # Semantics
//!
//! - Distances are Euclidean, in **Ångström**, matching the fixed unit of
//!   [`Position`](mi_chemistry::Position).
//! - `distance_matrix` requires **every** atom to carry a position: a
//!   partially placed molecule would silently mix invented zeros into
//!   physics, so it is rejected instead.
//! - Computation happens in `f64` on the host and enters the tensor as
//!   `f32` — well below the accuracy of any source geometry.

use crate::error::{FeaturizationError, FeaturizationResult};
use mi_chemistry::Molecule;
use mi_tensor::{Device, Tensor, TensorShape};

/// Builds the pairwise Euclidean distance matrix, shape `[N, N]`, in
/// Ångström.
///
/// The matrix is symmetric with a zero diagonal. Computation is O(N²),
/// which is negligible at molecular sizes (QM9 tops out below 30 atoms).
///
/// # Errors
///
/// Returns [`FeaturizationError::MissingPosition`] naming the first atom
/// without a position, and [`FeaturizationError::Tensor`] if tensor
/// ingress fails.
pub fn distance_matrix(molecule: &Molecule, device: &Device) -> FeaturizationResult<Tensor<2>> {
    let count = molecule.atom_count();
    let positions = collect_positions(molecule)?;
    let mut data = Vec::with_capacity(count * count);
    for i in 0..count {
        for j in 0..count {
            let (xi, yi, zi) = positions[i];
            let (xj, yj, zj) = positions[j];
            let dx = xi - xj;
            let dy = yi - yj;
            let dz = zi - zj;
            data.push((dx * dx + dy * dy + dz * dz).sqrt() as f32);
        }
    }
    let tensor = Tensor::from_data(data, TensorShape::new(vec![count, count])?, device)?;
    Ok(tensor)
}

/// Returns one distance per chemical bond, in `molecule.bonds()` order.
///
/// Pairs with graph-level features: row `k` of a bond-feature matrix can
/// be extended with `bond_distances()[k]` by model code that wants a
/// geometric column.
///
/// # Errors
///
/// Returns [`FeaturizationError::MissingPosition`] naming the first atom
/// without a position among bonded atoms.
pub fn bond_distances(molecule: &Molecule) -> FeaturizationResult<Vec<f32>> {
    let positions = collect_positions(molecule)?;
    Ok(molecule
        .bonds()
        .iter()
        .map(|bond| {
            let (first, second) = bond.endpoints();
            let (x1, y1, z1) = positions[first];
            let (x2, y2, z2) = positions[second];
            let (dx, dy, dz) = (x1 - x2, y1 - y2, z1 - z2);
            (dx * dx + dy * dy + dz * dz).sqrt() as f32
        })
        .collect())
}

/// Collects every atom's position, rejecting molecules with gaps.
fn collect_positions(molecule: &Molecule) -> FeaturizationResult<Vec<(f64, f64, f64)>> {
    molecule
        .atoms()
        .iter()
        .enumerate()
        .map(|(index, atom)| {
            atom.position()
                .map(mi_chemistry::Position::components)
                .ok_or(FeaturizationError::MissingPosition { atom_index: index })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FeaturizationError;
    use mi_chemistry::{Atom, BondType, ChemicalElement, Position};
    use mi_tensor::Cpu;

    /// Water with textbook-like geometry: O at the origin, H at ±0.7572 y.
    fn water() -> Molecule {
        let mut molecule = Molecule::new().with_name("water");
        let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
        molecule
            .add_atom(Atom::new(element("O")).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
        molecule.add_atom(
            Atom::new(element("H")).with_position(Position::new(0.0, 0.7572, 0.0).unwrap()),
        );
        molecule.add_atom(
            Atom::new(element("H")).with_position(Position::new(0.0, -0.7572, 0.0).unwrap()),
        );
        molecule.add_bond_between(0, 1, BondType::Single).unwrap();
        molecule.add_bond_between(0, 2, BondType::Single).unwrap();
        molecule
    }

    #[test]
    fn distance_matrix_is_exact_symmetric_and_zero_diagonal() {
        let matrix = distance_matrix(&water(), &Cpu::inference()).unwrap();
        assert_eq!(matrix.shape().dims(), &[3, 3]);
        let values = matrix.to_vec().unwrap();
        // Diagonal is exactly zero.
        for index in [0, 4, 8] {
            assert_eq!(values[index], 0.0);
        }
        // O–H distances are exactly 0.7572 (single-axis);
        // H–H spans both hydrogens (flat index 5).
        assert_eq!(values[1], 0.7572);
        assert_eq!(values[3], 0.7572);
        assert_eq!(values[2], 0.7572);
        assert_eq!(values[5], 1.5144);
    }

    #[test]
    fn monoatomic_distance_matrix_is_one_zero() {
        let mut molecule = Molecule::new();
        molecule.add_atom(
            Atom::new(ChemicalElement::from_symbol("He").unwrap())
                .with_position(Position::new(1.0, 2.0, 3.0).unwrap()),
        );
        let matrix = distance_matrix(&molecule, &Cpu::inference()).unwrap();
        assert_eq!(matrix.shape().dims(), &[1, 1]);
        assert_eq!(matrix.to_vec().unwrap(), vec![0.0]);
    }

    #[test]
    fn missing_position_is_rejected_with_the_atom_index() {
        let mut molecule = Molecule::new();
        let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
        molecule
            .add_atom(Atom::new(element("O")).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
        molecule.add_atom(Atom::new(element("H"))); // no position
        let error = distance_matrix(&molecule, &Cpu::inference())
            .map(|_| ())
            .unwrap_err();
        assert_eq!(error, FeaturizationError::MissingPosition { atom_index: 1 });
    }

    #[test]
    fn bond_distances_follow_bond_order() {
        let distances = bond_distances(&water()).unwrap();
        assert_eq!(distances.len(), 2);
        assert_eq!(distances[0], 0.7572);
        assert_eq!(distances[1], 0.7572);
    }

    #[test]
    fn three_dimensional_distances_are_euclidean() {
        let mut molecule = Molecule::new();
        let carbon = ChemicalElement::from_symbol("C").unwrap();
        molecule.add_atom(Atom::new(carbon).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
        molecule.add_atom(Atom::new(carbon).with_position(Position::new(1.0, 1.0, 1.0).unwrap()));
        molecule.add_bond_between(0, 1, BondType::Single).unwrap();
        let distances = bond_distances(&molecule).unwrap();
        assert!((distances[0] - 3.0f64.sqrt() as f32).abs() < 1e-6);
    }
}
