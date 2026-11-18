"""One-shot updater: descriptor layout v1 -> v2 (29 slots).

Run with: python -I scripts/update_descriptor_v2.py
Idempotence is not guaranteed; run once.
"""

PATH = "crates/mi-featurization/src/descriptor.rs"

OLD_DOC = """//! # Descriptor layout v1 ([`DESCRIPTOR_WIDTH`] = 18)
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
//! presence-mask convention of the targets; no position is invented."""

NEW_DOC = """//! # Descriptor layout v2 ([`DESCRIPTOR_WIDTH`] = 29)
//!
//! v2 extends the Phase 7 layout v1 (slots 0–17, unchanged) with the
//! composition and whole-molecule geometry the electronic properties
//! actually depend on:
//!
//! | Slots | Content | Source |
//! | ----- | ------- | ------ |
//! | 0–5   | per-column **sums** of `NodeFeatures::to_vector` (total atomic number, net formal charge, isotope sums, degree sums) | node schema v1 |
//! | 6–12  | per-column **sums** of `EdgeFeatures::to_vector` over chemical (undirected) bonds (bond-type counts, total bond order) | edge schema v1 |
//! | 13–14 | atom count, bond count | topology |
//! | 15    | `has_positions` flag (1.0 when geometry was included) | mask convention |
//! | 16–17 | mean and maximum bond distance in Ångström (0.0 when positions are absent) | [`geometry`](crate::geometry) |
//! | 18    | molecular weight (Σ standard atomic masses, u) | [`ChemicalElement::atomic_mass`](mi_chemistry) |
//! | 19    | heavy-atom count (Z > 1) | element reference data |
//! | 20    | heteroatom count (Z ∉ {1, 6}) | element reference data |
//! | 21–26 | composition counts `[H, C, N, O, F, other]` | element reference data |
//! | 27    | radius of gyration in Å (0.0 when positions are absent) | atom positions |
//! | 28    | molecular diameter (max pairwise atom distance, Å; 0.0 when absent) | atom positions |
//!
//! Aggregating a categorical column (slot 0 sums atomic numbers) is
//! physically meaningful here: the sum of atomic numbers is the total
//! electron count. Every v2 slot is either an aggregate of versioned
//! schema columns, a count over the element reference data, or pure
//! geometry from the Cartesian positions — nothing is invented. A
//! molecule without positions still gets a descriptor — the flag flips
//! to 0 and the geometric slots are 0, exactly the presence-mask
//! convention of the targets; no position is invented."""

OLD_WIDTH = """/// Width of the fixed-size descriptor vector (layout v1).
pub const DESCRIPTOR_WIDTH: usize = 18;"""

NEW_WIDTH = """/// Width of the fixed-size descriptor vector (layout v2).
pub const DESCRIPTOR_WIDTH: usize = 29;"""

OLD_BODY = """    // Slots 15–17: geometry presence flag and bond-distance aggregates.
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

    let shape = TensorShape::new(vec![DESCRIPTOR_WIDTH])?;"""

NEW_BODY = """    // Slots 15–17: geometry presence flag and bond-distance aggregates.
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

    // Slots 18–26: composition from the element reference data.
    let mut weight = 0.0f64;
    let mut heavy_atoms = 0.0f64;
    let mut heteroatoms = 0.0f64;
    let mut composition = [0.0f64; 6]; // [H, C, N, O, F, other]
    for atom in molecule.atoms() {
        let number = u32::from(atom.atomic_number());
        weight += atom.atomic_mass();
        if number > 1 {
            heavy_atoms += 1.0;
        }
        if number != 1 && number != 6 {
            heteroatoms += 1.0;
        }
        match number {
            1 => composition[0] += 1.0,
            6 => composition[1] += 1.0,
            7 => composition[2] += 1.0,
            8 => composition[3] += 1.0,
            9 => composition[4] += 1.0,
            _ => composition[5] += 1.0,
        }
    }
    descriptor[18] = weight;
    descriptor[19] = heavy_atoms;
    descriptor[20] = heteroatoms;
    descriptor[21..27].copy_from_slice(&composition);

    // Slots 27–28: whole-molecule geometry — radius of gyration and
    // diameter. Both require every atom to carry a position (the same
    // no-invented-geometry convention as slots 15–17).
    let positions: Vec<(f64, f64, f64)> = molecule
        .atoms()
        .iter()
        .filter_map(|atom| atom.position().map(|p| p.components()))
        .collect();
    if positions.len() == molecule.atoms().len() && positions.len() > 1 {
        let count = positions.len() as f64;
        let centroid = (
            positions.iter().map(|p| p.0).sum::<f64>() / count,
            positions.iter().map(|p| p.1).sum::<f64>() / count,
            positions.iter().map(|p| p.2).sum::<f64>() / count,
        );
        let mean_square = positions
            .iter()
            .map(|p| {
                let dx = p.0 - centroid.0;
                let dy = p.1 - centroid.1;
                let dz = p.2 - centroid.2;
                dx * dx + dy * dy + dz * dz
            })
            .sum::<f64>()
            / count;
        descriptor[27] = mean_square.sqrt();
        let mut diameter = 0.0f64;
        for (i, a) in positions.iter().enumerate() {
            for b in &positions[i + 1..] {
                let dx = a.0 - b.0;
                let dy = a.1 - b.1;
                let dz = a.2 - b.2;
                diameter = diameter.max((dx * dx + dy * dy + dz * dz).sqrt());
            }
        }
        descriptor[28] = diameter;
    }

    let shape = TensorShape::new(vec![DESCRIPTOR_WIDTH])?;"""

OLD_WIDTH_TEST = """    #[test]
    fn layout_width_is_eighteen() {
        assert_eq!(DESCRIPTOR_WIDTH, 18);
        let vector = descriptor_vector(&water(), &device()).unwrap();
        assert_eq!(vector.shape().dims(), &[DESCRIPTOR_WIDTH]);
    }"""

NEW_WIDTH_TEST = """    #[test]
    fn layout_width_is_twenty_nine() {
        assert_eq!(DESCRIPTOR_WIDTH, 29);
        let vector = descriptor_vector(&water(), &device()).unwrap();
        assert_eq!(vector.shape().dims(), &[DESCRIPTOR_WIDTH]);
    }"""

OLD_GEO_TEST = """        assert_eq!(values[15], 1.0);
        assert_eq!(values[16], 1.5);
        assert_eq!(values[17], 1.5);
    }"""

NEW_GEO_TEST = """        assert_eq!(values[15], 1.0);
        assert_eq!(values[16], 1.5);
        assert_eq!(values[17], 1.5);
        // Radius of gyration: the centroid sits midway on the bond, so
        // both atoms sit 0.75 Å from it. Diameter = bond length = 1.5 Å.
        assert!((values[27] - 0.75).abs() < 1e-9);
        assert!((values[28] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn composition_slots_are_exact_for_water() {
        let values = descriptor_vector(&water(), &device()).unwrap().to_vec().unwrap();
        // Molecular weight: 15.999 + 1.008 + 1.008.
        assert!((values[18] - 18.015).abs() < 1e-9);
        // One heavy atom, one heteroatom (the oxygen).
        assert_eq!(values[19], 1.0);
        assert_eq!(values[20], 1.0);
        // Composition [H, C, N, O, F, other] = [2, 0, 0, 1, 0, 0].
        assert_eq!(&values[21..27], &[2.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        // No positions: whole-molecule geometry stays 0.
        assert_eq!(values[27], 0.0);
        assert_eq!(values[28], 0.0);
    }

    #[test]
    fn carbon_dioxide_composition_and_geometry() {
        let mut molecule = Molecule::new();
        let element = |symbol: &str| ChemicalElement::from_symbol(symbol).unwrap();
        molecule
            .add_atom(Atom::new(element("C")).with_position(Position::new(0.0, 0.0, 0.0).unwrap()));
        molecule.add_atom(
            Atom::new(element("O")).with_position(Position::new(1.2, 0.0, 0.0).unwrap()),
        );
        molecule.add_atom(
            Atom::new(element("O")).with_position(Position::new(-1.2, 0.0, 0.0).unwrap()),
        );
        let values = descriptor_vector(&molecule, &device()).unwrap().to_vec().unwrap();
        // Composition [H, C, N, O, F, other] = [0, 1, 0, 2, 0, 0]; three
        // heavy atoms, two heteroatoms; weight 12.011 + 2 × 15.999.
        assert_eq!(&values[21..27], &[0.0, 1.0, 0.0, 2.0, 0.0, 0.0]);
        assert_eq!(values[19], 3.0);
        assert_eq!(values[20], 2.0);
        assert!((values[18] - (12.011 + 2.0 * 15.999)).abs() < 1e-9);
        // Linear molecule: Rg = 0.8 Å (atoms at −1.2, 0, +1.2), diameter 2.4 Å.
        assert!((values[27] - 0.8).abs() < 1e-9);
        assert!((values[28] - 2.4).abs() < 1e-9);
    }"""


def main() -> None:
    text = open(PATH, encoding="utf-8").read()
    for label, old, new in [
        ("module doc", OLD_DOC, NEW_DOC),
        ("width const", OLD_WIDTH, NEW_WIDTH),
        ("body", OLD_BODY, NEW_BODY),
        ("width test", OLD_WIDTH_TEST, NEW_WIDTH_TEST),
        ("geometry test", OLD_GEO_TEST, NEW_GEO_TEST),
    ]:
        if old not in text:
            raise SystemExit(f"pattern not found: {label}")
        text = text.replace(old, new)
        print(f"applied: {label}")
    open(PATH, "w", encoding="utf-8").write(text)
    print("descriptor v2 written")


if __name__ == "__main__":
    main()
