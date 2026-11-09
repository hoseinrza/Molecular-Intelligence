//! Dataset entry: a molecule plus its molecular properties.

use crate::error::DataError;
use crate::property::ElectronicProperties;
use mi_chemistry::Molecule;
use mi_graph::MolecularGraph;

/// One dataset record: a validated [`Molecule`] and its
/// [`ElectronicProperties`].
///
/// The molecule is validated at construction, so a dataset can never hold a
/// structurally invalid molecule; empty molecules are rejected (a dataset
/// record without a molecular structure is meaningless). Entries are
/// immutable after construction; the entry's position index is defined by
/// the dataset it is pushed into.
#[derive(Debug, Clone, PartialEq)]
pub struct MolecularEntry {
    molecule: Molecule,
    properties: ElectronicProperties,
}

impl MolecularEntry {
    /// Creates an entry from a molecule with no properties attached.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::InvalidMolecule`] if `molecule` fails its
    /// structural validation (in particular, empty molecules are rejected).
    pub fn new(molecule: Molecule) -> Result<Self, DataError> {
        molecule.validate().map_err(DataError::InvalidMolecule)?;
        Ok(Self {
            molecule,
            properties: ElectronicProperties::new(),
        })
    }

    /// Attaches molecular properties and returns the entry.
    pub fn with_properties(mut self, properties: ElectronicProperties) -> Self {
        self.properties = properties;
        self
    }

    /// The molecule of this record.
    pub fn molecule(&self) -> &Molecule {
        &self.molecule
    }

    /// The properties of this record (absent properties are [`None`]).
    pub fn properties(&self) -> &ElectronicProperties {
        &self.properties
    }

    /// Molecular graph view of this record's molecule.
    ///
    /// The graph is converted on demand through `mi-graph` and is
    /// **recomputed on every call** — no caching exists in Phase 3 (see the
    /// architecture documentation for the planned caching strategy).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`mi_graph::GraphError`] if graph conversion
    /// fails; for a validated entry this is defensive only.
    pub fn graph(&self) -> Result<MolecularGraph, mi_graph::GraphError> {
        MolecularGraph::from_molecule(&self.molecule)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::Energy;
    use crate::test_support;

    #[test]
    fn valid_molecule_creates_entry_without_properties() {
        let entry = MolecularEntry::new(test_support::water()).unwrap();
        assert_eq!(entry.molecule().atom_count(), 3);
        assert_eq!(entry.properties(), &ElectronicProperties::new());
        assert_eq!(entry.properties().homo(), None);
    }

    #[test]
    fn empty_molecule_is_rejected() {
        let error = MolecularEntry::new(Molecule::new()).unwrap_err();
        assert!(matches!(
            error,
            DataError::InvalidMolecule(
                mi_chemistry::ChemistryError::InvalidMolecularStructure { .. }
            )
        ));
    }

    #[test]
    fn properties_are_attached_and_accessible() {
        let entry = MolecularEntry::new(test_support::methane())
            .unwrap()
            .with_properties(test_support::properties(-14.0, 5.0));
        assert_eq!(entry.properties().homo().map(Energy::ev), Some(-14.0));
        assert_eq!(entry.properties().lumo().map(Energy::ev), Some(5.0));
        // Gap is derived when not stored.
        assert_eq!(
            entry.properties().homo_lumo_gap().map(Energy::ev),
            Some(19.0)
        );
    }

    #[test]
    fn graph_view_matches_the_molecule() {
        let entry = MolecularEntry::new(test_support::water()).unwrap();
        let graph = entry.graph().unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.degree(0).unwrap(), 2);
    }

    #[test]
    fn entries_compare_by_value() {
        let first = MolecularEntry::new(test_support::water()).unwrap();
        let second = MolecularEntry::new(test_support::water()).unwrap();
        assert_eq!(first, second);
        let different = MolecularEntry::new(test_support::methane()).unwrap();
        assert_ne!(first, different);
    }
}
