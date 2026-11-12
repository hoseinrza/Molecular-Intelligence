//! Molecular file format I/O for Molecular Intelligence.
//!
//! This crate brings real molecular data into the project: it parses and
//! writes the exchange formats of computational chemistry and imports
//! them into the validated, typed dataset layer of `mi-data`. It depends
//! on `mi-chemistry` (chemical truth) and `mi-data` (datasets) and on
//! nothing else — no ML, no tensors: the chemistry layer stays
//! learning-algorithm-free, and file I/O stays free of learning
//! algorithms too.
//!
//! # Formats
//!
//! - **XYZ** — element symbols plus Cartesian positions in Ångström;
//!   the coordinate lingua franca ([`xyz`]).
//! - **SDF / MDL molfile V2000** — the structure-carrier format of
//!   chemical datasets: positions, bonds with orders, formal charges,
//!   per-record data fields, multi-record files ([`sdf`]).
//!
//! Both modules provide parsing *and* writing, and the round-trip
//! contracts are tested (see the module docs for the precision each
//! format preserves).
//!
//! # Import and provenance
//!
//! [`import`] turns parsed records into a
//! [`MolecularDataset`](mi_data::MolecularDataset) with electronic
//! properties attached, converting energy values according to the
//! [`DatasetMetadata`] energy unit. Its core rule: **invalid data is
//! never silently dropped or accepted** — every record either enters the
//! dataset or is reported with its reason (or, in strict mode, aborts
//! the import).
//!
//! ```
//! use mi_io::{import_sdf, parse_sdf, DatasetMetadata, EnergyUnit, ImportOptions};
//!
//! let input = "\
//! water
//!   mi-io     0
//!
//!   3  2  0  0  0  0  0  0  0  0999 V2000
//!     0.0000    0.0000    0.1173 O   0  0  0  0  0  0  0  0  0  0  0  0
//!     0.0000    0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
//!     0.0000   -0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0
//!   1  2  1  0  0  0  0
//!   1  3  1  0  0  0  0
//! M  END
//! > <homo>
//! -0.3133
//!
//! $$$$
//! ";
//!
//! let records = parse_sdf(input)?;
//! let metadata = DatasetMetadata::new("toy", EnergyUnit::Hartree)?
//!     .with_functional("B3LYP")?;
//! let imported = import_sdf(&records, metadata, &ImportOptions::new())?;
//!
//! assert_eq!(imported.dataset().len(), 1);
//! assert!(imported.skipped().is_empty());
//! let homo = imported.dataset().entry(0)?.properties().homo().unwrap();
//! // The Hartree value was converted to electronvolts on the way in.
//! let expected = -0.3133 * mi_data::Energy::HARTREE_TO_EV;
//! assert!((homo.ev() - expected).abs() < 1e-9);
//! # Ok::<(), mi_io::IoError>(())
//! ```
//!
//! # Error policy
//!
//! Every fallible operation reports [`IoError`] with the 1-based file
//! line number of the offending data wherever that is meaningful — a
//! failing dataset can be debugged directly in its source file. Unknown
//! element symbols, non-finite coordinates, unsupported bond orders, and
//! unparseable property values are errors, never approximations.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod import;
pub mod metadata;
pub mod sdf;
pub mod xyz;

pub use error::{IoError, IoResult};
pub use import::{
    ImportOptions, ImportedDataset, PropertyFields, SkippedEntry, import_sdf, import_xyz,
};
pub use metadata::{DatasetMetadata, EnergyUnit};
pub use sdf::{
    SdfRecord, parse_sdf, parse_sdf_file, write_sdf_file, write_sdf_molecule, write_sdf_record,
    write_sdf_records,
};
pub use xyz::{parse_xyz, parse_xyz_file, write_xyz, write_xyz_file};
