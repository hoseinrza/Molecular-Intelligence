//! End-to-end QM9 pipeline: streaming file → loader → dataset → split →
//! statistics. Mirrors the usage documented in
//! `docs/architecture/qm9-pipeline.md`; fixtures are inline and
//! Hartree-valued like the real distribution.

use mi_data::SplitFractions;
use mi_data::random_split;
use mi_io::{Qm9Options, load_qm9_sdf, uncharacterized_indices};

/// Three records; the second (position 2) is listed as uncharacterized
/// and the third carries no property fields (structural-only entry).
const RECORDS: &str = "\
gdb9_1
  mi-io     0

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.7572    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.7572    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
> <homo>
-0.4650

> <lumo>
0.0500

> <gap>
0.5150

$$$$
gdb9_2
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 He  0  0  0  0  0  0  0  0  0  0  0  0
M  END
> <homo>
-0.9000

$$$$
gdb9_3
  mi-io     0

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.2    0.0    0.0 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
M  END
$$$$
";

fn pipeline_fixtures(test: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "mi_io_qm9_pipeline_{}_{}",
        std::process::id(),
        test
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let sdf = dir.join("qm9.sdf");
    std::fs::write(&sdf, RECORDS).unwrap();
    let uncharacterized = dir.join("uncharacterized.txt");
    std::fs::write(&uncharacterized, "2\n").unwrap();
    (sdf, uncharacterized)
}

#[test]
fn streams_loader_splits_and_reports_statistics() {
    let (sdf, uncharacterized) = pipeline_fixtures("streams_loader");
    let excluded = uncharacterized_indices(&uncharacterized).unwrap();

    let report = load_qm9_sdf(&sdf, &Qm9Options::new(), &excluded).unwrap();
    assert_eq!(report.record_count(), 3);
    assert_eq!(report.excluded_uncharacterized(), 1);
    assert_eq!(report.title_mismatches(), 0);

    let dataset = report.dataset();
    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.name(), Some("QM9"));

    // Hartree → eV conversion happened at the boundary; the structural-only
    // entry carries no properties.
    let first = dataset.entry(0).unwrap();
    let expected_homo = -0.4650 * mi_data::Energy::HARTREE_TO_EV;
    assert!((first.properties().homo().unwrap().ev() - expected_homo).abs() < 1e-9);
    let second = dataset.entry(1).unwrap();
    assert_eq!(second.properties().homo(), None);

    // Provenance is complete: DFT level, unit, source id.
    let metadata = report.metadata();
    assert_eq!(metadata.functional(), Some("B3LYP"));
    assert_eq!(metadata.energy_unit(), mi_io::EnergyUnit::Hartree);

    // Reproducible split over the loaded dataset.
    let split = random_split(dataset, SplitFractions::new(0.5, 0.25, 0.25).unwrap(), 42).unwrap();
    assert_eq!(split.train().len(), 1);
    assert_eq!(split.validation().len(), 0);
    assert_eq!(split.test().len(), 1);
    assert_eq!(split.total_size(), 2);

    // Statistics describe the loaded (pre-split) dataset.
    let statistics = report.dataset().statistics();
    assert_eq!(statistics.entry_count(), 2);
    let homo = statistics.homo().unwrap();
    assert_eq!(homo.count(), 1);
}

#[test]
fn streaming_reader_walks_the_same_file() {
    let (sdf, _) = pipeline_fixtures("streaming_reader");
    let reader = mi_io::sdf::SdfRecordReader::from_path(&sdf).unwrap();
    let titles: Vec<Option<String>> = reader
        .map(|record| record.unwrap().molecule().name().map(str::to_string))
        .collect();
    assert_eq!(
        titles,
        vec![
            Some("gdb9_1".into()),
            Some("gdb9_2".into()),
            Some("gdb9_3".into())
        ]
    );
}
