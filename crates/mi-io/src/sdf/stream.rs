//! Streaming SDF record iteration.
//!
//! [`crate::sdf::parse_sdf`] materializes every record of a document in
//! memory, which is right for test fixtures and small files — and wrong
//! for real datasets like QM9, whose single SDF file carries ~134k records
//! (~1.3 GB). [`SdfRecordReader`] iterates records one at a time from any
//! [`BufRead`](std::io::BufRead) source: memory stays O(one record), and
//! every error still reports its 1-based file line because the reader
//! tracks positions exactly like the eager parser.

use super::parser::{SdfRecord, parse_record};
use crate::error::{IoError, IoResult};
use std::io::{BufRead, BufReader};

/// Iterates the records of an SDF document lazily.
///
/// Record boundaries follow the same grammar as [`parse_sdf`]
/// (`crate::sdf::parse_sdf`): `$$$$`-terminated blocks, a trailing record
/// without separator is accepted, whitespace-only blocks are skipped, and
/// an empty document yields no items — the caller decides whether that is
/// an error (the QM9 loader reports `IoError::EmptyInput`).
pub struct SdfRecordReader<R: BufRead> {
    reader: R,
    next_line_number: usize,
    finished: bool,
}

impl<R: BufRead> std::fmt::Debug for SdfRecordReader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The inner reader is not inspectable; report iteration state only.
        f.debug_struct("SdfRecordReader")
            .field("next_line_number", &self.next_line_number)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<S: std::io::Read> SdfRecordReader<BufReader<S>> {
    /// Wraps any byte source.
    pub fn new(reader: S) -> Self {
        Self {
            reader: BufReader::new(reader),
            next_line_number: 0,
            finished: false,
        }
    }
}

impl SdfRecordReader<BufReader<std::fs::File>> {
    /// Opens an SDF file on disk for streaming iteration.
    ///
    /// # Errors
    ///
    /// Returns [`IoError::File`] when the file cannot be opened.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> IoResult<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|error| IoError::File {
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
        Ok(Self::new(file))
    }
}

impl<R: BufRead> Iterator for SdfRecordReader<R> {
    type Item = IoResult<SdfRecord>;

    /// Reads the next record: lines up to the next `$$$$` separator (or
    /// end of input for a trailing record), parsed with the shared record
    /// grammar. Parse errors surface as `Err` items and **end the
    /// iteration** — a malformed record is a data condition, not
    /// something to silently skip; the QM9 loader applies its own
    /// record-level skip policy on top of successfully parsed records.
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let mut lines: Vec<(usize, String)> = Vec::new();
        let mut line = String::new();
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => {
                    // End of input: a trailing record without `$$$$`.
                    self.finished = true;
                    break;
                }
                Ok(_) => {
                    self.next_line_number += 1;
                    let number = self.next_line_number;
                    let text = line.trim_end_matches(['\n', '\r']).to_string();
                    if text.trim() == "$$$$" {
                        break;
                    }
                    // Blank lines between records are separator noise and are
                    // skipped; blank lines inside a record are structural
                    // (empty comment lines, data-field terminators) and stay.
                    if lines.is_empty() && text.trim().is_empty() {
                        continue;
                    }
                    lines.push((number - 1, text));
                }
                Err(error) => {
                    self.finished = true;
                    return Some(Err(IoError::InvalidLine {
                        line: self.next_line_number + 1,
                        reason: format!("I/O error while reading: {error}"),
                    }));
                }
            }
        }
        if lines.is_empty() {
            // Two consecutive separators (or separator + EOF): no record here.
            if self.finished {
                return None;
            }
            return self.next();
        }
        let borrowed: Vec<(usize, &str)> = lines
            .iter()
            .map(|(number, text)| (*number, text.as_str()))
            .collect();
        Some(parse_record(&borrowed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::IoError;

    const TWO_RECORDS: &str = "\
first
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0    0.0    0.0 H   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
second
  mi-io     0

  1  0  0  0  0  0  0  0  0  0999 V2000
    1.0    0.0    0.0 He  0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
";

    fn names(reader: &mut SdfRecordReader<BufReader<&[u8]>>) -> Vec<String> {
        reader
            .map(|record| record.unwrap().molecule().name().unwrap_or("").to_string())
            .collect()
    }

    #[test]
    fn iterates_records_lazily() {
        let mut reader = SdfRecordReader::new(TWO_RECORDS.as_bytes());
        assert_eq!(names(&mut reader), vec!["first", "second"]);
    }

    #[test]
    fn trailing_record_without_separator_is_yielded() {
        // Strip only the final separator: the second record then ends at EOF.
        let input = TWO_RECORDS.strip_suffix("$$$$\n").unwrap();
        let mut reader = SdfRecordReader::new(input.as_bytes());
        assert_eq!(names(&mut reader), vec!["first", "second"]);
    }

    #[test]
    fn whitespace_only_blocks_are_skipped() {
        let input = format!("\n\n{TWO_RECORDS}\n\n");
        let mut reader = SdfRecordReader::new(input.as_bytes());
        assert_eq!(names(&mut reader), vec!["first", "second"]);
    }

    #[test]
    fn empty_input_yields_no_records() {
        let mut reader = SdfRecordReader::new("".as_bytes());
        assert_eq!(reader.next(), None);
    }

    #[test]
    fn malformed_record_reports_file_line() {
        // The counts line is broken: the record parser must point at line 4.
        let input = "\
broken
  mi-io     0

  ??  ??
M  END
$$$$
";
        let mut reader = SdfRecordReader::new(input.as_bytes());
        let error = reader.next().unwrap().unwrap_err();
        assert!(
            matches!(error, IoError::InvalidCountsLine { line: 4, .. }),
            "unexpected: {error}"
        );
        // Iteration stops after a malformed record.
        assert_eq!(reader.next(), None);
    }

    #[test]
    fn from_path_reports_missing_files() {
        let error = SdfRecordReader::from_path("/nonexistent/qm9.sdf").unwrap_err();
        assert!(matches!(error, IoError::File { .. }));
    }
}
