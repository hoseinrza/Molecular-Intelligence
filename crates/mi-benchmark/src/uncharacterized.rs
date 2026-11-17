//! Parses QM9's `uncharacterized.txt` into the exclusion index set.
//!
//! The distribution's file is prose followed by a table whose data rows
//! carry the 1-based molecule index as the first token of the line
//! (verified against the 3054-row distribution file). The loader matches
//! exclusions by record position, which equals the index for a complete,
//! ordered file.

use std::collections::BTreeSet;
use std::path::Path;

/// Reads the exclusion file and returns the molecule indices to exclude.
///
/// # Errors
///
/// Returns an error message when the file cannot be read or holds no
/// parseable index rows (which would silently disable exclusions).
pub fn parse_uncharacterized(path: impl AsRef<Path>) -> Result<BTreeSet<usize>, String> {
    let text = std::fs::read_to_string(path.as_ref())
        .map_err(|error| format!("cannot read {}: {error}", path.as_ref().display()))?;
    let indices = parse_uncharacterized_text(&text);
    if indices.is_empty() {
        return Err(format!(
            "{} holds no index rows; refusing to run without exclusions",
            path.as_ref().display()
        ));
    }
    Ok(indices)
}

/// Parses the exclusion rows out of the file's text.
///
/// A line yields an index when its first whitespace-separated token is a
/// plain decimal number; prose headers and separator rules never start
/// with a number.
pub fn parse_uncharacterized_text(text: &str) -> BTreeSet<usize> {
    text.lines()
        .filter_map(|line| line.split_whitespace().next()?.parse::<usize>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rows_and_ignores_prose() {
        let text = "\
List of molecules among the 133885 GDB9 molecules for which the Corina
generated Cartesian coordinates differ from the equilibrium geometry.

================================================================================
#   Index                           GDB17 SMILES
================================================================================
      58                               NC(=N)C#N
      61                               NC(=N)C=O
";
        let indices = parse_uncharacterized_text(text);
        assert_eq!(indices, BTreeSet::from([58, 61]));
    }

    #[test]
    fn empty_text_yields_no_indices() {
        assert!(parse_uncharacterized_text("only prose\nno numbers\n").is_empty());
    }
}
