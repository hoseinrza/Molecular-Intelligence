//! Cartesian positions.

use crate::error::ChemistryError;

/// A Cartesian position in **Ångström** (x, y, z).
///
/// The unit is fixed at the type level: molecular file formats that carry
/// coordinates exchange them in Ångström (XYZ, MDL molfiles), so this crate
/// stores positions in Ångström and leaves unit conversion — e.g. to
/// Bohr for quantum chemistry — to explicit code at the boundary.
///
/// Only finite components can be constructed; NaN or infinite coordinates
/// are rejected, so every [`Position`] is safe to compare and serialize.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    x: f64,
    y: f64,
    z: f64,
}

impl Position {
    /// Creates a position from Cartesian components in Ångström.
    ///
    /// # Errors
    ///
    /// Returns [`ChemistryError::InvalidPosition`] if any component is NaN
    /// or infinite.
    pub fn new(x: f64, y: f64, z: f64) -> Result<Self, ChemistryError> {
        for (component, value) in [("x", x), ("y", y), ("z", z)] {
            if !value.is_finite() {
                return Err(ChemistryError::InvalidPosition { component, value });
            }
        }
        Ok(Self { x, y, z })
    }

    /// Returns the x component in Ångström.
    pub fn x(self) -> f64 {
        self.x
    }

    /// Returns the y component in Ångström.
    pub fn y(self) -> f64 {
        self.y
    }

    /// Returns the z component in Ångström.
    pub fn z(self) -> f64 {
        self.z
    }

    /// Returns all three components in Ångström as `(x, y, z)`.
    pub fn components(self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_round_trip() {
        let position = Position::new(1.25, -0.5, 0.0).unwrap();
        assert_eq!(position.components(), (1.25, -0.5, 0.0));
        assert_eq!(position.x(), 1.25);
        assert_eq!(position.y(), -0.5);
        assert_eq!(position.z(), 0.0);
    }

    #[test]
    fn non_finite_components_are_rejected() {
        for (x, y, z) in [
            (f64::NAN, 0.0, 0.0),
            (0.0, f64::INFINITY, 0.0),
            (0.0, 0.0, f64::NEG_INFINITY),
        ] {
            let error = Position::new(x, y, z).unwrap_err();
            assert!(
                error.to_string().contains("position"),
                "unexpected message: {error}"
            );
        }
    }

    #[test]
    fn positions_compare_by_value() {
        assert_eq!(Position::new(1.0, 2.0, 3.0), Position::new(1.0, 2.0, 3.0));
        assert_ne!(Position::new(1.0, 2.0, 3.0), Position::new(1.0, 2.0, 3.1));
    }
}
