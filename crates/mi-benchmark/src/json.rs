//! Minimal JSON value tree with a serializer for benchmark reports.
//!
//! Hand-rolled so the benchmark keeps zero dependencies beyond the
//! workspace; all report strings are project-controlled but are escaped
//! properly anyway.

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// A finite number (serialized via its `Display` form).
    Number(f64),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<Json>),
    /// An object with insertion-ordered keys.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Serializes compactly (no whitespace beyond string contents).
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Self::Number(value) => {
                if value.is_finite() {
                    out.push_str(&value.to_string());
                } else {
                    out.push_str("null");
                }
            }
            Self::String(value) => {
                out.push('"');
                for character in value.chars() {
                    match character {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        control if (control as u32) < 0x20 => {
                            out.push_str(&format!("\\u{:04x}", control as u32));
                        }
                        other => out.push(other),
                    }
                }
                out.push('"');
            }
            Self::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Self::Object(entries) => {
                out.push('{');
                for (index, (key, value)) in entries.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    Self::String(key.clone()).write(out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_serialize() {
        assert_eq!(Json::Null.to_json(), "null");
        assert_eq!(Json::Bool(true).to_json(), "true");
        assert_eq!(Json::Number(1.5).to_json(), "1.5");
        assert_eq!(Json::Number(f64::NAN).to_json(), "null");
    }

    #[test]
    fn strings_are_escaped() {
        let value = Json::String("a\"b\\c\nd\te\u{1}".into());
        assert_eq!(value.to_json(), "\"a\\\"b\\\\c\\nd\\te\\u0001\"");
    }

    #[test]
    fn nested_containers_serialize_in_order() {
        let value = Json::Object(vec![
            ("name".into(), Json::String("qm9".into())),
            (
                "counts".into(),
                Json::Array(vec![Json::Number(1.0), Json::Number(2.0)]),
            ),
            (
                "meta".into(),
                Json::Object(vec![("ok".into(), Json::Bool(false))]),
            ),
        ]);
        assert_eq!(
            value.to_json(),
            "{\"name\":\"qm9\",\"counts\":[1,2],\"meta\":{\"ok\":false}}"
        );
    }
}
