/// Numeric element types considered by the tensor layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float32,
    Float64,
    Int32,
    Int64,
}
