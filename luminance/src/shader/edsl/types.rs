//! Shader EDSL type information.

use std::iter::once;

use crate::shader::types::{Mat22, Mat33, Mat44, Vec2, Vec3, Vec4};

/// Type representation — akin to [`PrimType`] glued with array dimensions, if any.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Type {
  /// Primitive type, representing a type without array dimensions.
  prim_ty: PrimType,

  /// Array dimensions, if any.
  ///
  /// Dimensions are sorted from outer to inner; i.e. `[[i32; N]; M]`’s dimensions is encoded as `vec![M, N]`.
  array_dims: Vec<usize>,
}

/// Primitive supported types.
///
/// Types without array dimensions are known as _primitive types_ and are exhaustively constructed thanks to
/// [`PrimType`].
#[non_exhaustive]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum PrimType {
  /// An integral type.
  ///
  /// The [`Dim`] argument represents the vector dimension — do not confuse it with an array dimension.
  Int(Dim),

  /// An unsigned integral type.
  ///
  /// The [`Dim`] argument represents the vector dimension — do not confuse it with an array dimension.
  UInt(Dim),

  /// An floating type.
  ///
  /// The [`Dim`] argument represents the vector dimension — do not confuse it with an array dimension.
  Float(Dim),

  /// A boolean type.
  ///
  /// The [`Dim`] argument represents the vector dimension — do not confuse it with an array dimension.
  Bool(Dim),

  /// A N×M floating matrix.
  ///
  /// The [`MatrixDim`] provides the information required to know the exact dimension of the matrix.
  Matrix(MatrixDim),
}

/// Class of types that are recognized by the EDSL.
///
/// Any type implementing this type family is _representable_ in the EDSL.
pub trait ToPrimType {
  /// Mapped primitive type.
  const PRIM_TYPE: PrimType;
}

macro_rules! impl_ToPrimType {
  ($t:ty, $q:ident, $d:expr) => {
    impl ToPrimType for $t {
      const PRIM_TYPE: PrimType = PrimType::$q($d);
    }
  };
}

impl_ToPrimType!(i32, Int, Dim::Scalar);
impl_ToPrimType!(u32, UInt, Dim::Scalar);
impl_ToPrimType!(f32, Float, Dim::Scalar);
impl_ToPrimType!(bool, Bool, Dim::Scalar);
impl_ToPrimType!(Vec2<i32>, Int, Dim::D2);
impl_ToPrimType!(Vec2<u32>, UInt, Dim::D2);
impl_ToPrimType!(Vec2<f32>, Float, Dim::D2);
impl_ToPrimType!(Vec2<bool>, Bool, Dim::D2);
impl_ToPrimType!(Vec3<i32>, Int, Dim::D3);
impl_ToPrimType!(Vec3<u32>, UInt, Dim::D3);
impl_ToPrimType!(Vec3<f32>, Float, Dim::D3);
impl_ToPrimType!(Vec3<bool>, Bool, Dim::D3);
impl_ToPrimType!(Vec4<i32>, Int, Dim::D4);
impl_ToPrimType!(Vec4<u32>, UInt, Dim::D4);
impl_ToPrimType!(Vec4<f32>, Float, Dim::D4);
impl_ToPrimType!(Vec4<bool>, Bool, Dim::D4);
impl_ToPrimType!(Mat22<f32>, Matrix, MatrixDim::D22);
impl_ToPrimType!(Mat33<f32>, Matrix, MatrixDim::D33);
impl_ToPrimType!(Mat44<f32>, Matrix, MatrixDim::D44);

/// Represent a type (primitive type and array dimension) in the EDSL.
///
/// Any type implementing [`ToType`] is representable in the EDSL. Any type implementing [`ToPrimType`] automatically
/// also implements [`ToType`].
pub trait ToType {
  fn ty() -> Type;
}

impl<T> ToType for T
where
  T: ToPrimType,
{
  fn ty() -> Type {
    Type {
      prim_ty: T::PRIM_TYPE,
      array_dims: Vec::new(),
    }
  }
}

impl<T, const N: usize> ToType for [T; N]
where
  T: ToType,
{
  fn ty() -> Type {
    let Type {
      prim_ty,
      array_dims,
    } = T::ty();
    let array_dims = once(N).chain(array_dims).collect(); // FIXME: might require recursion?

    Type {
      prim_ty,
      array_dims,
    }
  }
}

/// Dimension of a primitive type.
///
/// Primitive types currently can have one of four dimension:
///
/// - [`Dim::Scalar`]: designates a scalar value.
/// - [`Dim::D2`]: designates a 2D vector.
/// - [`Dim::D3`]: designates a 3D vector.
/// - [`Dim::D4`]: designates a 4D vector.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Dim {
  /// Scalar value.
  Scalar,

  /// 2D vector.
  D2,

  /// 3D vector.
  D3,

  /// 4D vector.
  D4,
}

/// Matrix dimension.
///
/// Matrices can have several dimensions. Most of the time, you will be interested in squared dimensions, e.g. 2×2, 3×3
/// and 4×4. However, other dimensions exist.
///
/// > Note: matrices are expressed in column-major.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum MatrixDim {
  /// Squared 2 dimension.
  D22,
  /// 2×3 dimension.
  D23,
  /// 2×4 dimension.
  D24,
  /// 3×2 dimension.
  D32,
  /// Squared 3 dimension.
  D33,
  /// 3×4 dimension.
  D34,
  /// 4×2 dimension.
  D42,
  /// 4×3 dimension.
  D43,
  /// Squared 4 dimension.
  D44,
}
