//! Swizzle interface.

/// Select a channel to extract from into a swizzled expession.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SwizzleSelector {
  /// Select the `.x` (or `.r`) channel.
  X,

  /// Select the `.y` (or `.g`) channel.
  Y,

  /// Select the `.z` (or `.b`) channel.
  Z,

  /// Select the `.w` (or `.a`) channel.
  W,
}

/// Swizzle channel selector.
///
/// This type gives the dimension of the target expression (output) and dimension of the source expression (input). The
/// [`SwizzleSelector`] also to select a specific channel in the input expression.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Swizzle {
  /// Create a one-channel expression.
  D1(SwizzleSelector),

  /// Create a two-channel expression.
  D2(SwizzleSelector, SwizzleSelector),

  /// Create a three-channel expression.
  D3(SwizzleSelector, SwizzleSelector, SwizzleSelector),

  /// Create a four-channel expression.
  D4(
    SwizzleSelector,
    SwizzleSelector,
    SwizzleSelector,
    SwizzleSelector,
  ),
}

/// Interface to implement to swizzle an expression.
///
/// If you plan to use your implementor with the [`sw!`](sw) macro, `S` must be one of the following types:
///
/// - [`SwizzleSelector`]: to implement `sw!(.x)`.
/// - [[`SwizzleSelector`]; 2]: to implement `sw!(.xx)`.
/// - [[`SwizzleSelector`]; 3]: to implement `sw!(.xxx)`.
/// - [[`SwizzleSelector`]; 4]: to implement `sw!(.xxxx)`.
pub trait Swizzlable<S> {
  type Output;

  fn swizzle(&self, sw: S) -> Self::Output;
}

/// Expressions having a `x` or `r` coordinate.
///
/// Akin to swizzling with `.x` or `.r`, but easier.
pub trait HasX {
  type Output;

  fn x(&self) -> Self::Output;
  fn r(&self) -> Self::Output {
    self.x()
  }
}

/// Expressions having a `y` or `g` coordinate.
///
/// Akin to swizzling with `.y` or `.g`, but easier.
pub trait HasY {
  type Output;

  fn y(&self) -> Self::Output;
  fn g(&self) -> Self::Output {
    self.y()
  }
}

/// Expressions having a `z` or `b` coordinate.
///
/// Akin to swizzling with `.z` or `.b`, but easier.
pub trait HasZ {
  type Output;

  fn z(&self) -> Self::Output;
  fn b(&self) -> Self::Output {
    self.z()
  }
}

/// Expressions having a `w` or `a` coordinate.
///
/// Akin to swizzling with `.w` or `.a`, but easier.
pub trait HasW {
  type Output;

  fn w(&self) -> Self::Output;
  fn a(&self) -> Self::Output {
    self.w()
  }
}
