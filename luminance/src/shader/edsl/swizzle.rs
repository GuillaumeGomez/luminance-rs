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
