//! Function scopes.

use crate::shader::edsl::builtins::BuiltIn;

/// Hierarchical and namespaced handle.
///
/// Handles live in different namespaces:
///
/// - The _built-in_ namespace gathers all built-ins.
/// - The _global_ namespace gathers everything that can be declared at top-level of a shader stage — i.e. mainly
///   constants for this namespace.
/// - The _input_ namespace gathers inputs.
/// - The _output_ namespace gathers outputs.
/// - The _function argument_ namespace gives handles to function arguments, which exist only in a function body.
/// - The _function variable_ namespace gives handles to variables defined in function bodies. This namespace is
/// hierarchical: for each scope, a new namespace is created. The depth at which a namespace is located is referred to
/// as its _subscope_.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ScopedHandle {
  BuiltIn(BuiltIn),
  Global(u16),
  FunArg(u16),
  FunVar { subscope: u16, handle: u16 },
  Output(String),  // FIXME: to switch to usize
  Uniform(String), // FIXME: to switch to usize

  // new type-sound representation
  Input(usize),
}
