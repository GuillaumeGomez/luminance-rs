//! Function scopes.

use std::{
  marker::PhantomData,
  ops::{Deref, DerefMut},
};

use crate::shader::edsl::{
  builtins::BuiltIn,
  expr::{ErasedExpr, Expr, Var},
  types::{ToType, Type},
};

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

impl ScopedHandle {
  /// Create a scoped handle representing a variable scoped to the current function at given depth.
  pub(crate) fn fun_var(subscope: u16, handle: u16) -> Self {
    Self::FunVar { subscope, handle }
  }
}

/// Lexical scope that must output a `R`.
///
/// Scopes are the only way to add control flow expressions to shaders. [`Scope<R>`] is the most general one, parent
/// of all scopes. Depending on the kind of control flow, several kinds of scopes are possible:
///
/// - [`Scope<R>`] is the most general one and every scopes share its features.
/// - [`EscapeScope<R>`] is a special kind of [`Scope<R>`] that allows escaping from anywhere in the scope.
/// - [`LoopScope<R>`] is a special kind of [`EscapeScope<R>`] that also allows to escape local looping expressions,
///   such as `for` and `while` loops.
///
/// A [`Scope<R>`] allows to perform a bunch of actions:
///
/// - Creating variable via [`Scope::var`]. Expressions of type [`Expr<T>`] where [`T: ToType`](ToType) are bound in a
///   [`Scope<R>`] via [`Scope::var`] and a [`Var<T>`] is returned, representing the bound variable.
/// - Variable mutation via [`Scope::set`]. Any [`Var<T>`] declared previously and still reachable in the current [`Scope`]
///   can be mutated.
/// - Introducing conditional statements with [`Scope::when`] and [`Scope::unless`].
/// - Introducing looping statements with [`Scope::loop_for`] and [`Scope::loop_while`].
#[derive(Debug)]
pub struct Scope<R> {
  pub erased: ErasedScope,
  _phantom: PhantomData<R>,
}

impl<R> Scope<R>
where
  Return: From<R>,
{
  pub(crate) fn new(depth: u16) -> Self {
    let erased = ErasedScope::new(depth);

    Scope {
      erased,
      _phantom: PhantomData,
    }
  }

  /// Create a new fresh scope under the current scope.
  pub(crate) fn deeper(&self) -> Self {
    Scope::new(self.erased.depth + 1)
  }

  /// Bind an expression to a variable in the current scope.
  ///
  /// `let v = s.var(e);` binds the `e` expression to `v` in the `s` [`Scope<T>`], and `e` must have type [`Expr<T>`]
  /// and `v` must be a [`Var<T>`], with [`T: ToType`](ToType).
  ///
  /// # Return
  ///
  /// The resulting [`Var<T>`] contains the representation of the binding in the EDSL and the actual binding is
  /// recorded in the current scope.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// let v = s.var(3.1415); // assign the literal 3.1415 to v
  /// let q = s.var(v * 2.); // assign v * 2. to q
  /// #   })
  /// # });
  /// ```
  pub fn var<T>(&mut self, init_value: impl Into<Expr<T>>) -> Var<T>
  where
    T: ToType,
  {
    let n = self.erased.next_var;
    let handle = ScopedHandle::fun_var(self.erased.depth, n);

    self.erased.next_var += 1;

    self.erased.instructions.push(ScopeInstr::VarDecl {
      ty: T::ty(),
      handle: handle.clone(),
      init_value: init_value.into().erased,
    });

    Var::new(handle)
  }

  /// For looping statement — `for`.
  ///
  /// `s.loop_for(i, |i| /* cond */, |i| /* fold */, |i| /* body */ )` inserts a looping statement into the EDSL
  /// representing a typical “for” loop. `i` is an [`Expr<T>`] satisfying [`T: ToType`](ToType) and is used as
  /// _initial_ value.
  ///
  /// In all the following closures, `i` refers to the initial value.
  ///
  /// The first `cond` closure must return an [`Expr<bool>`], representing the condition that is held until the loop
  /// exits. The second `fold` closure is a pure computation that must return an [`Expr<T>`] and that will be evaluated
  /// at the end of each iteration before the next check on `cond`. The last and third `body` closure is the body of the
  /// loop.
  ///
  /// The behaviors of the first two closures is important to understand. Those are akin to _filtering_ and _folding_.
  /// The closure returning the [`Expr<bool>`] is given the [`Expr<T>`] at each iteration and the second closure creates
  /// the new [`Expr<T>`] for the next iteration. Normally, people are used to write this pattern as `i++`, for
  /// instance, but in the case of our EDSL, it is more akin go `i + 1`, and this value is affected to a local
  /// accumulator hidden from the user.
  ///
  /// The [`LoopScope<R>`] argument to the `body` closure is a specialization of [`Scope<R>`] that allows breaking out
  /// of loops.
  ///
  /// # Examples
  ///
  /// ```
  /// use shades::{CanEscape as _, LoopScope, Scope, StageBuilder};
  ///
  /// StageBuilder::new_vertex_shader(|mut s, vertex| {
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     s.loop_for(0, |i| i.lt(10), |i| i + 1, |s: &mut LoopScope<()>, i| {
  ///       s.when(i.eq(5), |s: &mut LoopScope<()>| {
  ///         // when i == 5, abort from the main function
  ///         s.abort();
  ///       });
  ///     });
  ///   })
  /// });
  /// ```
  pub fn loop_for<T>(
    &mut self,
    init_value: impl Into<Expr<T>>,
    condition: impl FnOnce(&Expr<T>) -> Expr<bool>,
    iter_fold: impl FnOnce(&Expr<T>) -> Expr<T>,
    body: impl FnOnce(&mut LoopScope<R>, &Expr<T>),
  ) where
    T: ToType,
  {
    let mut scope = LoopScope::new(self.deeper());

    // bind the init value so that it’s available in all closures
    let init_var = scope.var(init_value);

    let condition = condition(&init_var);

    // generate the “post expr”, which is basically the free from of the third part of the for loop; people usually
    // set this to ++i, i++, etc., but in our case, the expression is to treat as a fold’s accumulator
    let post_expr = iter_fold(&init_var);

    body(&mut scope, &init_var);

    let scope = Scope::from(scope);
    self.erased.instructions.push(ScopeInstr::For {
      init_ty: T::ty(),
      init_handle: ScopedHandle::fun_var(scope.erased.depth, 0),
      init_expr: init_var.to_expr().erased,
      condition: condition.erased,
      post_expr: post_expr.erased,
      scope: scope.erased,
    });
  }

  /// While looping statement — `while`.
  ///
  /// `s.loop_while(cond, body)` inserts a looping statement into the EDSL representing a typical “while” loop.
  ///
  /// `cond` is an [`Expr<bool>`], representing the condition that is held until the loop exits. `body` is the content
  /// the loop will execute at each iteration.
  ///
  /// The [`LoopScope<R>`] argument to the `body` closure is a specialization of [`Scope<R>`] that allows breaking out
  /// of loops.
  ///
  /// # Examples
  ///
  /// ```
  /// use shades::{CanEscape as _, LoopScope, Scope, StageBuilder};
  ///
  /// StageBuilder::new_vertex_shader(|mut s, vertex| {
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     let i = s.var(10);
  ///
  ///     s.loop_while(i.lt(10), |s| {
  ///       s.set(&i, &i + 1);
  ///     });
  ///   })
  /// });
  /// ```
  pub fn loop_while(
    &mut self,
    condition: impl Into<Expr<bool>>,
    body: impl FnOnce(&mut LoopScope<R>),
  ) {
    let mut scope = LoopScope::new(self.deeper());
    body(&mut scope);

    self.erased.instructions.push(ScopeInstr::While {
      condition: condition.into().erased,
      scope: Scope::from(scope).erased,
    });
  }

  /// Mutate a variable in the current scope.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// let v = s.var(1); // v = 1
  /// s.set(&v, 10); // v = 10
  /// #   })
  /// # });
  /// ```
  pub fn set<T>(&mut self, var: impl Into<Var<T>>, value: impl Into<Expr<T>>) {
    self.erased.instructions.push(ScopeInstr::MutateVar {
      var: var.into().to_expr().erased,
      expr: value.into().erased,
    });
  }
}

#[derive(Debug, PartialEq)]
pub struct ErasedScope {
  depth: u16,
  instructions: Vec<ScopeInstr>,
  next_var: u16,
}

impl ErasedScope {
  fn new(depth: u16) -> Self {
    Self {
      depth,
      instructions: Vec::new(),
      next_var: 0,
    }
  }
}

#[derive(Debug, PartialEq)]
pub enum ScopeInstr {
  VarDecl {
    ty: Type,
    handle: ScopedHandle,
    init_value: ErasedExpr,
  },

  Return(ErasedReturn),

  Continue,

  Break,

  If {
    condition: ErasedExpr,
    scope: ErasedScope,
  },

  ElseIf {
    condition: ErasedExpr,
    scope: ErasedScope,
  },

  Else {
    scope: ErasedScope,
  },

  For {
    init_ty: Type,
    init_handle: ScopedHandle,
    init_expr: ErasedExpr,
    condition: ErasedExpr,
    post_expr: ErasedExpr,
    scope: ErasedScope,
  },

  While {
    condition: ErasedExpr,
    scope: ErasedScope,
  },

  MutateVar {
    var: ErasedExpr,
    expr: ErasedExpr,
  },
}

/// A special kind of [`Scope`] that can also escape expressions out of its parent scope.
#[derive(Debug)]
pub struct EscapeScope<R>(Scope<R>);

impl<R> From<EscapeScope<R>> for Scope<R> {
  fn from(s: EscapeScope<R>) -> Self {
    s.0
  }
}

impl<R> Deref for EscapeScope<R> {
  type Target = Scope<R>;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl<R> DerefMut for EscapeScope<R> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.0
  }
}

impl<R> EscapeScope<R>
where
  Return: From<R>,
{
  fn new(s: Scope<R>) -> Self {
    Self(s)
  }

  /// Early-return the current function with an expression.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::StageBuilder;
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::{CanEscape as _, Expr, Scope};
  ///
  /// let _fun = s.fun(|s: &mut Scope<Expr<i32>>, arg: Expr<i32>| {
  ///   // if arg is less than 10, early-return with 0
  ///   s.when(arg.lt(10), |s| {
  ///     s.leave(0);
  ///   });
  ///
  ///   arg
  /// });
  ///
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// #   })
  /// # });
  /// ```
  pub fn leave(&mut self, ret: impl Into<R>) {
    self
      .erased
      .instructions
      .push(ScopeInstr::Return(Return::from(ret.into()).erased));
  }
}

impl EscapeScope<()> {
  /// Early-abort the current function.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::StageBuilder;
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::{CanEscape as _, Expr, Scope};
  ///
  /// let _fun = s.fun(|s: &mut Scope<()>, arg: Expr<i32>| {
  ///   // if arg is less than 10, early-return with 0
  ///   s.when(arg.lt(10), |s| {
  ///     s.abort();
  ///   });
  ///
  ///   // do something else…
  /// });
  ///
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// #   })
  /// # });
  /// ```
  pub fn abort(&mut self) {
    self
      .erased
      .instructions
      .push(ScopeInstr::Return(ErasedReturn::Void));
  }
}

/// A special kind of [`EscapeScope`] that can also break loops.
#[derive(Debug)]
pub struct LoopScope<R>(EscapeScope<R>);

impl<R> From<LoopScope<R>> for Scope<R> {
  fn from(s: LoopScope<R>) -> Self {
    s.0.into()
  }
}

impl<R> Deref for LoopScope<R> {
  type Target = EscapeScope<R>;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl<R> DerefMut for LoopScope<R> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.0
  }
}

impl<R> LoopScope<R>
where
  Return: From<R>,
{
  fn new(s: Scope<R>) -> Self {
    Self(EscapeScope::new(s))
  }

  /// Break the current iteration of the nearest loop and continue to the next iteration.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// use shades::CanEscape as _;
  ///
  /// s.loop_while(true, |s| {
  ///   s.loop_continue();
  /// });
  /// #   })
  /// # });
  /// ```
  pub fn loop_continue(&mut self) {
    self.erased.instructions.push(ScopeInstr::Continue);
  }

  /// Break the nearest loop.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// use shades::CanEscape as _;
  ///
  /// s.loop_while(true, |s| {
  ///   s.loop_break();
  /// });
  /// #   })
  /// # });
  /// ```
  pub fn loop_break(&mut self) {
    self.erased.instructions.push(ScopeInstr::Break);
  }
}

/// Scopes allowing to enter conditional scopes.
///
/// Conditional scopes allow to break out of a function by early-return / aborting the function.
pub trait CanEscape<R>
where
  Return: From<R>,
{
  /// Scope type inside the scope of the conditional.
  type InnerScope;

  /// Conditional statement — `if`.
  ///
  /// `s.when(cond, |s: &mut EscapeScope<R>| { /* body */ })` inserts a conditional branch in the EDSL using the `cond`
  /// expression as truth and the passed closure as body to run when the represented condition is `true`. The
  /// [`EscapeScope<R>`] provides you with the possibility to escape and leave the function earlier, either by returning
  /// an expression or by aborting the function, depending on the value of `R`: `Expr<_>` allows for early-returns and
  /// `()` allows for aborting.
  ///
  /// # Return
  ///
  /// A [`When<R>`], authorizing the same escape rules with `R`. This object allows you to chain other conditional
  /// statements, commonly referred to as `else if` and `else` in common languages.
  ///
  /// Have a look at the documentation of [`When`] for further information.
  ///
  /// # Examples
  ///
  /// Early-return:
  ///
  /// ```
  /// use shades::{CanEscape as _, EscapeScope, Expr, Scope, StageBuilder, lit};
  ///
  /// StageBuilder::new_vertex_shader(|mut s, vertex| {
  ///   let f = s.fun(|s: &mut Scope<Expr<i32>>| {
  ///     s.when(lit!(1).lt(3), |s: &mut EscapeScope<Expr<i32>>| {
  ///       // do something in here
  ///
  ///       // early-return with 0; only possible if the function returns Expr<i32>
  ///       s.leave(0);
  ///     });
  ///
  ///     lit!(1)
  ///   });
  ///
  ///   s.main_fun(|s: &mut Scope<()>| {
  /// # #[cfg(feature = "fun-call")]
  ///     let x = s.var(f());
  ///   })
  /// });
  /// ```
  ///
  /// Aborting a function:
  ///
  /// ```
  /// use shades::{CanEscape as _, EscapeScope, Scope, StageBuilder, lit};
  ///
  /// StageBuilder::new_vertex_shader(|mut s, vertex| {
  ///   s.main_fun(|s: &mut Scope<()>| {
  ///     s.when(lit!(1).lt(3), |s: &mut EscapeScope<()>| {
  ///       // do something in here
  ///
  ///       // break the parent function by aborting; this is possible because the return type is ()
  ///       s.abort();
  ///     });
  ///   })
  /// });
  /// ```
  fn when<'a>(
    &'a mut self,
    condition: impl Into<Expr<bool>>,
    body: impl FnOnce(&mut Self::InnerScope),
  ) -> When<'a, R>;

  /// Complement form of [`Scope::when`].
  ///
  /// This method does the same thing as [`Scope::when`] but applies the [`Not::not`](std::ops::Not::not) operator on
  /// the condition first.
  fn unless<'a>(
    &'a mut self,
    condition: impl Into<Expr<bool>>,
    body: impl FnOnce(&mut Self::InnerScope),
  ) -> When<'a, R> {
    self.when(!condition.into(), body)
  }
}

impl<R> CanEscape<R> for Scope<R>
where
  Return: From<R>,
{
  type InnerScope = EscapeScope<R>;

  fn when<'a>(
    &'a mut self,
    condition: impl Into<Expr<bool>>,
    body: impl FnOnce(&mut Self::InnerScope),
  ) -> When<'a, R> {
    let mut scope = EscapeScope::new(self.deeper());
    body(&mut scope);

    self.erased.instructions.push(ScopeInstr::If {
      condition: condition.into().erased,
      scope: Scope::from(scope).erased,
    });

    When { parent_scope: self }
  }
}

impl<R> CanEscape<R> for LoopScope<R>
where
  Return: From<R>,
{
  type InnerScope = LoopScope<R>;

  fn when<'a>(
    &'a mut self,
    condition: impl Into<Expr<bool>>,
    body: impl FnOnce(&mut Self::InnerScope),
  ) -> When<'a, R> {
    let mut scope = LoopScope::new(self.deeper());
    body(&mut scope);

    self.erased.instructions.push(ScopeInstr::If {
      condition: condition.into().erased,
      scope: Scope::from(scope).erased,
    });

    When { parent_scope: self }
  }
}

/// Conditional combinator.
///
/// A [`When<R>`] is returned from functions such as [`CanEscape::when`] or [`CanEscape::unless`] and allows to continue
/// chaining conditional statements, encoding the concept of `else if` and `else` in more traditional languages.
#[derive(Debug)]
pub struct When<'a, R> {
  /// The scope from which this [`When`] expression comes from.
  ///
  /// This will be handy if we want to chain this when with others (corresponding to `else if` and `else`, for
  /// instance).
  parent_scope: &'a mut Scope<R>,
}

impl<R> When<'_, R>
where
  Return: From<R>,
{
  /// Add a conditional branch — `else if`.
  ///
  /// This method is often found chained after [`CanEscape::when`] and allows to add a new conditional if the previous
  /// conditional fails (i.e. `else if`). The behavior is the same as with [`CanEscape::when`].
  ///
  /// # Return
  ///
  /// Another [`When<R>`], allowing to add more conditional branches.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// use shades::{CanEscape as _, lit};
  ///
  /// let x = lit!(1);
  ///
  /// // you will need CanEscape in order to use when
  /// s.when(x.lt(2), |s| {
  ///   // do something if x < 2
  /// }).or_else(x.lt(10), |s| {
  ///   // do something if x < 10
  /// });
  /// #   })
  /// # });
  /// ```
  pub fn or_else(
    self,
    condition: impl Into<Expr<bool>>,
    body: impl FnOnce(&mut EscapeScope<R>),
  ) -> Self {
    let mut scope = EscapeScope::new(self.parent_scope.deeper());
    body(&mut scope);

    self
      .parent_scope
      .erased
      .instructions
      .push(ScopeInstr::ElseIf {
        condition: condition.into().erased,
        scope: Scope::from(scope).erased,
      });

    self
  }

  /// Add a final catch-all conditional branch — `else`.
  ///
  /// This method is often found chained after [`CanEscape::when`] and allows to finish the chain of conditional
  /// branches if the previous conditional fails (i.e. `else`). The behavior is the same as with [`CanEscape::when`].
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// use shades::{CanEscape as _, lit};
  ///
  /// let x = lit!(1);
  ///
  /// // you will need CanEscape in order to use when
  /// s.when(x.lt(2), |s| {
  ///   // do something if x < 2
  /// }).or(|s| {
  ///   // do something if x >= 2
  /// });
  /// #   })
  /// # });
  /// ```
  ///
  /// Can chain and mix conditional but [`When::or`] cannot be anywhere else but the end of the chain:
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// use shades::{CanEscape as _, lit};
  ///
  /// let x = lit!(1);
  ///
  /// // you will need CanEscape in order to use when
  /// s.when(x.lt(2), |s| {
  ///   // do something if x < 2
  /// }).or_else(x.lt(5), |s| {
  ///   // do something if x < 5
  /// }).or_else(x.lt(10), |s| {
  ///   // do something if x < 10
  /// }).or(|s| {
  ///   // else, do this
  /// });
  /// #   })
  /// # });
  /// ```
  pub fn or(self, body: impl FnOnce(&mut EscapeScope<R>)) {
    let mut scope = EscapeScope::new(self.parent_scope.deeper());
    body(&mut scope);

    self
      .parent_scope
      .erased
      .instructions
      .push(ScopeInstr::Else {
        scope: Scope::from(scope).erased,
      });
  }
}

/// Function return.
///
/// This type represents a function return and is used to annotate values that can be returned from functions (i.e.
/// expressions).
#[derive(Clone, Debug, PartialEq)]
pub struct Return {
  pub erased: ErasedReturn,
}

/// Erased return.
///
/// Either `Void` (i.e. `void`) or an expression. The type of the expression is also present for convenience.
#[derive(Clone, Debug, PartialEq)]
pub enum ErasedReturn {
  Void,
  Expr(Type, ErasedExpr),
}

impl From<()> for Return {
  fn from(_: ()) -> Self {
    Return {
      erased: ErasedReturn::Void,
    }
  }
}

impl<T> From<Expr<T>> for Return
where
  T: ToType,
{
  fn from(expr: Expr<T>) -> Self {
    Return {
      erased: ErasedReturn::Expr(T::ty(), expr.erased),
    }
  }
}
