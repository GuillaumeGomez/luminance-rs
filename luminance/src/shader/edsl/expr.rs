//! Shader EDSL expressions.

use std::{marker::PhantomData, ops};

use crate::shader::{
  edsl::{
    fun::ErasedFunHandle,
    scope::ScopedHandle,
    swizzle::Swizzle,
    types::{ToType, Type},
  },
  types::{Mat22, Mat33, Mat44, Vec2, Vec3, Vec4},
};

/// Representation of an expression.
#[derive(Clone, Debug, PartialEq)]
pub enum ErasedExpr {
  // scalars
  LitInt(i32),
  LitUInt(u32),
  LitFloat(f32),
  LitBool(bool),
  // vectors
  LitInt2([i32; 2]),
  LitUInt2([u32; 2]),
  LitFloat2([f32; 2]),
  LitBool2([bool; 2]),
  LitInt3([i32; 3]),
  LitUInt3([u32; 3]),
  LitFloat3([f32; 3]),
  LitBool3([bool; 3]),
  LitInt4([i32; 4]),
  LitUInt4([u32; 4]),
  LitFloat4([f32; 4]),
  LitBool4([bool; 4]),
  // matrices
  LitM22(Mat22<f32>),
  // LitM23(M23),
  // LitM24(M24),
  // LitM32(M32),
  LitM33(Mat33<f32>),
  // LitM34(M34),
  // LitM42(M42),
  // LitM43(M43),
  LitM44(Mat44<f32>),
  // arrays
  Array(Type, Vec<ErasedExpr>),
  // var
  Var(ScopedHandle),
  // built-in functions and operators
  Not(Box<Self>),
  And(Box<Self>, Box<Self>),
  Or(Box<Self>, Box<Self>),
  Xor(Box<Self>, Box<Self>),
  BitOr(Box<Self>, Box<Self>),
  BitAnd(Box<Self>, Box<Self>),
  BitXor(Box<Self>, Box<Self>),
  Neg(Box<Self>),
  Add(Box<Self>, Box<Self>),
  Sub(Box<Self>, Box<Self>),
  Mul(Box<Self>, Box<Self>),
  Div(Box<Self>, Box<Self>),
  Rem(Box<Self>, Box<Self>),
  Shl(Box<Self>, Box<Self>),
  Shr(Box<Self>, Box<Self>),
  Eq(Box<Self>, Box<Self>),
  Neq(Box<Self>, Box<Self>),
  Lt(Box<Self>, Box<Self>),
  Lte(Box<Self>, Box<Self>),
  Gt(Box<Self>, Box<Self>),
  Gte(Box<Self>, Box<Self>),
  // function call
  FunCall(ErasedFunHandle, Vec<Self>),
  // swizzle
  Swizzle(Box<Self>, Swizzle),
  // field expression, as in a struct Foo { float x; }, foo.x is an Expr representing the x field on object foo
  Field { object: Box<Self>, field: Box<Self> },
  ArrayLookup { object: Box<Self>, index: Box<Self> },
}

/// Expression representation.
///
/// An expression is anything that carries a (typed) value and that can be combined in various ways with other
/// expressions. A literal, a constant or a variable are all expressions. The sum (as in `a + b`) of two expressions is
/// also an expression. A function call returning an expression is also an expression, as in `a * sin(b)`. Accessing an
/// element in an array (which is an expression as it carries items) via an index (an expression) is also an
/// expression — e.g. `levels[y * HEIGHT + x] * size`. The same thing applies to field access, swizzling, etc. etc.
///
/// On a general note, expressions are pretty central to the EDSL, as they are the _lower level concept_ you will be
/// able to manipulate. Expressions are side effect free, so a variable, for instance, can either be considered as an
/// expression or not. If `x` is a variable (see [`Var`]), then `x * 10` is an expression, but using `x` to mutate its
/// content does not make a use of `x` as an expression. It means that expressions are read-only and even though you
/// can go from higher constructs (like variables) to expressions, the opposite direction is forbidden.
///
/// # Literals
///
/// The most obvious kind of expression is a literal — e.g. `1`, `false`, `3.14`, etc. Any type `T` that defines an
/// implementor `From<T> for Expr<T>` can be used as literal. You can then use, for instance, `1.into()`,
/// `Expr::from(1)`, etc.
///
/// A much better and easier way to create literals is to use the [`lit!`](lit) macro, which basically does the lifting
/// for you, but also accept more forms to create more complex literals, such as scalar vectors. See its documentation
/// for further details.
///
/// It’s important to notice that because of how Rust infers type, type ambiguities might occur when using literals —
/// hence, the use of [`lit!`](lit) should help. For instance, in `1 + 2`, the type of `1` is ambiguous because of how
/// the implementors for [`Add`](std::ops::Add) are picked. In such a case, you are advised to use [`lit!`](lit).
///
/// ## Automatic lifting
///
/// Sometimes, you will want to pass literals to form other expressions, function calls, etc. Most of the API has been
/// written in a way that if no ambiguity would occur, then you can use the Rust type directly. For instance, if `x`
/// has the type `Expr<i32>`, then `x + 1` is the same as `x + lit!(1)`. You can use this property with literals too:
/// `lit!(1) + 2 + 3 + 4`.
///
/// That automatic lifting is valid for a lot of traits and methods throughout this crate.
///
/// # Expressions from side-effects
///
/// Some side-effects will create expressions, such as creating a variable or a constant. Most of the time, you
/// shouldn’t have to worry about the type of the expression as it should be inferred based on the side-effect.
///
/// # Expression macros
///
/// Some macros will create expressions for you, such as [`lit!`](lit), [`vec2!`](vec2), [`vec3!`](vec3) and
/// [`vec4!`](vec4) or the [`sw!`](sw) macros. Most of the time, those macros will work by automatically adding a
/// reference (`&`) to their arguments so that you don’t have to worry about that either.
#[derive(Debug)]
pub struct Expr<T>
where
  T: ?Sized,
{
  erased: ErasedExpr,
  _phantom: PhantomData<T>,
}

impl<T> From<&'_ Self> for Expr<T>
where
  T: ?Sized,
{
  fn from(e: &Self) -> Self {
    Self::new(e.erased.clone())
  }
}

impl<T> Clone for Expr<T>
where
  T: ?Sized,
{
  fn clone(&self) -> Self {
    Self::new(self.erased.clone())
  }
}

impl<T> Expr<T>
where
  T: ?Sized,
{
  /// Type an [`ErasedExpr`] and return it wrapped in [`Expr<T>`].
  const fn new(erased: ErasedExpr) -> Self {
    Self {
      erased,
      _phantom: PhantomData,
    }
  }

  /// Equality expression.
  ///
  /// This method builds an expression representing the equality between two expressions.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing the equality between the two input expressions.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::{lit, vec2};
  ///
  /// let _ = lit!(1).eq(1); // 1 == 1;
  /// let _ = vec2!(1., 2.).eq(vec2!(0., 0.)); // vec2(1., 2.) == vec2(0., 0.)
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  pub fn eq(&self, rhs: impl Into<Expr<T>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Eq(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  /// Inequality expression.
  ///
  /// This method builds an expression representing the inequality between two expressions.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing the inequality between the two input expressions.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::{lit, vec2};
  ///
  /// let _ = lit!(1).neq(1); // 1 != 1;
  /// let _ = vec2!(1., 2.).eq(vec2!(0., 0.)); // vec2(1., 2.) != vec2(0., 0.)
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  pub fn neq(&self, rhs: impl Into<Expr<T>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Neq(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }
}

/// Trait allowing to create 2D scalar vector ([`Vec2`]) constructors.
///
/// 2D scalar vectors can be created from either two sole scalars or a single 2D scalar vector (identity function).
///
/// The `A` type variable represents the arguments type. In the case of several arguments, tuples are used.
///
/// You are advised to use the [`vec2!`](vec2) macro instead as the interface of this function is not really
/// user-friendly.
pub trait IntoVec2<A> {
  /// Make a [`Vec2`] from `A`.
  fn vec2(args: A) -> Self;
}

impl<T> IntoVec2<(Expr<T>, Expr<T>)> for Expr<Vec2<T>> {
  fn vec2(args: (Expr<T>, Expr<T>)) -> Self {
    let (x, y) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec2,
      vec![x.erased, y.erased],
    ))
  }
}

/// Trait allowing to create 3D scalar vector ([`Vec3`]) constructors.
///
/// 3D scalar vectors can be created from either three sole scalars, a single 2D scalar vector with a single scalar or
/// a single 3D scalar vector (identity function).
///
/// The `A` type variable represents the arguments type. In the case of several arguments, tuples are used.
///
/// You are advised to use the [`vec3!`](vec3) macro instead as the interface of this function is not really
/// user-friendly.
pub trait IntoVec3<A> {
  /// Make a [`Vec3`] from `A`.
  fn vec3(args: A) -> Self;
}

impl<T> IntoVec3<(Expr<Vec2<T>>, Expr<T>)> for Expr<Vec3<T>> {
  fn vec3(args: (Expr<Vec2<T>>, Expr<T>)) -> Self {
    let (xy, z) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec3,
      vec![xy.erased, z.erased],
    ))
  }
}

impl<T> IntoVec3<(Expr<T>, Expr<T>, Expr<T>)> for Expr<Vec3<T>> {
  fn vec3(args: (Expr<T>, Expr<T>, Expr<T>)) -> Self {
    let (x, y, z) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec3,
      vec![x.erased, y.erased, z.erased],
    ))
  }
}

/// Trait allowing to create 4D scalar vector ([`Vec4`]) constructors.
///
/// 4D scalar vectors can be created from either four sole scalars, a single 3D scalar vector with a single scalar,
/// two 2D scalar vectors, a 2D scalar vector and a sole scalar or a single 4D scalar vector (identity function).
///
/// The `A` type variable represents the arguments type. In the case of several arguments, tuples are used.
///
/// You are advised to use the [`vec4!`](vec4) macro instead as the interface of this function is not really
/// user-friendly.
pub trait IntoVec4<A> {
  /// Make a [`Vec4`] from `A`.
  fn vec4(args: A) -> Self;
}

impl<T> IntoVec4<(Expr<Vec3<T>>, Expr<T>)> for Expr<Vec4<T>> {
  fn vec4(args: (Expr<Vec3<T>>, Expr<T>)) -> Self {
    let (xyz, w) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec4,
      vec![xyz.erased, w.erased],
    ))
  }
}

impl<T> IntoVec4<(Expr<Vec2<T>>, Expr<Vec2<T>>)> for Expr<Vec4<T>> {
  fn vec4(args: (Expr<Vec2<T>>, Expr<Vec2<T>>)) -> Self {
    let (xy, zw) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec4,
      vec![xy.erased, zw.erased],
    ))
  }
}

impl<'a, T> IntoVec4<(Expr<Vec2<T>>, Expr<T>, Expr<T>)> for Expr<Vec4<T>> {
  fn vec4(args: (Expr<Vec2<T>>, Expr<T>, Expr<T>)) -> Self {
    let (xy, z, w) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec4,
      vec![xy.erased, z.erased, w.erased],
    ))
  }
}

impl<'a, T> IntoVec4<(Expr<T>, Expr<T>, Expr<T>, Expr<T>)> for Expr<Vec4<T>> {
  fn vec4(args: (Expr<T>, Expr<T>, Expr<T>, Expr<T>)) -> Self {
    let (x, y, z, w) = args;
    Expr::new(ErasedExpr::FunCall(
      ErasedFunHandle::Vec4,
      vec![x.erased, y.erased, z.erased, w.erased],
    ))
  }
}

impl<T> Expr<T>
where
  T: PartialOrd,
{
  /// Less-than expression.
  ///
  /// This method builds an expression representing the binary operation `a < b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a < b`.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::lit;
  ///
  /// let _ = lit!(1).lt(2); // 1 < 2;
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  pub fn lt(&self, rhs: impl Into<Expr<T>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Lt(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  /// Less-than-or-equal expression.
  ///
  /// This method builds an expression representing the binary operation `a <= b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a <= b`.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::lit;
  ///
  /// let _ = lit!(1).lte(2); // 1 <= 2;
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  pub fn lte(&self, rhs: impl Into<Expr<T>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Lte(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  /// Greater-than expression.
  ///
  /// This method builds an expression representing the binary operation `a > b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a > b`.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::lit;
  ///
  /// let _ = lit!(1).gt(2); // 1 > 2;
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  pub fn gt(&self, rhs: impl Into<Expr<T>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Gt(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  /// Less-than-or-equal expression.
  ///
  /// This method builds an expression representing the binary operation `a <= b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a <= b`.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::lit;
  ///
  /// let _ = lit!(1).lte(2); // 1 <= 2;
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  pub fn gte(&self, rhs: impl Into<Expr<T>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Gte(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }
}

impl Expr<bool> {
  /// Logical _and_ expression.
  ///
  /// This method builds an expression representing the logical operation `a AND b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a AND b`.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::lit;
  ///
  /// let _ = lit!(true).and(false); // true && false
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  pub fn and(&self, rhs: impl Into<Expr<bool>>) -> Expr<bool> {
    Expr::new(ErasedExpr::And(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  /// Logical _or_ expression.
  ///
  /// This method builds an expression representing the logical operation `a OR b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a OR b`.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::lit;
  ///
  /// let _ = lit!(true).or(false); // true || false
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  pub fn or(&self, rhs: impl Into<Expr<bool>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Or(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }

  /// Logical _exclusive or_ expression.
  ///
  /// This method builds an expression representing the logical operation `a XOR b`.
  ///
  /// # Return
  ///
  /// An [`Expr<bool>`] representing `a XOR b`.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::lit;
  ///
  /// let _ = lit!(true).xor(false); // true ^^ false
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  pub fn xor(&self, rhs: impl Into<Expr<bool>>) -> Expr<bool> {
    Expr::new(ErasedExpr::Xor(
      Box::new(self.erased.clone()),
      Box::new(rhs.into().erased),
    ))
  }
}

impl<T> Expr<[T]> {
  /// Array lookup.
  ///
  /// The expression `a.at(i)` represents an _array lookup_, where `a` is an array — which type must be either
  /// [`Expr<[T]>`](Expr) or [`Expr<[T; N]>`](Expr) – and `i` is an [`Expr<i32>`].
  ///
  /// # Return
  ///
  /// The resulting [`Expr<T>`] represents the array lookup in `a` at index `i`.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::lit;
  ///
  /// let _ = lit!([1, 2, 3]).at(2); // [1, 2, 3][2]
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  pub fn at(&self, index: impl Into<Expr<i32>>) -> Expr<T> {
    Expr::new(ErasedExpr::ArrayLookup {
      object: Box::new(self.erased.clone()),
      index: Box::new(index.into().erased),
    })
  }
}

impl<T, const N: usize> Expr<[T; N]> {
  /// Array lookup.
  ///
  /// The expression `a.at(i)` represents an _array lookup_, where `a` is an array — which type must be either
  /// [`Expr<[T]>`](Expr) or [`Expr<[T; N]>`](Expr) – and `i` is an [`Expr<i32>`].
  ///
  /// # Return
  ///
  /// The resulting [`Expr<T>`] represents the array lookup in `a` at index `i`.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// use shades::lit;
  ///
  /// let _ = lit!([1, 2, 3]).at(2); // [1, 2, 3][2]
  /// # s.main_fun(|s: &mut Scope<()>| {})
  /// # });
  /// ```
  pub fn at(&self, index: impl Into<Expr<i32>>) -> Expr<T> {
    Expr::new(ErasedExpr::ArrayLookup {
      object: Box::new(self.erased.clone()),
      index: Box::new(index.into().erased),
    })
  }
}

// not
macro_rules! impl_Not_Expr {
  ($t:ty) => {
    impl ops::Not for Expr<$t> {
      type Output = Self;

      fn not(self) -> Self::Output {
        Expr::new(ErasedExpr::Not(Box::new(self.erased)))
      }
    }

    impl<'a> ops::Not for &'a Expr<$t> {
      type Output = Expr<$t>;

      fn not(self) -> Self::Output {
        Expr::new(ErasedExpr::Not(Box::new(self.erased.clone())))
      }
    }

    impl ops::Not for Var<$t> {
      type Output = Expr<$t>;

      fn not(self) -> Self::Output {
        Expr::new(ErasedExpr::Not(Box::new(self.0.erased)))
      }
    }

    impl<'a> ops::Not for &'a Var<$t> {
      type Output = Expr<$t>;

      fn not(self) -> Self::Output {
        Expr::new(ErasedExpr::Not(Box::new(self.0.erased.clone())))
      }
    }
  };
}

impl_Not_Expr!(bool);
impl_Not_Expr!(Vec2<bool>);
impl_Not_Expr!(Vec3<bool>);
impl_Not_Expr!(Vec4<bool>);

// neg
macro_rules! impl_Neg {
  ($t:ty) => {
    impl ops::Neg for Expr<$t> {
      type Output = Self;

      fn neg(self) -> Self::Output {
        Expr::new(ErasedExpr::Neg(Box::new(self.erased)))
      }
    }

    impl<'a> ops::Neg for &'a Expr<$t> {
      type Output = Expr<$t>;

      fn neg(self) -> Self::Output {
        Expr::new(ErasedExpr::Neg(Box::new(self.erased.clone())))
      }
    }

    impl ops::Neg for Var<$t> {
      type Output = Expr<$t>;

      fn neg(self) -> Self::Output {
        Expr::new(ErasedExpr::Neg(Box::new(self.0.erased)))
      }
    }

    impl<'a> ops::Neg for &'a Var<$t> {
      type Output = Expr<$t>;

      fn neg(self) -> Self::Output {
        Expr::new(ErasedExpr::Neg(Box::new(self.0.erased.clone())))
      }
    }
  };
}

impl_Neg!(i32);
impl_Neg!(Vec2<i32>);
impl_Neg!(Vec3<i32>);
impl_Neg!(Vec4<i32>);

impl_Neg!(u32);
impl_Neg!(Vec2<u32>);
impl_Neg!(Vec3<u32>);
impl_Neg!(Vec4<u32>);

impl_Neg!(f32);
impl_Neg!(Vec2<f32>);
impl_Neg!(Vec3<f32>);
impl_Neg!(Vec4<f32>);

// binary arithmetic and logical (+, -, *, /, %)
// binop
macro_rules! impl_binop_Expr {
  ($op:ident, $meth_name:ident, $a:ty, $b:ty) => {
    impl_binop_Expr!($op, $meth_name, $a, $b, $a);
  };

  ($op:ident, $meth_name:ident, $a:ty, $b:ty, $r:ty) => {
    // expr OP expr
    impl<'a> ops::$op<Expr<$b>> for Expr<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }

    // var OP expr
    impl<'a> ops::$op<Expr<$b>> for Var<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.0.erased),
          Box::new(rhs.erased),
        ))
      }
    }

    // expr OP var
    impl<'a> ops::$op<Var<$b>> for Expr<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: Var<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased),
          Box::new(rhs.0.erased),
        ))
      }
    }

    // var OP var
    impl<'a> ops::$op<Var<$b>> for Var<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: Var<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.0.erased),
          Box::new(rhs.0.erased),
        ))
      }
    }

    // expr OP &expr
    impl<'a> ops::$op<&'a Expr<$b>> for Expr<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: &'a Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    // var OP &expr
    impl<'a> ops::$op<&'a Expr<$b>> for Var<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: &'a Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.0.erased),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    // expr OP &var
    impl<'a> ops::$op<&'a Var<$b>> for Expr<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: &'a Var<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased),
          Box::new(rhs.0.erased.clone()),
        ))
      }
    }

    // var OP &var
    impl<'a> ops::$op<&'a Var<$b>> for Var<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: &'a Var<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.0.erased),
          Box::new(rhs.0.erased.clone()),
        ))
      }
    }

    // &expr OP expr
    impl<'a> ops::$op<Expr<$b>> for &'a Expr<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased),
        ))
      }
    }

    // &var OP expr
    impl<'a> ops::$op<Expr<$b>> for &'a Var<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.0.erased.clone()),
          Box::new(rhs.erased),
        ))
      }
    }

    // &expr OP var
    impl<'a> ops::$op<Var<$b>> for &'a Expr<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: Var<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.0.erased),
        ))
      }
    }

    // &var OP var
    impl<'a> ops::$op<Var<$b>> for &'a Var<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: Var<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.0.erased.clone()),
          Box::new(rhs.0.erased),
        ))
      }
    }

    // &expr OP &expr
    impl<'a> ops::$op<&'a Expr<$b>> for &'a Expr<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: &'a Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    // &var OP &expr
    impl<'a> ops::$op<&'a Expr<$b>> for &'a Var<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: &'a Expr<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.0.erased.clone()),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    // &expr OP &var
    impl<'a> ops::$op<&'a Var<$b>> for &'a Expr<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: &'a Var<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.0.erased.clone()),
        ))
      }
    }

    // &var OP &var
    impl<'a> ops::$op<&'a Var<$b>> for &'a Var<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: &'a Var<$b>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.0.erased.clone()),
          Box::new(rhs.0.erased.clone()),
        ))
      }
    }

    // expr OP t, where t is automatically lifted
    impl<'a> ops::$op<$b> for Expr<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: $b) -> Self::Output {
        let rhs = Expr::from(rhs);
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }

    // var OP t, where t is automatically lifted
    impl<'a> ops::$op<$b> for Var<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: $b) -> Self::Output {
        let rhs = Expr::from(rhs);
        Expr::new(ErasedExpr::$op(
          Box::new(self.0.erased),
          Box::new(rhs.erased),
        ))
      }
    }

    // &expr OP t, where t is automatically lifted
    impl<'a> ops::$op<$b> for &'a Expr<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: $b) -> Self::Output {
        let rhs: Expr<$b> = rhs.into();
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased),
        ))
      }
    }

    // &var OP t, where t is automatically lifted
    impl<'a> ops::$op<$b> for &'a Var<$a> {
      type Output = Expr<$r>;

      fn $meth_name(self, rhs: $b) -> Self::Output {
        let rhs: Expr<$b> = rhs.into();
        Expr::new(ErasedExpr::$op(
          Box::new(self.0.erased.clone()),
          Box::new(rhs.erased),
        ))
      }
    }
  };
}

// or
impl_binop_Expr!(BitOr, bitor, bool, bool);
impl_binop_Expr!(BitOr, bitor, Vec2<bool>, Vec2<bool>);
impl_binop_Expr!(BitOr, bitor, Vec2<bool>, bool);
impl_binop_Expr!(BitOr, bitor, Vec3<bool>, Vec3<bool>);
impl_binop_Expr!(BitOr, bitor, Vec3<bool>, bool);
impl_binop_Expr!(BitOr, bitor, Vec4<bool>, Vec4<bool>);
impl_binop_Expr!(BitOr, bitor, Vec4<bool>, bool);

// and
impl_binop_Expr!(BitAnd, bitand, bool, bool);
impl_binop_Expr!(BitAnd, bitand, Vec2<bool>, Vec2<bool>);
impl_binop_Expr!(BitAnd, bitand, Vec2<bool>, bool);
impl_binop_Expr!(BitAnd, bitand, Vec3<bool>, Vec3<bool>);
impl_binop_Expr!(BitAnd, bitand, Vec3<bool>, bool);
impl_binop_Expr!(BitAnd, bitand, Vec4<bool>, Vec4<bool>);
impl_binop_Expr!(BitAnd, bitand, Vec4<bool>, bool);

// xor
impl_binop_Expr!(BitXor, bitxor, bool, bool);
impl_binop_Expr!(BitXor, bitxor, Vec2<bool>, Vec2<bool>);
impl_binop_Expr!(BitXor, bitxor, Vec2<bool>, bool);
impl_binop_Expr!(BitXor, bitxor, Vec3<bool>, Vec3<bool>);
impl_binop_Expr!(BitXor, bitxor, Vec3<bool>, bool);
impl_binop_Expr!(BitXor, bitxor, Vec4<bool>, Vec4<bool>);
impl_binop_Expr!(BitXor, bitxor, Vec4<bool>, bool);

/// Run a macro on all supported types to generate the impl for them
///
/// The macro has to have to take two `ty` as argument and yield a `std::ops` trait implementor.
macro_rules! impl_binarith_Expr {
  ($op:ident, $meth_name:ident) => {
    impl_binop_Expr!($op, $meth_name, i32, i32);
    impl_binop_Expr!($op, $meth_name, Vec2<i32>, Vec2<i32>);
    impl_binop_Expr!($op, $meth_name, Vec2<i32>, i32);
    impl_binop_Expr!($op, $meth_name, Vec3<i32>, Vec3<i32>);
    impl_binop_Expr!($op, $meth_name, Vec3<i32>, i32);
    impl_binop_Expr!($op, $meth_name, Vec4<i32>, Vec4<i32>);
    impl_binop_Expr!($op, $meth_name, Vec4<i32>, i32);

    impl_binop_Expr!($op, $meth_name, u32, u32);
    impl_binop_Expr!($op, $meth_name, Vec2<u32>, Vec2<u32>);
    impl_binop_Expr!($op, $meth_name, Vec2<u32>, u32);
    impl_binop_Expr!($op, $meth_name, Vec3<u32>, Vec3<u32>);
    impl_binop_Expr!($op, $meth_name, Vec3<u32>, u32);
    impl_binop_Expr!($op, $meth_name, Vec4<u32>, Vec4<u32>);
    impl_binop_Expr!($op, $meth_name, Vec4<u32>, u32);

    impl_binop_Expr!($op, $meth_name, f32, f32);
    impl_binop_Expr!($op, $meth_name, Vec2<f32>, Vec2<f32>);
    impl_binop_Expr!($op, $meth_name, Vec2<f32>, f32);
    impl_binop_Expr!($op, $meth_name, Vec3<f32>, Vec3<f32>);
    impl_binop_Expr!($op, $meth_name, Vec3<f32>, f32);
    impl_binop_Expr!($op, $meth_name, Vec4<f32>, Vec4<f32>);
    impl_binop_Expr!($op, $meth_name, Vec4<f32>, f32);
  };
}

impl_binarith_Expr!(Add, add);
impl_binarith_Expr!(Sub, sub);
impl_binarith_Expr!(Mul, mul);
impl_binarith_Expr!(Div, div);

impl_binop_Expr!(Rem, rem, f32, f32);
impl_binop_Expr!(Rem, rem, Vec2<f32>, Vec2<f32>);
impl_binop_Expr!(Rem, rem, Vec2<f32>, f32);
impl_binop_Expr!(Rem, rem, Vec3<f32>, Vec3<f32>);
impl_binop_Expr!(Rem, rem, Vec3<f32>, f32);
impl_binop_Expr!(Rem, rem, Vec4<f32>, Vec4<f32>);
impl_binop_Expr!(Rem, rem, Vec4<f32>, f32);

impl_binop_Expr!(Mul, mul, Mat22<f32>, Mat22<f32>);
impl_binop_Expr!(Mul, mul, Mat22<f32>, Vec2<f32>, Vec2<f32>);
impl_binop_Expr!(Mul, mul, Vec2<f32>, Mat22<f32>, Mat22<f32>);
impl_binop_Expr!(Mul, mul, Mat33<f32>, Mat33<f32>);
impl_binop_Expr!(Mul, mul, Mat33<f32>, Vec3<f32>, Vec3<f32>);
impl_binop_Expr!(Mul, mul, Vec3<f32>, Mat33<f32>, Mat33<f32>);
impl_binop_Expr!(Mul, mul, Mat44<f32>, Mat44<f32>);
impl_binop_Expr!(Mul, mul, Mat44<f32>, Vec4<f32>, Vec4<f32>);
impl_binop_Expr!(Mul, mul, Vec4<f32>, Mat44<f32>, Mat44<f32>);

macro_rules! impl_binshift_Expr {
  ($op:ident, $meth_name:ident, $ty:ty) => {
    // expr OP expr
    impl ops::$op<Expr<u32>> for Expr<$ty> {
      type Output = Expr<$ty>;

      fn $meth_name(self, rhs: Expr<u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }

    impl<'a> ops::$op<Expr<u32>> for &'a Expr<$ty> {
      type Output = Expr<$ty>;

      fn $meth_name(self, rhs: Expr<u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased),
        ))
      }
    }

    impl<'a> ops::$op<&'a Expr<u32>> for Expr<$ty> {
      type Output = Expr<$ty>;

      fn $meth_name(self, rhs: &'a Expr<u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    impl<'a> ops::$op<&'a Expr<u32>> for &'a Expr<$ty> {
      type Output = Expr<$ty>;

      fn $meth_name(self, rhs: &'a Expr<u32>) -> Self::Output {
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased.clone()),
        ))
      }
    }

    // expr OP bits
    impl ops::$op<u32> for Expr<$ty> {
      type Output = Self;

      fn $meth_name(self, rhs: u32) -> Self::Output {
        let rhs = Expr::from(rhs);
        Expr::new(ErasedExpr::$op(Box::new(self.erased), Box::new(rhs.erased)))
      }
    }

    impl<'a> ops::$op<u32> for &'a Expr<$ty> {
      type Output = Expr<$ty>;

      fn $meth_name(self, rhs: u32) -> Self::Output {
        let rhs = Expr::from(rhs);
        Expr::new(ErasedExpr::$op(
          Box::new(self.erased.clone()),
          Box::new(rhs.erased),
        ))
      }
    }
  };
}

/// Binary shift generating macro.
macro_rules! impl_binshifts_Expr {
  ($op:ident, $meth_name:ident) => {
    impl_binshift_Expr!($op, $meth_name, i32);
    impl_binshift_Expr!($op, $meth_name, Vec2<i32>);
    impl_binshift_Expr!($op, $meth_name, Vec3<i32>);
    impl_binshift_Expr!($op, $meth_name, Vec4<i32>);

    impl_binshift_Expr!($op, $meth_name, u32);
    impl_binshift_Expr!($op, $meth_name, Vec2<u32>);
    impl_binshift_Expr!($op, $meth_name, Vec3<u32>);
    impl_binshift_Expr!($op, $meth_name, Vec4<u32>);

    impl_binshift_Expr!($op, $meth_name, f32);
    impl_binshift_Expr!($op, $meth_name, Vec2<f32>);
    impl_binshift_Expr!($op, $meth_name, Vec3<f32>);
    impl_binshift_Expr!($op, $meth_name, Vec4<f32>);
  };
}

impl_binshifts_Expr!(Shl, shl);
impl_binshifts_Expr!(Shr, shr);

macro_rules! impl_From_Expr_scalar {
  ($t:ty, $q:ident) => {
    impl From<$t> for Expr<$t> {
      fn from(a: $t) -> Self {
        Self::new(ErasedExpr::$q(a))
      }
    }

    impl<'a> From<&'a $t> for Expr<$t> {
      fn from(a: &'a $t) -> Self {
        Self::new(ErasedExpr::$q(*a))
      }
    }
  };
}

impl_From_Expr_scalar!(i32, LitInt);
impl_From_Expr_scalar!(u32, LitUInt);
impl_From_Expr_scalar!(f32, LitFloat);
impl_From_Expr_scalar!(bool, LitBool);

macro_rules! impl_From_Expr_vn {
  ($t:ty, $q:ident) => {
    impl From<$t> for Expr<$t> {
      fn from(a: $t) -> Self {
        Self::new(ErasedExpr::$q(a.0))
      }
    }

    impl<'a> From<&'a $t> for Expr<$t> {
      fn from(a: &'a $t) -> Self {
        Self::new(ErasedExpr::$q(a.0))
      }
    }
  };
}

impl_From_Expr_vn!(Vec2<i32>, LitInt2);
impl_From_Expr_vn!(Vec2<u32>, LitUInt2);
impl_From_Expr_vn!(Vec2<f32>, LitFloat2);
impl_From_Expr_vn!(Vec2<bool>, LitBool2);
impl_From_Expr_vn!(Vec3<i32>, LitInt3);
impl_From_Expr_vn!(Vec3<u32>, LitUInt3);
impl_From_Expr_vn!(Vec3<f32>, LitFloat3);
impl_From_Expr_vn!(Vec3<bool>, LitBool3);
impl_From_Expr_vn!(Vec4<i32>, LitInt4);
impl_From_Expr_vn!(Vec4<u32>, LitUInt4);
impl_From_Expr_vn!(Vec4<f32>, LitFloat4);
impl_From_Expr_vn!(Vec4<bool>, LitBool4);

impl<T, const N: usize> From<[T; N]> for Expr<[T; N]>
where
  Expr<T>: From<T>,
  T: Clone + ToType,
{
  fn from(array: [T; N]) -> Self {
    let array = array
      .iter()
      .cloned()
      .map(|t| Expr::from(t).erased)
      .collect();
    Self::new(ErasedExpr::Array(<[T; N] as ToType>::ty(), array))
  }
}

impl<'a, T, const N: usize> From<&'a [T; N]> for Expr<[T; N]>
where
  Expr<T>: From<T>,
  T: Clone + ToType,
{
  fn from(array: &'a [T; N]) -> Self {
    let array = array
      .iter()
      .cloned()
      .map(|t| Expr::from(t).erased)
      .collect();
    Self::new(ErasedExpr::Array(<[T; N] as ToType>::ty(), array))
  }
}

impl<T, const N: usize> From<[Expr<T>; N]> for Expr<[T; N]>
where
  Expr<T>: From<T>,
  T: ToType,
{
  fn from(array: [Expr<T>; N]) -> Self {
    let array = array.iter().cloned().map(|e| e.erased).collect();
    Self::new(ErasedExpr::Array(<[T; N] as ToType>::ty(), array))
  }
}

impl<'a, T, const N: usize> From<&'a [Expr<T>; N]> for Expr<[T; N]>
where
  Expr<T>: From<T>,
  T: ToType,
{
  fn from(array: &'a [Expr<T>; N]) -> Self {
    let array = array.iter().cloned().map(|e| e.erased).collect();
    Self::new(ErasedExpr::Array(<[T; N] as ToType>::ty(), array))
  }
}

/// Create various forms of literal expressions.
///
/// This macro allows you to create _literal expressions_ by lifting Rust constants into the EDSL. The way this is done
/// is via several forms:
///
/// - `lit!(x)` lifts a single Rust expression into the EDSL. It’s isomorphic to `Expr::from(x)`.
/// - `lit!(x, y)` lifts two Rust expressions into the EDSL as a 2D scalar vector. It’s isomorphic to
///   `Expr::from(V2::from([x, y]))`.
/// - `lit!(x, y, z)` lifts three Rust expressions into the EDSL as a 3D scalar vector. It’s isomorphic to
///   `Expr::from(V3::from([x, y, z]))`.
/// - `lit!(x, y, z, w)` lifts three Rust expressions into the EDSL as a 3D scalar vector. It’s isomorphic to
///   `Expr::from(V4::from([x, y, z, w]))`.
///
/// Most of the time, type inference will kick in and you shouldn’t have to annotate the return expression.
///
/// # Examples
///
/// ```
/// # use shades::{Scope, StageBuilder};
/// # StageBuilder::new_vertex_shader(|mut s, vertex| {
/// use shades::{lit};
///
/// let _ = lit!(1);
/// let _ = lit!(false);
/// let _ = lit!(1., 2., 3., 4.);
/// # s.main_fun(|s: &mut Scope<()>| {})
/// # });
/// ```
#[macro_export]
macro_rules! lit {
  ($e:expr) => {
    $crate::Expr::from($e)
  };

  ($a:expr, $b:expr) => {
    $crate::Expr::from($crate::V2::from([$a, $b]))
  };

  ($a:expr, $b:expr, $c:expr) => {
    $crate::Expr::from($crate::V3::from([$a, $b, $c]))
  };

  ($a:expr, $b:expr, $c:expr, $d:expr) => {
    $crate::Expr::from($crate::V4::from([$a, $b, $c, $d]))
  };

  ([ $($item:expr),* ]) => {
    $crate::Expr::from([ $($item),* ])
  }
}

/// Create 2D scalar vectors via different forms.
///
/// This macro allows to create 2D ([`V2`]) scalar vectors from two forms:
///
/// - `vec2!(xy)`, which acts as the cast operator. Only types `T` satisfying [`Vec2`] are castable.
/// - `vec2!(x, y)`, which builds a [`V2<T>`] for `x: T` and `y: T`.
///
/// # Examples
///
/// ```
/// # use shades::{Scope, StageBuilder};
/// # StageBuilder::new_vertex_shader(|mut s, vertex| {
/// use shades::vec2;
///
/// let _ = vec2!(1, 2);
/// # s.main_fun(|s: &mut Scope<()>| {})
/// # });
/// ```
#[macro_export]
macro_rules! vec2 {
  ($a:expr) => {
    todo!("vec2 cast operator missing");
  };

  ($xy:expr, $z:expr) => {{
    use $crate::Vec2 as _;
    $crate::Expr::vec2(($crate::Expr::from(&$xy), $crate::Expr::from(&$z)))
  }};
}

/// Create 3D scalar vectors via different forms.
///
/// This macro allows to create 3D ([`V3`]) scalar vectors from several forms:
///
/// - `vec3!(xyz)`, which acts as the cast operator. Only types `T` satisfying [`Vec3`] are castable.
/// - `vec3!(xy, z)`, which builds a [`V3<T>`] with `xy` a value that can be turned into a `Expr<V2<T>>` and `z: T`
/// - `vec3!(x, y, z)`, which builds a [`V3<T>`] for `x: T`, `y: T` and `z: T`.
///
/// # Examples
///
/// ```
/// # use shades::{Scope, StageBuilder};
/// # StageBuilder::new_vertex_shader(|mut s, vertex| {
/// use shades::{vec2, vec3};
///
/// let _ = vec3!(1, 2, 3);
/// let _ = vec3!(vec2!(1, 2), 3);
/// # s.main_fun(|s: &mut Scope<()>| {})
/// # });
/// ```
#[macro_export]
macro_rules! vec3 {
  ($a:expr) => {
    todo!("vec3 cast operator missing");
  };

  ($xy:expr, $z:expr) => {{
    use $crate::Vec3 as _;
    $crate::Expr::vec3(($crate::Expr::from(&$xy), $crate::Expr::from(&$z)))
  }};

  ($x:expr, $y:expr, $z:expr) => {{
    use $crate::Vec3 as _;
    $crate::Expr::vec3((
      $crate::Expr::from(&$x),
      $crate::Expr::from(&$y),
      $crate::Expr::from(&$z),
    ))
  }};
}

/// Create 4D scalar vectors via different forms.
///
/// This macro allows to create 4D ([`V4`]) scalar vectors from several forms:
///
/// - `vec4!(xyzw)`, which acts as the cast operator. Only types `T` satisfying [`Vec4`] are castable.
/// - `vec4!(xyz, w)`, which builds a [`V4<T>`] with `xyz` a value that can be turned into a `Expr<V3<T>>` and `w: T`.
/// - `vec4!(xy, zw)`, which builds a [`V4<T>`] with `xy` and `zw` values that can be turned into `Expr<V3<T>>`.
/// - `vec4!(xy, z, w)`, which builds a [`V4<T>`] with `xy`, `z: T` and `w: T`.
/// - `vec4!(x, y, z, w)`, which builds a [`V3<T>`] for `x: T`, `y: T` and `z: T`.
///
/// # Examples
///
/// ```
/// # use shades::{Scope, StageBuilder};
/// # StageBuilder::new_vertex_shader(|mut s, vertex| {
/// use shades::{vec2, vec3, vec4};
///
/// let _ = vec4!(1, 2, 3, 4);
/// let _ = vec4!(vec3!(1, 2, 3), 4);
/// let _ = vec4!(vec2!(1, 2), vec2!(3, 4));
/// let _ = vec4!(vec2!(1, 2), 3, 4);
/// # s.main_fun(|s: &mut Scope<()>| {})
/// # });
/// ```
#[macro_export]
macro_rules! vec4 {
  ($a:expr) => {
    todo!("vec4 cast operator missing");
  };

  ($xy:expr, $zw:expr) => {{
    use $crate::Vec4 as _;
    $crate::Expr::vec4(($crate::Expr::from(&$xy), $crate::Expr::from(&$zw)))
  }};

  ($xy:expr, $z:expr, $w:expr) => {{
    use $crate::Vec4 as _;
    $crate::Expr::vec4((
      $crate::Expr::from(&$xy),
      $crate::Expr::from(&$z),
      $crate::Expr::from(&$w),
    ))
  }};

  ($x:expr, $y:expr, $z:expr, $w:expr) => {{
    use $crate::Vec4 as _;
    $crate::Expr::vec4((
      $crate::Expr::from(&$x),
      $crate::Expr::from(&$y),
      $crate::Expr::from(&$z),
      $crate::Expr::from(&$w),
    ))
  }};
}

/// Mutable variable.
///
/// A [`Var<T>`] is akin to an [`Expr<T>`] that can be mutated. You can go from a [`Var<T>`] to an [`Expr<T>`] via
/// either the [`From`] or [`Var::to_expr`] method.
///
/// Variables, because they allow mutations, allow to write more complicated shader functions. Also, lots of graphics
/// pipelines’ properties are variables you will have to write to, such as [`VertexShaderEnv::position`].
#[derive(Debug)]
pub struct Var<T>(Expr<T>)
where
  T: ?Sized;

impl<'a, T> From<&'a Var<T>> for Var<T>
where
  T: ?Sized,
{
  fn from(v: &'a Self) -> Self {
    Var(v.0.clone())
  }
}

impl<T> From<Var<T>> for Expr<T>
where
  T: ?Sized,
{
  fn from(v: Var<T>) -> Self {
    v.0
  }
}

impl<'a, T> From<&'a Var<T>> for Expr<T>
where
  T: ?Sized,
{
  fn from(v: &'a Var<T>) -> Self {
    v.0.clone()
  }
}

impl<T> Var<T>
where
  T: ?Sized,
{
  /// Create a new [`Var<T>`] from a [`ScopedHandle`].
  const fn new(handle: ScopedHandle) -> Self {
    Self(Expr::new(ErasedExpr::Var(handle)))
  }

  /// Coerce [`Var<T>`] into [`Expr<T>`].
  ///
  /// > Note: use this function only when necessary. Lots of functions will accept both [`Expr<T>`] and [`Var<T>`],
  /// > performing the coercion for you automatically.
  ///
  /// # Return
  ///
  /// The expression representation of [`Var<T>`], allowing to pass the variable to functions or expressions that don’t
  /// easily coerce it automatically to [`Expr<T>`] already.
  ///
  /// # Examples
  ///
  /// ```
  /// # use shades::{Scope, StageBuilder};
  /// # StageBuilder::new_vertex_shader(|mut s, vertex| {
  /// #   s.main_fun(|s: &mut Scope<()>| {
  /// let v = s.var(123); // Var<i32>
  /// let e = v.to_expr(); // Expr<i32>
  /// #   })
  /// # });
  /// ```
  pub fn to_expr(&self) -> Expr<T> {
    self.0.clone()
  }
}

impl<T> Var<[T]> {
  pub fn at(&self, index: impl Into<Expr<i32>>) -> Var<T> {
    Var(self.to_expr().at(index))
  }
}

impl<T, const N: usize> Var<[T; N]> {
  pub fn at(&self, index: impl Into<Expr<i32>>) -> Var<T> {
    Var(self.to_expr().at(index))
  }
}

impl<T> ops::Deref for Var<T>
where
  T: ?Sized,
{
  type Target = Expr<T>;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl From<Mat22<f32>> for Expr<Mat22<f32>> {
  fn from(matrix: Mat22<f32>) -> Self {
    Self::new(ErasedExpr::LitM22(matrix))
  }
}
impl From<Mat33<f32>> for Expr<Mat33<f32>> {
  fn from(matrix: Mat33<f32>) -> Self {
    Self::new(ErasedExpr::LitM33(matrix))
  }
}
impl From<Mat44<f32>> for Expr<Mat44<f32>> {
  fn from(matrix: Mat44<f32>) -> Self {
    Self::new(ErasedExpr::LitM44(matrix))
  }
}
