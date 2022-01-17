//! Shader functions.

use std::marker::PhantomData;

use crate::shader::edsl::{
  expr::{ErasedExpr, Expr},
  scope::{ErasedReturn, ErasedScope, Return, Scope},
  types::{ToType, Type},
};

/// Erased function handle.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ErasedFunHandle {
  // cast operators
  Vec2,
  Vec3,
  Vec4,
  // trigonometry
  Radians,
  Degrees,
  Sin,
  Cos,
  Tan,
  ASin,
  ACos,
  ATan,
  SinH,
  CosH,
  TanH,
  ASinH,
  ACosH,
  ATanH,
  // exponential
  Pow,
  Exp,
  Exp2,
  Log,
  Log2,
  Sqrt,
  InverseSqrt,
  // common
  Abs,
  Sign,
  Floor,
  Trunc,
  Round,
  RoundEven,
  Ceil,
  Fract,
  Min,
  Max,
  Clamp,
  Mix,
  Step,
  SmoothStep,
  IsNan,
  IsInf,
  FloatBitsToInt,
  IntBitsToFloat,
  UIntBitsToFloat,
  FMA,
  Frexp,
  Ldexp,
  // floating-point pack and unpack functions
  PackUnorm2x16,
  PackSnorm2x16,
  PackUnorm4x8,
  PackSnorm4x8,
  UnpackUnorm2x16,
  UnpackSnorm2x16,
  UnpackUnorm4x8,
  UnpackSnorm4x8,
  PackHalf2x16,
  UnpackHalf2x16,
  // geometry functions
  Length,
  Distance,
  Dot,
  Cross,
  Normalize,
  FaceForward,
  Reflect,
  Refract,
  // matrix functions
  // TODO
  // vector relational functions
  VLt,
  VLte,
  VGt,
  VGte,
  VEq,
  VNeq,
  VAny,
  VAll,
  VNot,
  // integer functions
  UAddCarry,
  USubBorrow,
  UMulExtended,
  IMulExtended,
  BitfieldExtract,
  BitfieldInsert,
  BitfieldReverse,
  BitCount,
  FindLSB,
  FindMSB,
  // texture functions
  // TODO
  // geometry shader functions
  EmitStreamVertex,
  EndStreamPrimitive,
  EmitVertex,
  EndPrimitive,
  // fragment processing functions
  DFDX,
  DFDY,
  DFDXFine,
  DFDYFine,
  DFDXCoarse,
  DFDYCoarse,
  FWidth,
  FWidthFine,
  FWidthCoarse,
  InterpolateAtCentroid,
  InterpolateAtSample,
  InterpolateAtOffset,
  // shader invocation control functions
  Barrier,
  MemoryBarrier,
  MemoryBarrierAtomic,
  MemoryBarrierBuffer,
  MemoryBarrierShared,
  MemoryBarrierImage,
  GroupMemoryBarrier,
  // shader invocation group functions
  AnyInvocation,
  AllInvocations,
  AllInvocationsEqual,
  UserDefined(u16),
}

/// A function definition.
///
/// Function definitions contain the information required to know how to represent a function’s arguments, return type
/// and its body.
#[derive(Debug)]
pub struct FunDef<Ret, Args> {
  erased: ErasedFun,
  _phantom: PhantomData<(Ret, Args)>,
}

impl<Ret, Args> FunDef<Ret, Args> {
  fn new(erased: ErasedFun) -> Self {
    Self {
      erased,
      _phantom: PhantomData,
    }
  }
}

/// Erased function definition, which is basically a scope along with a return expression.
#[derive(Debug)]
struct ErasedFun {
  args: Vec<Type>,
  scope: ErasedScope,
  ret: ErasedReturn,
}

impl ErasedFun {
  fn new(args: Vec<Type>, scope: ErasedScope, ret: ErasedReturn) -> Self {
    Self { args, scope, ret }
  }
}

/// Function definition injection.
///
/// This trait represents _function definition injection_, i.e. types that can provide a function definition — see
/// [`FunDef`]. Ideally, only a very small number of types can do this: polymorphic types implementing the [`FnOnce`]
/// trait with different number of arguments. Namely, closures / lambdas with various numbers of arguments.
///
/// You are not supposed to implement this type by yourself. Instead, when creating functions in the EDSL, you just have
/// to pass lambdas to automatically get the proper function definition lifted into the EDSL.
///
/// See the [`StageBuilder::fun`] for further information.
///
/// # Caveats
///
/// This way of doing currently comes with a price: type inference is bad. You will — most of the time — have to
/// annotate the closure’s arguments. This is currently working on but progress on that matter is slow.
pub trait ToFun<Ret, Args> {
  fn build_fn(self) -> FunDef<Ret, Args>;
}

/// Functions with no argument.
impl<F, Ret> ToFun<Ret, ()> for F
where
  Self: FnOnce(&mut Scope<Ret>) -> Ret,
  Return: From<Ret>,
{
  fn build_fn(self) -> FunDef<Ret, ()> {
    let mut scope = Scope::new(0);
    let ret = self(&mut scope);

    let erased = ErasedFun::new(Vec::new(), scope.erased, Return::from(ret).erased);

    FunDef::new(erased)
  }
}

/// 1 argument functions.
impl<F, R, A> ToFun<R, Expr<A>> for F
where
  Self: FnOnce(&mut Scope<R>, Expr<A>) -> R,
  Return: From<R>,
  A: ToType,
{
  fn build_fn(self) -> FunDef<R, Expr<A>> {
    let mut scope = Scope::new(0);
    let arg = Expr::fun_arg(0);
    let ret = self(&mut scope, arg);
    let erased = ErasedFun::new(vec![A::ty()], scope.erased, Return::from(ret).erased);

    FunDef::new(erased)
  }
}

/// 1+ arity functions.
macro_rules! impl_ToFun_args {
  ($($arg:ident , $arg_ident:ident , $arg_rank:expr),*) => {
    impl<F, R, $($arg),*> ToFun<R, ($(Expr<$arg>),*)> for F
    where
      Self: FnOnce(&mut Scope<R>, $(Expr<$arg>),*) -> R,
      Return: From<R>,
      $($arg: ToType),*
    {
      fn build_fn(self) -> FunDef<R, ($(Expr<$arg>),*)> {
        let mut scope = Scope::new(0);
        $( let $arg_ident = Expr::fun_arg($arg_rank); )*
        let args = vec![$( $arg::ty() ),*];
        let ret = self(&mut scope, $($arg_ident),*);
        let erased = ErasedFun::new(args, scope.erased, Return::from(ret).erased);

        FunDef::new(erased)
      }
    }
  }
}

impl_ToFun_args!(A0, a0, 0, A1, a1, 1);
impl_ToFun_args!(A0, a0, 0, A1, a1, 1, A2, a2, 2);
impl_ToFun_args!(A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3);
impl_ToFun_args!(A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4);
impl_ToFun_args!(A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5);
impl_ToFun_args!(A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8, 8
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10, A11, a11, 11
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10, A11, a11, 11, A12, a12, 12
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10, A11, a11, 11, A12, a12, 12, A13, a13, 13
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10, A11, a11, 11, A12, a12, 12, A13, a13, 13, A14, a14, 14
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10, A11, a11, 11, A12, a12, 12, A13, a13, 13, A14, a14, 14, A15, a15, 15
);
impl_ToFun_args!(
  A0, a0, 0, A1, a1, 1, A2, a2, 2, A3, a3, 3, A4, a4, 4, A5, a5, 5, A6, a6, 6, A7, a7, 7, A8, a8,
  8, A9, a9, 9, A10, a10, 10, A11, a11, 11, A12, a12, 12, A13, a13, 13, A14, a14, 14, A15, a15, 15,
  A16, a16, 16
);

/// An opaque function handle, used to call user-defined functions.
///
/// Function handles are created with the [`StageBuilder::fun`] function, introducing new functions in the EDSL. You
/// can then call the functions in the context of generating new expressions, returning them or creating variables.
///
/// Injecting a function call in the EDSL is done via two current mechanisms:
///
/// - Either call the [`FunHandle::call`] method:
///   - It is a function without argument if the represented function doesn’t have any argument.
///   - It is a unary function if the represented function has a single argument.
///   - It takes a tuple encapsulating the arguments for a n-ary represented function.
/// - **On nightly only**, you can enable the `shader-edsl-fun-call` feature-gate and calling the function will do the same thing
///   as [`FunHandle::call`]. However, because of how the various [`FnOnce`], [`FnMut`] and [`Fn`] traits are made,
///   functions taking several arguments take them as separate arguments as if it was a regular Rust function (they
///   don’t take a tuple as with the [`FunHandle::call`] method).
///
/// # Examples
///
/// A unary function squaring its argument, the regular way:
///
/// ```
/// # use shades::StageBuilder;
/// # StageBuilder::new_vertex_shader(|mut s, vertex| {
/// use shades::{Expr, FunHandle, Scope, lit, vec2};
///
/// let square = s.fun(|s: &mut Scope<Expr<i32>>, a: Expr<i32>| {
///   &a * &a
/// });
///
/// s.main_fun(|s: &mut Scope<()>| {
///   // call square with 3 and bind the result to a variable
///   let squared = s.var(square.call(lit!(3)));
/// })
/// # });
/// ```
///
/// The same function but with the `shader-edsl-fun-call` feature-gate enabled:
///
/// ```
/// # use shades::StageBuilder;
/// # StageBuilder::new_vertex_shader(|mut s, vertex| {
/// use shades::{Expr, Scope, lit};
///
/// let square = s.fun(|s: &mut Scope<Expr<i32>>, a: Expr<i32>| {
///   &a * &a
/// });
///
/// s.main_fun(|s: &mut Scope<()>| {
///   // call square with 3 and bind the result to a variable
///   # #[cfg(feature = "fun-call")]
///   let squared = s.var(square(lit!(3)));
/// })
/// # });
/// ```
///
/// A function taking two 3D vectors and a floating scalar and returning their linear interpolation, called with
/// three arguments:
///
/// ```
/// # use shades::StageBuilder;
/// # StageBuilder::new_vertex_shader(|mut s, vertex| {
/// use shades::{Expr, Mix as _, Scope, V3, lit, vec3};
///
/// let lerp = s.fun(|s: &mut Scope<Expr<V3<f32>>>, a: Expr<V3<f32>>, b: Expr<V3<f32>>, t: Expr<f32>| {
///   a.mix(b, t)
/// });
///
/// s.main_fun(|s: &mut Scope<()>| {
///   # #[cfg(feature = "fun-call")]
///   let a = vec3!(0., 0., 0.);
///   let b = vec3!(1., 1., 1.);
///
///   // call lerp here and bind it to a local variable
///   # #[cfg(feature = "fun-call")]
///   let result = s.var(lerp(a, b, lit!(0.75)));
/// })
/// # });
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct FunHandle<R, A> {
  erased: ErasedFunHandle,
  _phantom: PhantomData<(R, A)>,
}

impl<R> FunHandle<Expr<R>, ()> {
  /// Create an expression representing a function call to this function with no argument.
  ///
  /// See the documentation of [`FunHandle`] for examples.
  pub fn call(&self) -> Expr<R> {
    Expr::new(ErasedExpr::FunCall(self.erased.clone(), Vec::new()))
  }
}

/// Function call with no argument.
#[cfg(feature = "shader-edsl-fun-call")]
impl<R> FnOnce<()> for FunHandle<Expr<R>, ()> {
  type Output = Expr<R>;

  extern "rust-call" fn call_once(self, _: ()) -> Self::Output {
    self.call()
  }
}

/// Function call with no argument.
#[cfg(feature = "shader-edsl-fun-call")]
impl<R> FnMut<()> for FunHandle<Expr<R>, ()> {
  extern "rust-call" fn call_mut(&mut self, _: ()) -> Self::Output {
    self.call()
  }
}

/// Function call with no argument.
#[cfg(feature = "shader-edsl-fun-call")]
impl<R> Fn<()> for FunHandle<Expr<R>, ()> {
  extern "rust-call" fn call(&self, _: ()) -> Self::Output {
    self.call()
  }
}

impl<R, A> FunHandle<Expr<R>, Expr<A>> {
  /// Create an expression representing a function call to this unary function.
  ///
  /// See the documentation of [`FunHandle`] for examples.
  pub fn call(&self, a: Expr<A>) -> Expr<R> {
    Expr::new(ErasedExpr::FunCall(self.erased.clone(), vec![a.erased]))
  }
}

/// Unary function call.
#[cfg(feature = "shader-edsl-fun-call")]
impl<R, A> FnOnce<(Expr<A>,)> for FunHandle<Expr<R>, Expr<A>> {
  type Output = Expr<R>;

  extern "rust-call" fn call_once(self, a: (Expr<A>,)) -> Self::Output {
    self.call(a.0)
  }
}

/// Unary function call.
#[cfg(feature = "shader-edsl-fun-call")]
impl<R, A> FnMut<(Expr<A>,)> for FunHandle<Expr<R>, Expr<A>> {
  extern "rust-call" fn call_mut(&mut self, a: (Expr<A>,)) -> Self::Output {
    self.call(a.0)
  }
}

/// Unary function call.
#[cfg(feature = "shader-edsl-fun-call")]
impl<R, A> Fn<(Expr<A>,)> for FunHandle<Expr<R>, Expr<A>> {
  extern "rust-call" fn call(&self, a: (Expr<A>,)) -> Self::Output {
    self.call(a.0)
  }
}

// the first stage must be named S0
macro_rules! impl_FunCall {
  ( $( ( $arg_name:ident, $arg_ty:ident ) ),*) => {
    impl<R, $($arg_ty),*> FunHandle<Expr<R>, ($(Expr<$arg_ty>),*)>
    {
      /// Create an expression representing a function call to this function.
      ///
      /// See the documentation of [`FunHandle`] for examples.
      pub fn call(&self, $($arg_name : Expr<$arg_ty>),*) -> Expr<R> {
        Expr::new(ErasedExpr::FunCall(self.erased.clone(), vec![$($arg_name.erased),*]))
      }
    }

    #[cfg(feature = "shader-edsl-fun-call")]
    impl<R, $($arg_ty),*> FnOnce<($(Expr<$arg_ty>),*)> for FunHandle<Expr<R>, ($(Expr<$arg_ty>),*)>
    {
      type Output = Expr<R>;

      extern "rust-call" fn call_once(self, ($($arg_name),*): ($(Expr<$arg_ty>),*)) -> Self::Output {
        self.call($($arg_name),*)
      }
    }

    #[cfg(feature = "shader-edsl-fun-call")]
    impl<R, $($arg_ty),*> FnMut<($(Expr<$arg_ty>),*)> for FunHandle<Expr<R>, ($(Expr<$arg_ty>),*)>
    {
      extern "rust-call" fn call_mut(&mut self, ($($arg_name),*): ($(Expr<$arg_ty>),*)) -> Self::Output {
        self.call($($arg_name),*)
      }
    }

    #[cfg(feature = "shader-edsl-fun-call")]
    impl<R, $($arg_ty),*> Fn<($(Expr<$arg_ty>),*)> for FunHandle<Expr<R>, ($(Expr<$arg_ty>),*)>
    {
      extern "rust-call" fn call(&self, ($($arg_name),*): ($(Expr<$arg_ty>),*)) -> Self::Output {
        self.call($($arg_name),*)
      }
    }
  };
}

// implement function calls for Expr up to 16 arguments
macro_rules! impl_FunCall_rec {
  ( ( $a:ident, $b:ident ) , ( $x:ident, $y:ident )) => {
    impl_FunCall!(($a, $b), ($x, $y));
  };

  ( ( $a:ident, $b:ident ) , ( $x: ident, $y: ident ) , $($r:tt)* ) => {
    impl_FunCall_rec!(($a, $b), $($r)*);
    impl_FunCall!(($a, $b), ($x, $y), $($r)*);
  };
}

impl_FunCall_rec!(
  (a, A),
  (b, B),
  (c, C),
  (d, D),
  (e, E),
  (f, F),
  (g, G),
  (h, H),
  (i, I),
  (j, J),
  (k, K),
  (l, L),
  (m, M),
  (n, N),
  (o, O),
  (p, P)
);
