//! Shader builtins.

use crate::shader::{
  edsl::expr::{ErasedExpr, Expr, Var},
  types::Vec4,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BuiltIn {
  Vertex(VertexBuiltIn),
  TessCtrl(TessCtrlBuiltIn),
  TessEval(TessEvalBuiltIn),
  Geometry(GeometryBuiltIn),
  Fragment(FragmentBuiltIn),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VertexBuiltIn {
  VertexID,
  InstanceID,
  BaseVertex,
  BaseInstance,
  Position,
  PointSize,
  ClipDistance,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TessCtrlBuiltIn {
  MaxPatchVerticesIn,
  PatchVerticesIn,
  PrimitiveID,
  InvocationID,
  TessellationLevelOuter,
  TessellationLevelInner,
  In,
  Out,
  Position,
  PointSize,
  ClipDistance,
  CullDistance,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TessEvalBuiltIn {
  TessCoord,
  MaxPatchVerticesIn,
  PatchVerticesIn,
  PrimitiveID,
  TessellationLevelOuter,
  TessellationLevelInner,
  In,
  Out,
  Position,
  PointSize,
  ClipDistance,
  CullDistance,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GeometryBuiltIn {
  In,
  Out,
  Position,
  PointSize,
  ClipDistance,
  CullDistance,
  PrimitiveID,
  PrimitiveIDIn,
  InvocationID,
  Layer,
  ViewportIndex,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FragmentBuiltIn {
  FragCoord,
  FrontFacing,
  PointCoord,
  SampleID,
  SamplePosition,
  SampleMaskIn,
  ClipDistance,
  CullDistance,
  PrimitiveID,
  Layer,
  ViewportIndex,
  FragDepth,
  SampleMask,
  HelperInvocation,
}

/// Vertex shader environment.
#[derive(Debug)]
pub struct VertexShaderEnv {
  // inputs
  /// ID of the current vertex.
  pub vertex_id: Expr<i32>,

  /// Instance ID of the current vertex.
  pub instance_id: Expr<i32>,

  /// Base vertex offset.
  pub base_vertex: Expr<i32>,

  /// Base instance vertex offset.
  pub base_instance: Expr<i32>,

  // outputs
  /// 4D position of the vertex.
  pub position: Var<Vec4<f32>>,

  /// Point size of the vertex.
  pub point_size: Var<f32>,

  // Clip distances to user-defined plans.
  pub clip_distance: Var<[f32]>,
}

impl VertexShaderEnv {
  pub(crate) fn new() -> Self {
    let vertex_id = Expr::builtin(BuiltIn::Vertex(VertexBuiltIn::VertexID));
    let instance_id = Expr::builtin(BuiltIn::Vertex(VertexBuiltIn::InstanceID));
    let base_vertex = Expr::builtin(BuiltIn::Vertex(VertexBuiltIn::BaseVertex));
    let base_instance = Expr::builtin(BuiltIn::Vertex(VertexBuiltIn::BaseInstance));
    let position = Expr::builtin(BuiltIn::Vertex(VertexBuiltIn::Position)).to_var();
    let point_size = Expr::builtin(BuiltIn::Vertex(VertexBuiltIn::PointSize)).to_var();
    let clip_distance = Expr::builtin(BuiltIn::Vertex(VertexBuiltIn::ClipDistance)).to_var();

    Self {
      vertex_id,
      instance_id,
      base_vertex,
      base_instance,
      position,
      point_size,
      clip_distance,
    }
  }
}

/// Tessellation control shader environment.
#[derive(Debug)]
pub struct TessCtrlShaderEnv {
  // inputs
  /// Maximum number of vertices per patch.
  pub max_patch_vertices_in: Expr<i32>,

  /// Number of vertices for the current patch.
  pub patch_vertices_in: Expr<i32>,

  /// ID of the current primitive.
  pub primitive_id: Expr<i32>,

  /// ID of the current tessellation control shader invocation.
  pub invocation_id: Expr<i32>,

  /// Array of per-vertex input expressions.
  pub input: Expr<[TessControlPerVertexIn]>,

  // outputs
  /// Outer tessellation levels.
  pub tess_level_outer: Var<[f32; 4]>,

  /// Inner tessellation levels.
  pub tess_level_inner: Var<[f32; 2]>,

  /// Array of per-vertex output variables.
  pub output: Var<[TessControlPerVertexOut]>,
}

impl TessCtrlShaderEnv {
  fn new() -> Self {
    let max_patch_vertices_in =
      Expr::builtin(BuiltIn::TessCtrl(TessCtrlBuiltIn::MaxPatchVerticesIn));
    let patch_vertices_in = Expr::builtin(BuiltIn::TessCtrl(TessCtrlBuiltIn::PatchVerticesIn));
    let primitive_id = Expr::builtin(BuiltIn::TessCtrl(TessCtrlBuiltIn::PrimitiveID));
    let invocation_id = Expr::builtin(BuiltIn::TessCtrl(TessCtrlBuiltIn::InvocationID));
    let input = Expr::builtin(BuiltIn::TessCtrl(TessCtrlBuiltIn::In));
    let tess_level_outer =
      Expr::builtin(BuiltIn::TessCtrl(TessCtrlBuiltIn::TessellationLevelOuter)).to_var();
    let tess_level_inner =
      Expr::builtin(BuiltIn::TessCtrl(TessCtrlBuiltIn::TessellationLevelInner)).to_var();
    let output = Expr::builtin(BuiltIn::TessCtrl(TessCtrlBuiltIn::Out)).to_var();

    Self {
      max_patch_vertices_in,
      patch_vertices_in,
      primitive_id,
      invocation_id,
      input,
      tess_level_outer,
      tess_level_inner,
      output,
    }
  }
}

/// Read-only, input tessellation control shader environment.
#[derive(Debug)]
pub struct TessControlPerVertexIn;

impl Expr<TessControlPerVertexIn> {
  pub fn position(&self) -> Expr<Vec4<f32>> {
    self.field(Expr::builtin(BuiltIn::TessCtrl(TessCtrlBuiltIn::Position)))
  }

  pub fn point_size(&self) -> Expr<f32> {
    self.field(Expr::builtin(BuiltIn::TessCtrl(TessCtrlBuiltIn::PointSize)))
  }

  pub fn clip_distance(&self) -> Expr<[f32]> {
    self.field(Expr::builtin(BuiltIn::TessCtrl(
      TessCtrlBuiltIn::ClipDistance,
    )))
  }

  pub fn cull_distance(&self) -> Expr<[f32]> {
    self.field(Expr::builtin(BuiltIn::TessCtrl(
      TessCtrlBuiltIn::CullDistance,
    )))
  }
}

/// Output tessellation control shader environment.
#[derive(Debug)]
pub struct TessControlPerVertexOut(());

impl Expr<TessControlPerVertexOut> {
  /// 4D position of the verte.
  pub fn position(&self) -> Var<V4<f32>> {
    let expr = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
        TessCtrlBuiltIn::Position,
      ))),
    };

    Var(Expr::new(expr))
  }

  /// Point size of the vertex.
  pub fn point_size(&self) -> Var<f32> {
    let expr = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
        TessCtrlBuiltIn::PointSize,
      ))),
    };

    Var(Expr::new(expr))
  }

  /// Clip distances to user-defined planes.
  pub fn clip_distance(&self) -> Var<[f32]> {
    let expr = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
        TessCtrlBuiltIn::ClipDistance,
      ))),
    };

    Var(Expr::new(expr))
  }

  /// Cull distances to user-defined planes.
  pub fn cull_distance(&self) -> Var<[f32]> {
    let expr = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessCtrl(
        TessCtrlBuiltIn::CullDistance,
      ))),
    };

    Var(Expr::new(expr))
  }
}

/// Tessellation evalution shader environm.nt
#[derive(Debug)]
pub struct TessEvalShaderEnv {
  // inputs
  /// Number of vertices in the current patch.
  pub patch_vertices_in: Expr<i32>,

  /// ID of the current primitive.
  pub primitive_id: Expr<i32>,

  /// Tessellation coordinates of the current vertex.
  pub tess_coord: Expr<V3<f32>>,

  /// Outer tessellation levels.
  pub tess_level_outer: Expr<[f32; 4]>,

  /// Inner tessellation levels.
  pub tess_level_inner: Expr<[f32; 2]>,

  /// Array of per-evertex expressions.
  pub input: Expr<[TessEvaluationPerVertexIn]>,

  // outputs
  /// 4D position of the vertex.
  pub position: Var<V4<f32>>,

  /// Point size of the vertex.
  pub point_size: Var<f32>,

  /// Clip distances to user-defined planes.
  pub clip_distance: Var<[f32]>,

  /// Cull distances to user-defined planes.
  pub cull_distance: Var<[f32]>,
}

impl TessEvalShaderEnv {
  fn new() -> Self {
    let patch_vertices_in = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
      TessEvalBuiltIn::PatchVerticesIn,
    )));
    let primitive_id = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
      TessEvalBuiltIn::PrimitiveID,
    )));
    let tess_coord = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
      TessEvalBuiltIn::TessCoord,
    )));
    let tess_level_outer = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
      TessEvalBuiltIn::TessellationLevelOuter,
    )));
    let tess_level_inner = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
      TessEvalBuiltIn::TessellationLevelInner,
    )));
    let input = Expr::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
      TessEvalBuiltIn::In,
    )));

    let position = Var::new(ScopedHandle::BuiltIn(BuiltIn::TessEval(
      TessEvalBuiltIn::Position,
    )));
    let point_size = Var::new(ScopedHandle::BuiltIn(BuiltIn::TessEval(
      TessEvalBuiltIn::PointSize,
    )));
    let clip_distance = Var::new(ScopedHandle::BuiltIn(BuiltIn::TessEval(
      TessEvalBuiltIn::ClipDistance,
    )));
    let cull_distance = Var::new(ScopedHandle::BuiltIn(BuiltIn::TessEval(
      TessEvalBuiltIn::ClipDistance,
    )));

    Self {
      patch_vertices_in,
      primitive_id,
      tess_coord,
      tess_level_outer,
      tess_level_inner,
      input,
      position,
      point_size,
      clip_distance,
      cull_distance,
    }
  }
}

/// Tessellation evaluation per-vertex expression.
#[derive(Debug)]
pub struct TessEvaluationPerVertexIn;

impl Expr<TessEvaluationPerVertexIn> {
  /// 4D position of the vertex.
  pub fn position(&self) -> Expr<V4<f32>> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
        TessEvalBuiltIn::Position,
      ))),
    };

    Expr::new(erased)
  }

  /// Point size of the vertex.
  pub fn point_size(&self) -> Expr<f32> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
        TessEvalBuiltIn::PointSize,
      ))),
    };

    Expr::new(erased)
  }

  /// Clip distances to user-defined planes.
  pub fn clip_distance(&self) -> Expr<[f32]> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
        TessEvalBuiltIn::ClipDistance,
      ))),
    };

    Expr::new(erased)
  }

  /// Cull distances to user-defined planes.
  pub fn cull_distance(&self) -> Expr<[f32]> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::TessEval(
        TessEvalBuiltIn::CullDistance,
      ))),
    };

    Expr::new(erased)
  }
}

/// Geometry shader environment.
#[derive(Debug)]
pub struct GeometryShaderEnv {
  // inputs
  /// Contains the index of the current primitive.
  pub primitive_id_in: Expr<i32>,

  /// ID of the current invocation of the geometry shader.
  pub invocation_id: Expr<i32>,

  /// Read-only environment for each vertices.
  pub input: Expr<[GeometryPerVertexIn]>,

  // outputs
  /// Output 4D vertex position.
  pub position: Var<V4<f32>>,

  /// Output vertex point size.
  pub point_size: Var<f32>,

  /// Output clip distances to user-defined planes.
  pub clip_distance: Var<[f32]>,

  /// Output cull distances to user-defined planes.
  pub cull_distance: Var<[f32]>,

  /// Primitive ID to write to in.
  pub primitive_id: Var<i32>,

  /// Layer to write to in.
  pub layer: Var<i32>,

  /// Viewport index to write to.
  pub viewport_index: Var<i32>,
}

impl GeometryShaderEnv {
  fn new() -> Self {
    let primitive_id_in = Expr::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
      GeometryBuiltIn::PrimitiveIDIn,
    )));
    let invocation_id = Expr::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
      GeometryBuiltIn::InvocationID,
    )));
    let input = Expr::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
      GeometryBuiltIn::In,
    )));

    let position = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::Position,
    )));
    let point_size = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::PointSize,
    )));
    let clip_distance = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::ClipDistance,
    )));
    let cull_distance = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::CullDistance,
    )));
    let primitive_id = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::PrimitiveID,
    )));
    let layer = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::Layer,
    )));
    let viewport_index = Var::new(ScopedHandle::BuiltIn(BuiltIn::Geometry(
      GeometryBuiltIn::ViewportIndex,
    )));

    Self {
      primitive_id_in,
      invocation_id,
      input,
      position,
      point_size,
      clip_distance,
      cull_distance,
      primitive_id,
      layer,
      viewport_index,
    }
  }
}

/// Read-only, input geometry shader environment.
#[derive(Debug)]
pub struct GeometryPerVertexIn;

impl Expr<GeometryPerVertexIn> {
  /// Provides 4D the position of the vertex.
  pub fn position(&self) -> Expr<V4<f32>> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
        GeometryBuiltIn::Position,
      ))),
    };

    Expr::new(erased)
  }

  /// Provides the size point of the vertex if it’s currently being rendered in point mode.
  pub fn point_size(&self) -> Expr<f32> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
        GeometryBuiltIn::PointSize,
      ))),
    };

    Expr::new(erased)
  }

  /// Clip distances to user planes of the vertex.
  pub fn clip_distance(&self) -> Expr<[f32]> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
        GeometryBuiltIn::ClipDistance,
      ))),
    };

    Expr::new(erased)
  }

  /// Cull distances to user planes of the vertex.
  pub fn cull_distance(&self) -> Expr<[f32]> {
    let erased = ErasedExpr::Field {
      object: Box::new(self.erased.clone()),
      field: Box::new(ErasedExpr::new_builtin(BuiltIn::Geometry(
        GeometryBuiltIn::CullDistance,
      ))),
    };

    Expr::new(erased)
  }
}

/// Fragment shader environment.
///
/// This type contains everything you have access to when writing a fragment shader.
#[derive(Debug)]
pub struct FragmentShaderEnv {
  // inputs
  /// Fragment coordinate in the framebuffer.
  pub frag_coord: Expr<V4<f32>>,

  /// Whether the fragment is front-facing.
  pub front_facing: Expr<bool>,

  /// Clip distances to user planes.
  ///
  /// This is an array giving the clip distances to each of the user clip planes.
  pub clip_distance: Expr<[f32]>,

  /// Cull distances to user planes.
  ///
  /// This is an array giving the cull distances to each of the user clip planes.
  pub cull_distance: Expr<[f32]>,

  /// Contains the 2D coordinates of a fragment within a point primitive.
  pub point_coord: Expr<V2<f32>>,

  /// ID of the primitive being currently rendered.
  pub primitive_id: Expr<i32>,

  /// ID of the sample being currently rendered.
  pub sample_id: Expr<i32>,

  /// Sample 2D coordinates.
  pub sample_position: Expr<V2<f32>>,

  /// Contains the computed sample coverage mask for the current fragment.
  pub sample_mask_in: Expr<i32>,

  /// Layer the fragment will be written to.
  pub layer: Expr<i32>,

  /// Viewport index the fragment will be written to.
  pub viewport_index: Expr<i32>,

  /// Indicates whether we are in a helper invocation of a fragment shader.
  pub helper_invocation: Expr<bool>,

  // outputs
  /// Depth of the fragment.
  pub frag_depth: Var<f32>,

  /// Sample mask of the fragment.
  pub sample_mask: Var<[i32]>,
}

impl FragmentShaderEnv {
  fn new() -> Self {
    let frag_coord = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::FragCoord,
    )));
    let front_facing = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::FrontFacing,
    )));
    let clip_distance = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::ClipDistance,
    )));
    let cull_distance = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::CullDistance,
    )));
    let point_coord = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::PointCoord,
    )));
    let primitive_id = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::PrimitiveID,
    )));
    let sample_id = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::SampleID,
    )));
    let sample_position = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::SamplePosition,
    )));
    let sample_mask_in = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::SampleMaskIn,
    )));
    let layer = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::Layer,
    )));
    let viewport_index = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::ViewportIndex,
    )));
    let helper_invocation = Expr::new(ErasedExpr::new_builtin(BuiltIn::Fragment(
      FragmentBuiltIn::HelperInvocation,
    )));

    let frag_depth = Var::new(ScopedHandle::BuiltIn(BuiltIn::Fragment(
      FragmentBuiltIn::FragDepth,
    )));
    let sample_mask = Var::new(ScopedHandle::BuiltIn(BuiltIn::Fragment(
      FragmentBuiltIn::SampleMask,
    )));

    Self {
      frag_coord,
      front_facing,
      clip_distance,
      cull_distance,
      point_coord,
      primitive_id,
      sample_id,
      sample_position,
      sample_mask_in,
      layer,
      viewport_index,
      helper_invocation,
      frag_depth,
      sample_mask,
    }
  }
}
