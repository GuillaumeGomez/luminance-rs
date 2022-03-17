# Luminance design

This document describes the overall design of [luminance] starting from its current version, 0.46.

<!-- vim-markdown-toc GFM -->

* [Foreword: crate ecosystem](#foreword-crate-ecosystem)
* [Goals and main decisions](#goals-and-main-decisions)
* [Soundness and correctness](#soundness-and-correctness)

<!-- vim-markdown-toc -->

## Foreword: crate ecosystem

[luminance] is the name of the _core_ crate but also the name of the Luminance ecosystem. The ecosystem comprises
several crates, classified in different themes:

- The core crate, [luminance], the subject of this very document.
- The proc-macro crate, [luminance-derive].
- “Backend” crates, providing technology-dependent implementation to run [luminance] code on different kind of tech,
  such as OpenGL, WebGL, Vulkan, etc.
- “Windowing” / “platform” crates, in order to run [luminance] code on specific platforms.

The goal of this document is to describe [luminance] and its core concept. For a description of the rest of the
ecosystem, you can glance through the [/docs].

## Goals and main decisions

[luminance] takes a different approach than other graphics crates in terms of how things should be done. The idea is
that code should drive as much as possible, from the memory perspective, bug perspective and logic perspective. Most of
the code you write should be checked as much as possible by your compiler, and the knowledge should be centralized as
much as possible. Bringing the knowledge together allows for better decisions, at the cost of flexibility. Indeed, the
goal of the ecosystem is not to be flexible in the sense to be usable with other graphics crates. The goal is to provide
a unique, safe and sound ecosystem. About this topic, see the [section about soundness](#soundness)

[luminance] was designed so that the following categories and families of _problems_ are avoided as much as possible.
Obviously, it is not possible to avoid everything, or every items in a given category, but still, the main incentive is
to minimize the number of items in this list that can bite both Luminance contributors, and Luminance end-users:

- Memory issues. Those range from memory leaks, double-free, use-after-free, (forbidden) random memory access, etc. This
  is the category that Rust defines its `unsafe` concept with, mainly.
- Panics. Panicking is a tool that can be interesting in some cases, but in a library, it does not have a place (at
  least not in the Luminance ecosystem). APIs will not be written in a way that misusing the API would result in a
  panic. For instance, accessing the _ith_ element of an array should be typed so that the returned element is
  `Option<T>`, not `T` with hidden panics.
- Typing issues. Types in Luminance are strong in the sense that they lift preconditions and invariants into the type
  system. Instead of having to check whether a value is always positive at runtime, the philosophy is to lift an
  arbitrary number into a `PositiveNumber` type once, doing the fallible conversion only once, and then assuming the
  invariant holds because we trust the type system. In that sense, types in Luminance follow closely the concept of
  _refinement typing_. That brings a lot of advantage, from less error-prone code, to better runtime performance
  (because the checks are done only once at construction, not at every use-case — if you don’t do refinement typing, you
  **must** check that kind of condition in each public functions taking such an argument, for instance; OpenGL is a good
  example of what not to do, for instance). Basically: leave the checks to the compiler, and enjoy raw runtime speed.
- Mutation and state corruption. The Luminance ecosystem is designed in a way that mutation and states are always
  decorated in a way that it is impossible (or close to) to corrupt global state or even local state. Some backend
  technologies have a huge dependency on global state invariants, and Luminance tries its hardest to keep the invariants
  safe from being violated. It goes from abstractions in the core crate, to state trackers indexed in the type system.
- Logical bugs. Lots of logical bugs can actually be avoided by webbing types and functions in a way to make it
  statically impossible to create bad runtime constructs. There are still exceptions, especially at the boundary of the
  crates (where we need to serialize / deserialize something, for instance), but Luminance does its best to ensure that
  logical expressions are sound and that users cannot express illogical statements — i.e. they won’t compile.
- Optimization problems. Some backend technologies require a very specific order of function calls, or formats, or
  arguments. All of this complexity and details are hidden behind [luminance]’s abstractions.

Because of all this, [luminance]’s API can be a bit frightening to the non-initiated. For instance, [luminance] uses a
lot all of these concepts:

- [Higher-Rank Trait Bounds (HRTBs)](https://doc.rust-lang.org/nomicon/hrtb.html).
- [Rank-2 types](https://wiki.haskell.org/Rank-N_types).
- Associated types, associated type constructors, type families.
- [Type states](https://en.wikipedia.org/wiki/Typestate_analysis).
- [Refinement types](https://en.wikipedia.org/wiki/Refinement_type).
- State trackers.
- And more…

As often as possible, complex type and function signatures will be well documented so that newcomers and people not used
to all of those concepts can understand and use the API nevertheless.

## Soundness and correctness

Rust has this `unsafe` keyword that people use to enter an unsafe section where one can do dangerous things, such as
dereferencing a pointer, type casting, calling FFI functions, etc. `unsafe` is also overloaded as soon as we start using
it to “convince people not to use someting unless they know exactly what they are doing.” In [luminance], that maps to
unsafe implementors, mostly (`unsafe trait` / `unsafe impl`). Misimplementing those traits will not make your
application crash or leak memory, but it will probably yield to misbehavior, incoherent state, etc.

For this reason, the concept of _soundness_ is a bit blended with safety in [luminance]. What we would need is an
`unsound` keyword, meaning that the code marked `unsound` can be unsound if not implemented properly. Because we do not
have such a keyword, `unsafe` is used and the definition of _safety_ in [luminance] is a superset of the one commonly
accepted in the Rust ecosystem: memory safety, plus soundness and correctness.

[luminance]: https://crates.io/crates/luminance
[luminance-derive]: https://crates.io/crates/luminance-derive
