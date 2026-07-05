#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

//! ALICE-Swarm: Multi-agent swarm control library.
//!
//! Provides Boids algorithm, formation control, consensus protocols,
//! task allocation, obstacle avoidance, communication topology,
//! leader-follower dynamics, and swarm metrics.

pub mod agent;
pub mod boids;
pub mod formation;
pub mod leader;
pub mod metrics;
pub mod obstacle;
pub mod prelude;
pub mod task;
pub mod topology;
pub mod vec2;

#[cfg(test)]
mod integration_tests;

// Backward-compat re-exports.
pub use crate::agent::*;
pub use crate::boids::*;
pub use crate::formation::*;
pub use crate::leader::*;
pub use crate::metrics::*;
pub use crate::obstacle::*;
pub use crate::task::*;
pub use crate::topology::*;
pub use crate::vec2::*;
