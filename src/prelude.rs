//! Convenience re-export (= `use alice_swarm::prelude::*;`).

pub use crate::agent::Agent;
pub use crate::boids::{
    boids_alignment, boids_cohesion, boids_separation, boids_steer, boids_step, BoidParams,
};
pub use crate::formation::{formation_steer, Formation};
pub use crate::leader::{leader_follower_steer, leader_follower_step};
pub use crate::metrics::{
    swarm_avg_speed, swarm_centroid, swarm_collision_count, swarm_connectivity, swarm_diameter,
    swarm_min_distance, swarm_order, swarm_spread,
};
pub use crate::obstacle::{obstacle_avoidance, path_blocked, Obstacle};
pub use crate::task::{allocate_auction, allocate_greedy, Allocation, Task};
pub use crate::topology::{consensus_step, consensus_step_vec2, Topology};
pub use crate::vec2::Vec2;
