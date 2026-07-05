//! `Agent` — a single swarm agent.

use crate::vec2::Vec2;

/// A single swarm agent with position, velocity, and an optional leader flag.
#[derive(Debug, Clone)]
pub struct Agent {
    pub id: usize,
    pub position: Vec2,
    pub velocity: Vec2,
    pub is_leader: bool,
}

impl Agent {
    #[must_use]
    pub const fn new(id: usize, position: Vec2, velocity: Vec2) -> Self {
        Self {
            id,
            position,
            velocity,
            is_leader: false,
        }
    }

    #[must_use]
    pub const fn with_leader(mut self, leader: bool) -> Self {
        self.is_leader = leader;
        self
    }
}
