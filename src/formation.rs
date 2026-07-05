//! Formation control.

use core::f64;

use crate::agent::Agent;
use crate::vec2::Vec2;

/// Formation shape definition: each slot is an offset from the formation center.
#[derive(Debug, Clone)]
pub struct Formation {
    pub offsets: Vec<Vec2>,
}

impl Formation {
    /// Create a line formation along the X-axis with given spacing.
    #[must_use]
    pub fn line(count: usize, spacing: f64) -> Self {
        let offsets = (0..count)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let x = (i as f64 - (count as f64 - 1.0) / 2.0) * spacing;
                Vec2::new(x, 0.0)
            })
            .collect();
        Self { offsets }
    }

    /// Create a ring formation with given radius.
    #[must_use]
    pub fn ring(count: usize, radius: f64) -> Self {
        let offsets = (0..count)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let angle = 2.0 * f64::consts::PI * (i as f64) / (count as f64);
                Vec2::new(radius * angle.cos(), radius * angle.sin())
            })
            .collect();
        Self { offsets }
    }

    /// Create a V-formation (wedge) with given spacing and opening angle.
    #[must_use]
    pub fn v_shape(count: usize, spacing: f64, half_angle: f64) -> Self {
        let mut offsets = Vec::with_capacity(count);
        offsets.push(Vec2::zero());
        for i in 1..count {
            #[allow(clippy::cast_precision_loss)]
            let rank = i.div_ceil(2) as f64;
            let side = if i % 2 == 1 { 1.0 } else { -1.0 };
            let x = side * rank * spacing * half_angle.sin();
            let y = -rank * spacing * half_angle.cos();
            offsets.push(Vec2::new(x, y));
        }
        Self { offsets }
    }

    /// Create a grid formation with given rows, cols, and spacing.
    #[must_use]
    pub fn grid(rows: usize, cols: usize, spacing: f64) -> Self {
        let mut offsets = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                #[allow(clippy::cast_precision_loss)]
                let x = (c as f64 - (cols as f64 - 1.0) / 2.0) * spacing;
                #[allow(clippy::cast_precision_loss)]
                let y = (r as f64 - (rows as f64 - 1.0) / 2.0) * spacing;
                offsets.push(Vec2::new(x, y));
            }
        }
        Self { offsets }
    }

    /// Number of slots in this formation.
    #[must_use]
    pub const fn slot_count(&self) -> usize {
        self.offsets.len()
    }
}

/// Compute steering forces to move agents toward their assigned formation slots.
#[must_use]
pub fn formation_steer(
    agents: &[Agent],
    formation: &Formation,
    center: Vec2,
    gain: f64,
) -> Vec<Vec2> {
    let n = agents.len().min(formation.slot_count());
    (0..n)
        .map(|i| {
            let target = center + formation.offsets[i];
            (target - agents[i].position) * gain
        })
        .collect()
}
