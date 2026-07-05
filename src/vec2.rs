//! `Vec2` — 2D vector for swarm computations.

/// 2D vector for swarm computations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    #[must_use]
    pub fn length(self) -> f64 {
        self.x.hypot(self.y)
    }

    #[must_use]
    pub fn length_sq(self) -> f64 {
        self.x.mul_add(self.x, self.y * self.y)
    }

    #[must_use]
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            Self::zero()
        } else {
            Self {
                x: self.x / len,
                y: self.y / len,
            }
        }
    }

    #[must_use]
    pub fn distance_to(self, other: Self) -> f64 {
        (self - other).length()
    }

    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.x.mul_add(other.x, self.y * other.y)
    }

    #[must_use]
    pub fn clamped(self, max_len: f64) -> Self {
        let len = self.length();
        if len > max_len && len > 1e-12 {
            let scale = max_len / len;
            Self {
                x: self.x * scale,
                y: self.y * scale,
            }
        } else {
            self
        }
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Mul<f64> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}
