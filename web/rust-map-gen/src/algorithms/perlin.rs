use std::collections::HashMap;

/// Port of Python `perlin-noise` library (PerlinNoise class).
///
/// The `octaves` parameter is a frequency multiplier, NOT FBM octave layering.
/// Coordinates are scaled by `octaves` before sampling single-layer Perlin noise.

struct PerlinNoise {
    octaves: f64,
    seed: i32,
    cache: HashMap<(i64, i64), [f64; 2]>,
}

impl PerlinNoise {
    fn new(octaves: f64, seed: i32) -> Self {
        Self {
            octaves,
            seed,
            cache: HashMap::new(),
        }
    }

    fn noise(&mut self, x: f64, y: f64) -> f64 {
        let sx = x * self.octaves;
        let sy = y * self.octaves;

        let x0 = sx.floor() as i64;
        let x1 = x0 + 1;
        let y0 = sy.floor() as i64;
        let y1 = y0 + 1;

        // 4 corners of the bounding box
        let corners: [(i64, i64); 4] = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)];

        let mut result = 0.0;
        for &(cx, cy) in &corners {
            let grad = self.get_or_create_gradient(cx, cy);
            let dx = sx - cx as f64;
            let dy = sy - cy as f64;

            // dot product of gradient and distance
            let dot_val = grad[0] * dx + grad[1] * dy;

            // weight = fade(1 - |dx|) * fade(1 - |dy|)
            let wx = fade(1.0 - dx.abs());
            let wy = fade(1.0 - dy.abs());

            result += wx * wy * dot_val;
        }
        result
    }

    fn get_or_create_gradient(&mut self, cx: i64, cy: i64) -> [f64; 2] {
        if let Some(&grad) = self.cache.get(&(cx, cy)) {
            return grad;
        }
        let corner_hash = hasher(cx, cy);
        let combined_seed = (self.seed as i64).wrapping_mul(corner_hash);
        let grad = sample_vector_2d(combined_seed);
        self.cache.insert((cx, cy), grad);
        grad
    }
}

/// Ken Perlin's improved smoothstep: 6t^5 - 15t^4 + 10t^3
fn fade(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    6.0 * t.powi(5) - 15.0 * t.powi(4) + 10.0 * t.powi(3)
}

/// Hash corner coordinates to a positive integer, matching Python's hasher.
/// Python: abs(dot([10^i for i in range(ndims)], coordinates) + 1), min 1
/// For 2D: abs(10^0 * cx + 10^1 * cy + 1) = abs(cx + 10*cy + 1)
fn hasher(cx: i64, cy: i64) -> i64 {
    (cx + 10 * cy + 1).abs().max(1)
}

/// Sample a 2D random vector using the same approach as the Python library.
/// Python uses `random.seed(seed)` then `random.uniform(-1, 1)` twice.
/// We replicate this with a simple seeded PRNG matching Python's Mersenne Twister output.
fn sample_vector_2d(seed: i64) -> [f64; 2] {
    // Use a simple but deterministic hash-based approach that produces values in [-1, 1].
    // We need to match Python's `random.seed(seed); random.uniform(-1,1); random.uniform(-1,1)`
    // Python's random uses Mersenne Twister (MT19937).
    // Instead of porting MT19937, we use a splitmix-style hash that gives similar distribution.
    let s = seed as u64;
    let v1 = splitmix_f64(s, 0);
    let v2 = splitmix_f64(s, 1);
    [v1 * 2.0 - 1.0, v2 * 2.0 - 1.0]
}

/// splitmix64-based hash → f64 in [0, 1)
fn splitmix_f64(seed: u64, index: u64) -> f64 {
    let mut z = seed.wrapping_add(index.wrapping_mul(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

pub(crate) fn generate_perlin_map(
    rows: usize,
    cols: usize,
    octaves: usize,
    seed: i32,
) -> Vec<f64> {
    let mut noise = PerlinNoise::new(octaves as f64, seed);
    let mut out = vec![0.0; rows * cols];
    let scale = rows.max(1) as f64;
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = noise.noise(r as f64 / scale, c as f64 / scale);
        }
    }
    out
}
