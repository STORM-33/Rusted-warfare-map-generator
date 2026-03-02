use rand::Rng;

use crate::state::Matrix;

pub fn subdivide(matrix: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(matrix.rows * 2, matrix.cols * 2);
    for r in 0..matrix.rows {
        for c in 0..matrix.cols {
            let v = matrix.get(r, c);
            let rr = r * 2;
            let cc = c * 2;
            out.set(rr, cc, v);
            out.set(rr + 1, cc, v);
            out.set(rr, cc + 1, v);
            out.set(rr + 1, cc + 1, v);
        }
    }
    out
}

pub fn randomize(matrix: &Matrix, smoothness: f64) -> Matrix {
    let mut rng = rand::thread_rng();
    let mut out = matrix.clone();
    if matrix.rows == 0 || matrix.cols == 0 {
        return out;
    }

    for r in 0..matrix.rows {
        for c in 0..matrix.cols {
            let current = matrix.get(r, c);
            let mut neighbor_count = 0_i32;

            for dr in -1..=1 {
                for dc in -1..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = clamp_index(r as isize + dr, matrix.rows);
                    let nc = clamp_index(c as isize + dc, matrix.cols);
                    if matrix.get(nr, nc) != current {
                        neighbor_count += 1;
                    }
                }
            }

            let base_prob = 0.2 + 0.1 * (neighbor_count as f64 - 3.0);
            let prob = (base_prob * (1.0 - smoothness * 0.85)).clamp(0.0, 1.0);
            if neighbor_count >= 3 && rng.gen::<f64>() < prob {
                out.set(r, c, 1 - current);
            }
        }
    }
    out
}

pub fn mirror(matrix: &Matrix, mirroring: &str) -> Matrix {
    let mut out = matrix.clone();
    match mirroring {
        "none" => out,
        "horizontal" => {
            let mid = out.rows / 2;
            for r in 0..mid {
                let src_row = mid - 1 - r;
                let dst_row = out.rows - mid + r;
                for c in 0..out.cols {
                    out.set(dst_row, c, out.get(src_row, c));
                }
            }
            out
        }
        "vertical" => {
            let mid = out.cols / 2;
            for c in 0..mid {
                let src_col = mid - 1 - c;
                let dst_col = out.cols - mid + c;
                for r in 0..out.rows {
                    out.set(r, dst_col, out.get(r, src_col));
                }
            }
            out
        }
        "diagonal1" => {
            let n = out.rows.min(out.cols);
            for i in 0..n {
                for j in (i + 1)..n {
                    out.set(j, i, out.get(i, j));
                }
            }
            out
        }
        "diagonal2" => {
            let source = out.clone();
            let n = source.rows.min(source.cols);
            for i in 0..n {
                for j in 0..n {
                    if i + j >= n {
                        out.set(j, i, source.get(n - 1 - i, n - 1 - j));
                    }
                }
            }
            out
        }
        "both" => {
            let mid_r = out.rows / 2;
            for r in 0..mid_r {
                let src_row = mid_r - 1 - r;
                let dst_row = out.rows - mid_r + r;
                for c in 0..out.cols {
                    out.set(dst_row, c, out.get(src_row, c));
                }
            }

            let mid_c = out.cols / 2;
            for c in 0..mid_c {
                let src_col = mid_c - 1 - c;
                let dst_col = out.cols - mid_c + c;
                for r in 0..out.rows {
                    out.set(r, dst_col, out.get(r, src_col));
                }
            }
            out
        }
        _ => out,
    }
}

pub fn scale_matrix(matrix: &Matrix, target_height: usize, target_width: usize) -> Matrix {
    if target_height == 0 || target_width == 0 || matrix.rows == 0 || matrix.cols == 0 {
        return Matrix::zeros(target_height, target_width);
    }

    let mut out = Matrix::zeros(target_height, target_width);
    for r in 0..target_height {
        let src_r = r * matrix.rows / target_height;
        for c in 0..target_width {
            let src_c = c * matrix.cols / target_width;
            out.set(r, c, matrix.get(src_r, src_c));
        }
    }
    out
}

fn clamp_index(value: isize, max_len: usize) -> usize {
    if max_len == 0 {
        return 0;
    }
    value.clamp(0, max_len as isize - 1) as usize
}

#[cfg(test)]
mod tests {
    use super::{mirror, scale_matrix, subdivide};
    use crate::state::Matrix;

    #[test]
    fn subdivide_doubles_matrix_size() {
        let src = Matrix::from_rows(vec![vec![1, 0], vec![0, 1]]).unwrap();
        let out = subdivide(&src);
        assert_eq!(out.rows, 4);
        assert_eq!(out.cols, 4);
        assert_eq!(out.get(0, 0), 1);
        assert_eq!(out.get(1, 1), 1);
        assert_eq!(out.get(2, 2), 1);
    }

    #[test]
    fn scale_matrix_matches_nearest_neighbor_indices() {
        let src = Matrix::from_rows(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let out = scale_matrix(&src, 4, 4);
        assert_eq!(out.get(0, 0), 1);
        assert_eq!(out.get(0, 3), 2);
        assert_eq!(out.get(3, 0), 3);
        assert_eq!(out.get(3, 3), 4);
    }

    #[test]
    fn vertical_mirror_copies_left_half_to_right() {
        let src = Matrix::from_rows(vec![
            vec![1, 0, 9, 9],
            vec![2, 0, 9, 9],
            vec![3, 0, 9, 9],
            vec![4, 0, 9, 9],
        ])
        .unwrap();
        let out = mirror(&src, "vertical");
        assert_eq!(out.get(0, 3), 1);
        assert_eq!(out.get(1, 3), 2);
        assert_eq!(out.get(2, 3), 3);
        assert_eq!(out.get(3, 3), 4);
    }
}
