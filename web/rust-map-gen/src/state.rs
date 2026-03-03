use serde::Serialize;

#[derive(Clone, Debug, PartialEq)]
pub enum HillDrawingMode {
    Brush,
    Polygon,
}

#[derive(Clone, Debug)]
pub struct PolygonData {
    pub id: u32,
    pub vertices: Vec<[i32; 2]>,
    pub edge_gaps: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct BrushStroke {
    pub points: Vec<[i32; 2]>,
    pub value: i32,
    pub brush_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<i32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<i32>) -> Result<Self, String> {
        if rows.saturating_mul(cols) != data.len() {
            return Err(format!(
                "Invalid matrix payload: rows={rows}, cols={cols}, data_len={}",
                data.len()
            ));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0; rows.saturating_mul(cols)],
        }
    }

    pub fn fill(rows: usize, cols: usize, value: i32) -> Self {
        Self {
            rows,
            cols,
            data: vec![value; rows.saturating_mul(cols)],
        }
    }

    pub fn from_rows(rows_data: Vec<Vec<i32>>) -> Option<Self> {
        let rows = rows_data.len();
        let cols = rows_data.first()?.len();
        if cols == 0 || rows_data.iter().any(|row| row.len() != cols) {
            return None;
        }
        let mut data = Vec::with_capacity(rows * cols);
        for row in rows_data {
            data.extend(row);
        }
        Self::new(rows, cols, data).ok()
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> i32 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: i32) {
        self.data[r * self.cols + c] = val;
    }

    pub fn same_shape(&self, rows: usize, cols: usize) -> bool {
        self.rows == rows && self.cols == cols
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum WizardStep {
    Coastline = 0,
    Hills = 1,
    HeightOcean = 2,
    CommandCenters = 3,
    Resources = 4,
    Finalize = 5,
}

#[derive(Clone, Debug)]
pub struct CcGroup {
    pub id: i32,
    pub positions: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct WizardState {
    pub initial_matrix: Option<Matrix>,
    pub height: usize,
    pub width: usize,
    pub mirroring: String,
    pub pattern: i32,
    pub num_height_levels: i32,
    pub num_ocean_levels: i32,
    pub num_command_centers: i32,
    pub num_resource_pulls: i32,
    pub randomized_matrix: Option<Matrix>,
    pub coastline_height_map: Option<Matrix>,
    // Legacy wall_matrix - kept for compatibility, but prefer brush_wall_matrix
    pub wall_matrix: Option<Matrix>,
    pub perlin_seed: Option<i32>,
    pub perlin_map: Option<Vec<f64>>,
    pub height_map: Option<Matrix>,
    pub units_matrix: Option<Matrix>,
    pub cc_positions: Vec<(usize, usize)>,
    pub cc_groups: Vec<CcGroup>,
    pub items_matrix: Option<Matrix>,
    pub resource_positions: Vec<(usize, usize)>,
    pub resource_groups: Vec<Vec<(usize, usize)>>,
    pub id_matrix: Option<Matrix>,
    // Legacy polygon_walls - kept for compatibility
    pub polygon_walls: Vec<PolygonData>,
    pub current_step: WizardStep,
    pub completed_step: i32,

    // ========== NEW: Hill Drawing Mode ==========
    pub hill_drawing_mode: HillDrawingMode,

    // ========== BRUSH MODE STATE ==========
    pub brush_wall_matrix: Option<Matrix>,
    pub brush_strokes: Vec<BrushStroke>,
    pub brush_strokes_redo: Vec<BrushStroke>,

    // ========== POLYGON MODE STATE ==========
    pub polygon_depth_matrix: Option<Matrix>,
    pub polygons: Vec<PolygonData>,
    pub mirrored_polygons: Vec<PolygonData>,
    pub polygons_undo: Vec<Vec<PolygonData>>,
    pub polygons_redo: Vec<Vec<PolygonData>>,
}

impl Default for WizardState {
    fn default() -> Self {
        Self {
            initial_matrix: Some(default_initial_matrix()),
            height: 160,
            width: 160,
            mirroring: "vertical".to_string(),
            pattern: 1,
            num_height_levels: 7,
            num_ocean_levels: 3,
            num_command_centers: 4,
            num_resource_pulls: 12,
            randomized_matrix: None,
            coastline_height_map: None,
            wall_matrix: None,
            perlin_seed: None,
            perlin_map: None,
            height_map: None,
            units_matrix: None,
            cc_positions: Vec::new(),
            cc_groups: Vec::new(),
            items_matrix: None,
            resource_positions: Vec::new(),
            resource_groups: Vec::new(),
            id_matrix: None,
            polygon_walls: Vec::new(),
            current_step: WizardStep::Coastline,
            completed_step: -1,
            // New fields
            hill_drawing_mode: HillDrawingMode::Brush,
            brush_wall_matrix: None,
            brush_strokes: Vec::new(),
            brush_strokes_redo: Vec::new(),
            polygon_depth_matrix: None,
            polygons: Vec::new(),
            mirrored_polygons: Vec::new(),
            polygons_undo: Vec::new(),
            polygons_redo: Vec::new(),
        }
    }
}

impl WizardState {
    pub fn invalidate_from(&mut self, step: WizardStep) {
        if step <= WizardStep::Coastline {
            self.randomized_matrix = None;
            self.coastline_height_map = None;
        }
        if step <= WizardStep::Hills {
            self.wall_matrix = None;
            self.polygon_walls.clear();
            // Also clear new matrices
            self.brush_wall_matrix = None;
            self.brush_strokes.clear();
            self.brush_strokes_redo.clear();
            self.polygon_depth_matrix = None;
            self.polygons.clear();
            self.mirrored_polygons.clear();
            self.polygons_undo.clear();
            self.polygons_redo.clear();
        }
        if step <= WizardStep::HeightOcean {
            self.perlin_seed = None;
            self.perlin_map = None;
            self.height_map = None;
        }
        if step <= WizardStep::CommandCenters {
            self.units_matrix = None;
            self.cc_positions.clear();
            self.cc_groups.clear();
        }
        if step <= WizardStep::Resources {
            self.items_matrix = None;
            self.resource_positions.clear();
            self.resource_groups.clear();
        }
        if step <= WizardStep::Finalize {
            self.id_matrix = None;
        }
        self.completed_step = self.completed_step.min(step as i32 - 1);
    }

    pub fn ensure_wall_matrix(&mut self) -> &mut Matrix {
        let expected = (self.height, self.width);
        let recreate = self
            .wall_matrix
            .as_ref()
            .map(|m| !m.same_shape(expected.0, expected.1))
            .unwrap_or(true);
        if recreate {
            self.wall_matrix = Some(Matrix::zeros(expected.0, expected.1));
        }
        self.wall_matrix.as_mut().expect("wall matrix initialized")
    }

    pub fn ensure_brush_wall_matrix(&mut self) -> &mut Matrix {
        let expected = (self.height, self.width);
        let recreate = self
            .brush_wall_matrix
            .as_ref()
            .map(|m| !m.same_shape(expected.0, expected.1))
            .unwrap_or(true);
        if recreate {
            self.brush_wall_matrix = Some(Matrix::zeros(expected.0, expected.1));
        }
        self.brush_wall_matrix.as_mut().expect("brush wall matrix initialized")
    }

    pub fn snapshot(&self) -> WizardSnapshot {
        // Return only the active mode's matrix in wall_matrix field
        let active_wall_matrix = match self.hill_drawing_mode {
            HillDrawingMode::Brush => {
                self.brush_wall_matrix.as_ref().map(MatrixPayload::from)
            }
            HillDrawingMode::Polygon => {
                self.polygon_depth_matrix.as_ref().map(MatrixPayload::from)
            }
        };

        WizardSnapshot {
            meta: SnapshotMeta {
                height: self.height as i32,
                width: self.width as i32,
                mirroring: self.mirroring.clone(),
                pattern: self.pattern,
                num_height_levels: self.num_height_levels,
                num_ocean_levels: self.num_ocean_levels,
                num_command_centers: self.num_command_centers,
                num_resource_pulls: self.num_resource_pulls,
                completed_step: self.completed_step,
                current_step: self.current_step as i32,
                hill_drawing_mode: match self.hill_drawing_mode {
                    HillDrawingMode::Brush => "brush".to_string(),
                    HillDrawingMode::Polygon => "polygon".to_string(),
                },
            },
            cc_positions: self
                .cc_positions
                .iter()
                .map(|(r, c)| [*r as i32, *c as i32])
                .collect(),
            resource_positions: self
                .resource_positions
                .iter()
                .map(|(r, c)| [*r as i32, *c as i32])
                .collect(),
            matrices: SnapshotMatrices {
                coastline_height_map: self.coastline_height_map.as_ref().map(MatrixPayload::from),
                wall_matrix: active_wall_matrix,
                height_map: self.height_map.as_ref().map(MatrixPayload::from),
                id_matrix: self.id_matrix.as_ref().map(MatrixPayload::from),
                items_matrix: self.items_matrix.as_ref().map(MatrixPayload::from),
                units_matrix: self.units_matrix.as_ref().map(MatrixPayload::from),
            },
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MatrixPayload {
    pub shape: [i32; 2],
    pub data: Vec<i32>,
}

impl From<&Matrix> for MatrixPayload {
    fn from(matrix: &Matrix) -> Self {
        Self {
            shape: [matrix.rows as i32, matrix.cols as i32],
            data: matrix.data.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SnapshotMeta {
    pub height: i32,
    pub width: i32,
    pub mirroring: String,
    pub pattern: i32,
    pub num_height_levels: i32,
    pub num_ocean_levels: i32,
    pub num_command_centers: i32,
    pub num_resource_pulls: i32,
    pub completed_step: i32,
    pub current_step: i32,
    pub hill_drawing_mode: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SnapshotMatrices {
    pub coastline_height_map: Option<MatrixPayload>,
    pub wall_matrix: Option<MatrixPayload>,
    pub height_map: Option<MatrixPayload>,
    pub id_matrix: Option<MatrixPayload>,
    pub items_matrix: Option<MatrixPayload>,
    pub units_matrix: Option<MatrixPayload>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WizardSnapshot {
    pub meta: SnapshotMeta,
    pub cc_positions: Vec<[i32; 2]>,
    pub resource_positions: Vec<[i32; 2]>,
    pub matrices: SnapshotMatrices,
}

fn default_initial_matrix() -> Matrix {
    Matrix::from_rows(vec![
        vec![0, 0, 1, 0, 0],
        vec![0, 1, 1, 1, 0],
        vec![1, 1, 1, 1, 1],
        vec![0, 1, 1, 1, 0],
        vec![0, 0, 1, 0, 0],
    ])
    .expect("default matrix must be valid")
}
