export type MatrixKey =
  | "coastline_height_map"
  | "wall_matrix"
  | "height_map"
  | "id_matrix"
  | "items_matrix"
  | "units_matrix";

export interface MatrixPayload {
  shape: [number, number];
  data: Int32Array;
}

export interface SerializedMatrixPayload {
  shape: [number, number];
  data: number[] | Int32Array;
}

export type HillDrawingMode = "brush" | "polygon";

export interface WizardSnapshotMeta {
  height: number;
  width: number;
  mirroring: string;
  pattern: number;
  num_height_levels: number;
  num_ocean_levels: number;
  num_command_centers: number;
  num_resource_pulls: number;
  completed_step: number;
  current_step: number;
  hill_drawing_mode: HillDrawingMode;
}

export interface WizardSnapshot {
  meta: WizardSnapshotMeta;
  cc_positions: [number, number][];
  resource_positions: [number, number][];
  matrices: Partial<Record<MatrixKey, MatrixPayload>>;
}

export interface SerializedWizardSnapshot
  extends Omit<WizardSnapshot, "matrices"> {
  matrices: Partial<Record<MatrixKey, SerializedMatrixPayload | null>>;
}

export type WorkerAction =
  | "init"
  | "run_coastline"
  | "draw_walls"
  | "clear_walls"
  | "set_wall_cells"
  | "set_polygon_walls"
  | "run_height_ocean"
  | "place_cc_manual"
  | "remove_cc_manual"
  | "place_cc_random"
  | "undo_cc"
  | "clear_cc"
  | "place_resource_manual"
  | "remove_resource_manual"
  | "place_resource_random"
  | "undo_resource"
  | "clear_resource"
  | "get_state_snapshot"
  | "run_finalize"
  | "quick_generate"
  | "reset_state"
  // New brush mode actions
  | "draw_brush_walls"
  | "undo_brush"
  | "redo_brush"
  | "clear_brush_walls"
  // New polygon mode actions
  | "update_polygons"
  | "undo_polygons"
  | "redo_polygons"
  | "clear_all_polygons"
  | "toggle_edge_gap"
  | "set_hill_drawing_mode";

export interface WorkerRequestMessage {
  type: WorkerAction;
  requestId: string;
  params?: Record<string, unknown>;
}

export interface WorkerLoadingMessage {
  type: "loading";
  stage: string;
  progress: number;
}

export interface WorkerInitCompleteMessage {
  type: "init_complete";
  requestId?: string;
}

export interface CoastlineFrame {
  label: string;
  shape: [number, number];
  data: Int32Array;
}

export interface SerializedCoastlineFrame {
  label: string;
  shape: [number, number];
  data: number[] | Int32Array;
}

export interface QuickGenerateFrame {
  label: string;
  height_map: MatrixPayload;
  id_matrix?: MatrixPayload;
  items_matrix?: MatrixPayload;
  units_matrix?: MatrixPayload;
}

export interface SerializedQuickGenerateFrame {
  label: string;
  height_map: SerializedMatrixPayload;
  id_matrix?: SerializedMatrixPayload | null;
  items_matrix?: SerializedMatrixPayload | null;
  units_matrix?: SerializedMatrixPayload | null;
}

export interface WorkerStepCompleteMessage {
  type: "step_complete";
  requestId: string;
  action: WorkerAction;
  snapshot?: WizardSnapshot;
  tmxBytes?: Uint8Array;
  frames?: CoastlineFrame[];
  quickFrames?: QuickGenerateFrame[];
  placed?: [number, number][];
  removed?: boolean;
}

export interface WorkerErrorMessage {
  type: "error";
  requestId?: string;
  action?: WorkerAction;
  message: string;
}

export type WorkerResponseMessage =
  | WorkerLoadingMessage
  | WorkerInitCompleteMessage
  | WorkerStepCompleteMessage
  | WorkerErrorMessage;

export type RenderMode = "sampled" | "full";

export interface Polygon {
  id: number;
  vertices: [number, number][]; // [row, col] grid coords
  edgeGaps: boolean[];          // per-edge: true = no wall on this edge
  closed: boolean;
}

export interface BrushStroke {
  points: [number, number][];
  value: 0 | 1 | 2;
  brushSize: number;
}

export type WallDrawingMode = "brush" | "polygon";
