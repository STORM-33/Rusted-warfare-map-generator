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
  | "run_height_ocean"
  | "place_cc_manual"
  | "place_cc_random"
  | "undo_cc"
  | "clear_cc"
  | "place_resource_manual"
  | "place_resource_random"
  | "undo_resource"
  | "clear_resource"
  | "get_state_snapshot"
  | "run_finalize"
  | "reset_state";

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

export interface WorkerStepCompleteMessage {
  type: "step_complete";
  requestId: string;
  action: WorkerAction;
  snapshot?: WizardSnapshot;
  tmxBytes?: Uint8Array;
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
