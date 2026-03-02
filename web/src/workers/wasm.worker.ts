/// <reference lib="webworker" />

import type {
  CoastlineFrame,
  MatrixKey,
  MatrixPayload,
  QuickGenerateFrame,
  SerializedCoastlineFrame,
  SerializedMatrixPayload,
  SerializedQuickGenerateFrame,
  SerializedWizardSnapshot,
  WorkerAction,
  WorkerRequestMessage,
  WorkerResponseMessage,
} from "@/lib/types";

type WasmModuleExports = {
  default: (moduleOrPath?: string) => Promise<unknown>;
  rpc_call: (method: string, paramsJson: string) => unknown;
};

const ctx = self as DedicatedWorkerGlobalScope;

let rpcCall: ((method: string, paramsJson: string) => unknown) | null = null;
let initPromise: Promise<void> | null = null;
let baseUrl = ctx.location.origin;

const dynamicImport = (moduleUrl: string) =>
  // Keep this runtime-dynamic so bundlers do not try to resolve local build-time files.
  (new Function("moduleUrl", "return import(moduleUrl);") as (
    url: string,
  ) => Promise<unknown>)(moduleUrl);

const post = (message: WorkerResponseMessage, transfer: Transferable[] = []) => {
  ctx.postMessage(message, transfer);
};

const postLoading = (stage: string, progress: number) => {
  post({ type: "loading", stage, progress });
};

const normalizeMatrix = (
  payload: SerializedMatrixPayload | null | undefined,
  transferables: Transferable[],
): MatrixPayload | undefined => {
  if (!payload) {
    return undefined;
  }
  const typed =
    payload.data instanceof Int32Array
      ? payload.data
      : Int32Array.from(payload.data);
  transferables.push(typed.buffer);
  return {
    shape: [payload.shape[0], payload.shape[1]],
    data: typed,
  };
};

const normalizeSnapshot = (
  snapshot: SerializedWizardSnapshot | undefined,
  transferables: Transferable[],
) => {
  if (!snapshot) {
    return undefined;
  }
  const matrices: Partial<Record<MatrixKey, MatrixPayload>> = {};
  const entries = Object.entries(snapshot.matrices ?? {}) as Array<
    [MatrixKey, SerializedMatrixPayload | null | undefined]
  >;
  for (const [key, payload] of entries) {
    const normalized = normalizeMatrix(payload, transferables);
    if (normalized) {
      matrices[key] = normalized;
    }
  }
  return {
    ...snapshot,
    matrices,
  };
};

const initializeRuntime = async () => {
  postLoading("wasm", 20);
  const wasmJsUrl = `${baseUrl}/wasm-pkg/rust_map_gen.js`;
  const wasmBinaryUrl = `${baseUrl}/wasm-pkg/rust_map_gen_bg.wasm`;
  let wasmModule: WasmModuleExports;
  try {
    wasmModule = (await dynamicImport(wasmJsUrl)) as WasmModuleExports;
  } catch (error) {
    const detail =
      error instanceof Error ? error.message : "Unknown dynamic import error";
    throw new Error(
      `Failed to load WASM JS module from ${wasmJsUrl}. ` +
      `Base URL is ${baseUrl}. Make sure the app port is correct and run "npm run build:wasm". ` +
      `Details: ${detail}`,
    );
  }
  if (typeof wasmModule.default !== "function") {
    throw new Error("WASM init() export is unavailable");
  }
  if (typeof wasmModule.rpc_call !== "function") {
    throw new Error("WASM rpc_call export is unavailable");
  }
  postLoading("wasm:init", 60);
  try {
    await wasmModule.default(wasmBinaryUrl);
  } catch (error) {
    const detail =
      error instanceof Error ? error.message : "Unknown WASM init error";
    throw new Error(
      `Failed to initialize WASM binary from ${wasmBinaryUrl}. ` +
      `Base URL is ${baseUrl}. Run "npm run build:wasm" and retry. ` +
      `Details: ${detail}`,
    );
  }
  rpcCall = wasmModule.rpc_call;
  postLoading("ready", 100);
};

const ensureInitialized = async () => {
  if (rpcCall) {
    return;
  }
  if (!initPromise) {
    initPromise = initializeRuntime();
  }
  await initPromise;
};

const invokeRpc = async (
  action: WorkerAction,
  params: Record<string, unknown> | undefined,
) => {
  await ensureInitialized();
  if (!rpcCall) {
    throw new Error("WASM bridge is unavailable");
  }
  const rawResult = rpcCall(action, JSON.stringify(params ?? {}));
  const transferables: Transferable[] = [];
  const resultObject =
    rawResult && typeof rawResult === "object"
      ? (rawResult as Record<string, unknown>)
      : {};
  const snapshot = normalizeSnapshot(
    resultObject.snapshot as SerializedWizardSnapshot | undefined,
    transferables,
  );

  let tmxBytes: Uint8Array | undefined;
  const tmxRaw = resultObject.tmx_bytes;
  if (tmxRaw instanceof Uint8Array) {
    tmxBytes = tmxRaw;
  } else if (Array.isArray(tmxRaw)) {
    tmxBytes = Uint8Array.from(tmxRaw as number[]);
  } else if (tmxRaw instanceof ArrayBuffer) {
    tmxBytes = new Uint8Array(tmxRaw);
  }
  if (tmxBytes) {
    transferables.push(tmxBytes.buffer);
  }

  let frames: CoastlineFrame[] | undefined;
  const rawFrames = resultObject.frames;
  if (Array.isArray(rawFrames) && rawFrames.length > 0) {
    frames = (rawFrames as SerializedCoastlineFrame[]).map((f) => {
      const typed =
        f.data instanceof Int32Array ? f.data : Int32Array.from(f.data);
      transferables.push(typed.buffer);
      return {
        label: f.label,
        shape: [f.shape[0], f.shape[1]] as [number, number],
        data: typed,
      };
    });
  }

  let quickFrames: QuickGenerateFrame[] | undefined;
  const rawQuickFrames = resultObject.quick_frames;
  if (Array.isArray(rawQuickFrames) && rawQuickFrames.length > 0) {
    quickFrames = (rawQuickFrames as SerializedQuickGenerateFrame[]).map((f) => {
      const hm = normalizeMatrix(f.height_map, transferables)!;
      const result: QuickGenerateFrame = { label: f.label, height_map: hm };
      if (f.id_matrix) {
        result.id_matrix = normalizeMatrix(f.id_matrix, transferables);
      }
      if (f.items_matrix) {
        result.items_matrix = normalizeMatrix(f.items_matrix, transferables);
      }
      if (f.units_matrix) {
        result.units_matrix = normalizeMatrix(f.units_matrix, transferables);
      }
      return result;
    });
  }

  let placed: [number, number][] | undefined;
  const rawPlaced = resultObject.placed;
  if (Array.isArray(rawPlaced)) {
    placed = rawPlaced
      .filter((pair): pair is [unknown, unknown] => Array.isArray(pair) && pair.length === 2)
      .map((pair): [number, number] => [Number(pair[0]), Number(pair[1])]);
  }
  const removed =
    typeof resultObject.removed === "boolean" ? resultObject.removed : undefined;

  return { snapshot, tmxBytes, frames, quickFrames, placed, removed, transferables };
};

ctx.onmessage = async (event: MessageEvent<WorkerRequestMessage>) => {
  const { type, requestId, params } = event.data;
  try {
    if (type === "init") {
      if (typeof params?.baseUrl === "string" && params.baseUrl.length > 0) {
        baseUrl = params.baseUrl;
      }
      await ensureInitialized();
      post({ type: "init_complete", requestId });
      return;
    }
    const { snapshot, tmxBytes, frames, quickFrames, placed, removed, transferables } = await invokeRpc(type, params);
    post(
      {
        type: "step_complete",
        requestId,
        action: type,
        snapshot,
        tmxBytes,
        frames,
        quickFrames,
        placed,
        removed,
      },
      transferables,
    );
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown worker error";
    post({
      type: "error",
      requestId,
      action: type,
      message,
    });
  }
};

export { };
