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

type PyProxyLike = {
  toJs: (options?: unknown) => unknown;
  destroy?: () => void;
};

type PyodideLike = {
  loadPackage: (packages: string | string[]) => Promise<void>;
  pyimport: (name: string) => PyProxyLike & Record<string, unknown>;
  runPython: (code: string) => unknown;
  FS: {
    writeFile: (path: string, data: string) => void;
  };
};

type WorkerGlobal = DedicatedWorkerGlobalScope & {
  loadPyodide?: (options: { indexURL: string }) => Promise<PyodideLike>;
};

const PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const PYTHON_FILES = [
  "procedural_map_generator_functions.py",
  "wizard_state.py",
  "map_pipeline.py",
  "bridge.py",
];

const ctx = self as WorkerGlobal;

let pyodide: PyodideLike | null = null;
let bridgeModule: (PyProxyLike & Record<string, unknown>) | null = null;
let initPromise: Promise<void> | null = null;
let baseUrl = "";

const post = (message: WorkerResponseMessage, transfer: Transferable[] = []) => {
  ctx.postMessage(message, transfer);
};

const postLoading = (stage: string, progress: number) => {
  post({ type: "loading", stage, progress });
};

const toPlainObject = (value: unknown): unknown => {
  if (value && typeof value === "object" && "toJs" in value) {
    const proxy = value as PyProxyLike;
    try {
      return proxy.toJs({ dict_converter: Object.fromEntries });
    } finally {
      proxy.destroy?.();
    }
  }
  return value;
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

const fetchPython = async (fileName: string) => {
  const response = await fetch(`${baseUrl}/python/${fileName}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${fileName}: ${response.status}`);
  }
  return response.text();
};

const initializeRuntime = async () => {
  postLoading("pyodide", 5);
  ctx.importScripts(`${PYODIDE_INDEX_URL}pyodide.js`);
  if (!ctx.loadPyodide) {
    throw new Error("Pyodide loader was not found on worker global scope");
  }
  pyodide = await ctx.loadPyodide({ indexURL: PYODIDE_INDEX_URL });

  postLoading("numpy", 20);
  await pyodide.loadPackage("numpy");

  postLoading("scipy", 35);
  await pyodide.loadPackage("scipy");

  postLoading("perlin-noise", 50);
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  try {
    const install = (micropip as Record<string, unknown>).install;
    if (typeof install !== "function") {
      throw new Error("micropip.install is unavailable");
    }
    await (install as (spec: string) => Promise<void>)("perlin-noise");
  } finally {
    micropip.destroy?.();
  }

  postLoading("python-files", 65);
  for (let i = 0; i < PYTHON_FILES.length; i += 1) {
    const fileName = PYTHON_FILES[i];
    const source = await fetchPython(fileName);
    pyodide.FS.writeFile(fileName, source);
    const fileProgress = 65 + Math.round(((i + 1) / PYTHON_FILES.length) * 25);
    postLoading(`python:${fileName}`, fileProgress);
  }

  pyodide.runPython(
    "import sys\nif '' not in sys.path:\n    sys.path.append('')\n",
  );
  bridgeModule = pyodide.pyimport("bridge");
  postLoading("ready", 100);
};

const ensureInitialized = async () => {
  if (bridgeModule) {
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
  if (!bridgeModule) {
    throw new Error("Python bridge is unavailable");
  }
  const rpcCall = bridgeModule.rpc_call as
    | ((method: string, paramsJson: string) => unknown)
    | undefined;
  if (!rpcCall) {
    throw new Error("bridge.rpc_call not found");
  }

  const rawResult = toPlainObject(rpcCall(action, JSON.stringify(params ?? {})));
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
      return { label: f.label, shape: [f.shape[0], f.shape[1]] as [number, number], data: typed };
    });
  }

  // Normalize quick_generate frames (each frame has multiple matrix payloads)
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

  return { snapshot, tmxBytes, frames, quickFrames, transferables };
};

ctx.onmessage = async (event: MessageEvent<WorkerRequestMessage>) => {
  const { type, requestId, params } = event.data;
  try {
    if (type === "init") {
      if (params?.baseUrl) {
        baseUrl = params.baseUrl as string;
      }
      await ensureInitialized();
      post({ type: "init_complete", requestId });
      return;
    }
    const { snapshot, tmxBytes, frames, quickFrames, transferables } = await invokeRpc(type, params);
    post(
      {
        type: "step_complete",
        requestId,
        action: type,
        snapshot,
        tmxBytes,
        frames,
        quickFrames,
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
