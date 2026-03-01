"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  WorkerAction,
  WorkerRequestMessage,
  WorkerResponseMessage,
  WorkerStepCompleteMessage,
  WizardSnapshot,
} from "@/lib/types";

type PendingRequest = {
  resolve: (value: WorkerStepCompleteMessage | void) => void;
  reject: (reason?: unknown) => void;
};

export type WorkerActionParams = Record<string, unknown>;

export interface UsePyodideResult {
  loading: boolean;
  ready: boolean;
  error: string | null;
  loadingStage: string;
  loadingProgress: number;
  snapshot: WizardSnapshot | null;
  callAction: (
    action: Exclude<WorkerAction, "init">,
    params?: WorkerActionParams,
  ) => Promise<WorkerStepCompleteMessage>;
}

export function usePyodide(): UsePyodideResult {
  const workerRef = useRef<Worker | null>(null);
  const requestCounterRef = useRef(0);
  const pendingRef = useRef<Map<string, PendingRequest>>(new Map());
  const initPromiseRef = useRef<Promise<void> | null>(null);

  const [loading, setLoading] = useState(true);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingStage, setLoadingStage] = useState("starting");
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [snapshot, setSnapshot] = useState<WizardSnapshot | null>(null);

  const sendRequest = useCallback(
    (type: WorkerAction, params?: WorkerActionParams) => {
      const worker = workerRef.current;
      if (!worker) {
        return Promise.reject(new Error("Pyodide worker is not initialized"));
      }
      const requestId = `${Date.now()}-${requestCounterRef.current++}`;
      const message: WorkerRequestMessage = { type, requestId, params };
      return new Promise<WorkerStepCompleteMessage | void>((resolve, reject) => {
        pendingRef.current.set(requestId, { resolve, reject });
        worker.postMessage(message);
      });
    },
    [],
  );

  useEffect(() => {
    const worker = new Worker(
      new URL("../workers/pyodide.worker.ts", import.meta.url),
    );
    workerRef.current = worker;

    const handleMessage = (event: MessageEvent<WorkerResponseMessage>) => {
      const message = event.data;
      if (message.type === "loading") {
        setLoading(true);
        setLoadingStage(message.stage);
        setLoadingProgress(message.progress);
        return;
      }

      if (message.type === "init_complete") {
        setLoading(false);
        setReady(true);
        setError(null);
        setLoadingStage("ready");
        setLoadingProgress(100);
        if (message.requestId) {
          const pending = pendingRef.current.get(message.requestId);
          if (pending) {
            pendingRef.current.delete(message.requestId);
            pending.resolve();
          }
        }
        return;
      }

      if (message.type === "step_complete") {
        if (message.snapshot) {
          setSnapshot(message.snapshot);
        }
        const pending = pendingRef.current.get(message.requestId);
        if (pending) {
          pendingRef.current.delete(message.requestId);
          pending.resolve(message);
        }
        return;
      }

      setLoading(false);
      setError(message.message);
      if (message.requestId) {
        const pending = pendingRef.current.get(message.requestId);
        if (pending) {
          pendingRef.current.delete(message.requestId);
          pending.reject(new Error(message.message));
        }
      }
    };

    worker.addEventListener("message", handleMessage);
    initPromiseRef.current = sendRequest("init", {
      baseUrl: window.location.origin,
    }).then(() => undefined);
    const pendingRequests = pendingRef.current;

    return () => {
      worker.removeEventListener("message", handleMessage);
      worker.terminate();
      workerRef.current = null;
      for (const pending of pendingRequests.values()) {
        pending.reject(new Error("Worker disposed"));
      }
      pendingRequests.clear();
    };
  }, [sendRequest]);

  const callAction = useCallback(
    async (
      action: Exclude<WorkerAction, "init">,
      params?: WorkerActionParams,
    ) => {
      if (initPromiseRef.current) {
        await initPromiseRef.current;
      }
      const response = await sendRequest(action, params);
      return response as WorkerStepCompleteMessage;
    },
    [sendRequest],
  );

  return {
    loading,
    ready,
    error,
    loadingStage,
    loadingProgress,
    snapshot,
    callAction,
  };
}
