export type TraceMode = "auto" | "color" | "binary";
export type CanvasBg = "surface" | "muted" | "checker";
export type StudioView = "original" | "vector" | "split";
export type VectorDrawMode = "filled" | "lines" | "both";

export interface VectorizeStats {
  mode: string;
  width: number;
  height: number;
  regions: number;
  paths: number;
  bytes: number;
  duration_ms?: number;
}

export interface QuotaInfo {
  unlimited: boolean;
  remainingCredits: number | null;
}

export interface VectorizeApiResponse {
  success: boolean;
  message: string;
  data: {
    svg: string;
    stats: VectorizeStats;
    quota?: QuotaInfo;
  };
}

export const TRACE_MODES: TraceMode[] = ["auto", "color", "binary"];

export const TRACE_MODE_LABELS: Record<TraceMode, string> = {
  auto: "Auto",
  color: "Flat color",
  binary: "B&W outline",
};

export const CANVAS_BGS: CanvasBg[] = ["surface", "muted", "checker"];

export const CANVAS_BG_LABELS: Record<CanvasBg, string> = {
  surface: "Default",
  muted: "Muted",
  checker: "Checker",
};

export const CANVAS_BG_CLASSES: Record<CanvasBg, string> = {
  surface: "bg-background vk-dot-grid",
  muted: "bg-muted vk-dot-grid",
  checker: "studio-canvas-checker",
};

export const MIN_ZOOM = 0.25;
export const MAX_ZOOM = 10;
