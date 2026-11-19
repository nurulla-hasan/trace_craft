export type TraceMode = "auto" | "color" | "binary";
export type CanvasBg = "white" | "dark" | "checker";
export type StudioView = "original" | "vector";

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

export const CANVAS_BGS: CanvasBg[] = ["white", "dark", "checker"];

export const CANVAS_BG_LABELS: Record<CanvasBg, string> = {
  white: "White",
  dark: "Dark",
  checker: "Grid",
};

export const CANVAS_BG_CLASSES: Record<CanvasBg, string> = {
  white: "bg-canvas-white",
  dark: "bg-canvas-dark",
  checker: "studio-canvas-checker",
};
