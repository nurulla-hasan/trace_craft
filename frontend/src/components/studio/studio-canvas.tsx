"use client";

import { useState } from "react";
import { ImagePlus, LoaderCircle, Sparkles, Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  CANVAS_BG_CLASSES,
  type CanvasBg,
  type StudioView,
} from "./types";

interface StudioCanvasProps {
  originalUrl: string | null;
  svgDataUrl: string | null;
  loading: boolean;
  view: StudioView;
  zoom: number;
  canvasBg: CanvasBg;
  onChooseImage: () => void;
  onFileDrop: (file: File) => void;
}

export function StudioCanvas({
  originalUrl,
  svgDataUrl,
  loading,
  view,
  zoom,
  canvasBg,
  onChooseImage,
  onFileDrop,
}: StudioCanvasProps) {
  const [isDragging, setIsDragging] = useState(false);
  const hasImage = Boolean(originalUrl);

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) onFileDrop(dropped);
  };

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      aria-busy={loading}
      aria-label="Studio canvas"
      className={cn(
        "relative flex min-h-120 flex-1 items-center justify-center overflow-auto p-4 pb-40 sm:p-8 sm:pb-32 lg:min-h-0 lg:pb-8",
        CANVAS_BG_CLASSES[canvasBg],
        isDragging && "ring-2 ring-primary ring-inset",
      )}
    >
      {!hasImage ? (
        <button
          type="button"
          onClick={onChooseImage}
          className="flex w-full max-w-md cursor-pointer flex-col items-center gap-3 rounded-2xl border-2 border-dashed border-studio-border bg-studio-panel/90 p-8 text-center shadow-sm transition-colors hover:border-primary/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <span className="flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <ImagePlus className="size-7" aria-hidden="true" />
          </span>
          <span className="text-base font-semibold text-studio-foreground">
            Drop an image on the canvas
          </span>
          <span className="text-sm leading-6 text-studio-muted">
            or click to browse. PNG, JPG, WebP, GIF, BMP and TIFF up to 5 MB.
          </span>
          <span className="inline-flex items-center gap-1.5 text-sm font-medium text-primary">
            <Upload className="size-4" aria-hidden="true" />
            Choose image
          </span>
        </button>
      ) : (
        <div className="flex h-full w-full items-center justify-center">
          {view === "vector" ? (
            svgDataUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={svgDataUrl}
                alt="Vectorized SVG preview"
                style={{ transform: `scale(${zoom})` }}
                className="max-h-full max-w-full object-contain transition-transform duration-200"
              />
            ) : (
              <div className="flex max-w-sm flex-col items-center gap-2 rounded-2xl bg-studio-panel/90 p-6 text-center shadow-sm">
                <Sparkles className="size-8 text-primary" aria-hidden="true" />
                <p className="text-sm font-semibold text-studio-foreground">
                  No vector yet
                </p>
                <p className="text-sm leading-6 text-studio-muted">
                  Pick a tracing mode, then press the sparkle button to
                  vectorize.
                </p>
              </div>
            )
          ) : (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={originalUrl ?? ""}
              alt="Original raster"
              style={{ transform: `scale(${zoom})` }}
              className="max-h-full max-w-full object-contain transition-transform duration-200"
            />
          )}
        </div>
      )}

      {loading && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-background/70 backdrop-blur-sm">
          <LoaderCircle
            className="size-8 animate-spin text-primary"
            aria-hidden="true"
          />
          <p className="text-sm font-medium text-foreground">
            Tracing your image…
          </p>
          <p className="text-xs text-muted-foreground">
            Separating colors and fitting Bezier curves.
          </p>
        </div>
      )}
    </div>
  );
}
