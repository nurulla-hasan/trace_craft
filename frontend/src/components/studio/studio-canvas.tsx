"use client";

import { useEffect, useRef, useState } from "react";
import { ImagePlus, LoaderCircle, Sparkles, Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  CANVAS_BG_CLASSES,
  MAX_ZOOM,
  MIN_ZOOM,
  type CanvasBg,
  type StudioView,
} from "./types";

export interface CanvasPan {
  x: number;
  y: number;
}

interface StudioCanvasProps {
  originalUrl: string | null;
  svgDataUrl: string | null;
  loading: boolean;
  view: StudioView;
  zoom: number;
  pan: CanvasPan;
  canvasBg: CanvasBg;
  onChooseImage: () => void;
  onFileDrop: (file: File) => void;
  onZoomChange: (nextZoom: number) => void;
  onPanChange: (nextPan: CanvasPan) => void;
  onResetView: () => void;
}

export function StudioCanvas({
  originalUrl,
  svgDataUrl,
  loading,
  view,
  zoom,
  pan,
  canvasBg,
  onChooseImage,
  onFileDrop,
  onZoomChange,
  onPanChange,
  onResetView,
}: StudioCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [isPanning, setIsPanning] = useState(false);

  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  const panStartRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  } | null>(null);

  useEffect(() => {
    zoomRef.current = zoom;
  }, [zoom]);

  useEffect(() => {
    panRef.current = pan;
  }, [pan]);

  const hasImage = Boolean(originalUrl);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !hasImage) return;

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();

      const currentZoom = zoomRef.current;
      const currentPan = panRef.current;

      const factor = event.ctrlKey
        ? Math.exp(-event.deltaY * 0.01)
        : event.deltaY < 0
          ? 1.12
          : 1 / 1.12;

      const nextZoom = Math.min(
        MAX_ZOOM,
        Math.max(MIN_ZOOM, Math.round(currentZoom * factor * 100) / 100),
      );

      if (nextZoom === currentZoom) return;

      const rect = container.getBoundingClientRect();
      const mouseX = event.clientX - (rect.left + rect.width / 2);
      const mouseY = event.clientY - (rect.top + rect.height / 2);

      const scaleRatio = nextZoom / currentZoom;
      const nextPan = {
        x: mouseX - (mouseX - currentPan.x) * scaleRatio,
        y: mouseY - (mouseY - currentPan.y) * scaleRatio,
      };

      onZoomChange(nextZoom);
      onPanChange(nextPan);
    };

    container.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      container.removeEventListener("wheel", handleWheel);
    };
  }, [hasImage, onZoomChange, onPanChange]);

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!hasImage || loading) return;
    if (event.button !== 0 && event.button !== 1) return;

    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    panStartRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: pan.x,
      originY: pan.y,
    };
    setIsPanning(true);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = panStartRef.current;
    if (!start || start.pointerId !== event.pointerId) return;

    const dx = event.clientX - start.startX;
    const dy = event.clientY - start.startY;
    onPanChange({
      x: start.originX + dx,
      y: start.originY + dy,
    });
  };

  const handlePointerEnd = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = panStartRef.current;
    if (!start || start.pointerId !== event.pointerId) return;

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    panStartRef.current = null;
    setIsPanning(false);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDraggingFile(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) onFileDrop(dropped);
  };

  const transformStyle: React.CSSProperties = {
    transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
    transformOrigin: "center",
  };

  return (
    <div
      ref={containerRef}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDraggingFile(true);
      }}
      onDragLeave={() => setIsDraggingFile(false)}
      onDrop={handleDrop}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerEnd}
      onPointerCancel={handlePointerEnd}
      onDoubleClick={() => {
        if (hasImage) onResetView();
      }}
      aria-busy={loading}
      aria-label="Studio canvas"
      className={cn(
        "relative flex min-h-120 flex-1 items-center justify-center overflow-hidden p-4 pb-40 select-none sm:p-8 sm:pb-32 lg:min-h-0 lg:pb-8",
        CANVAS_BG_CLASSES[canvasBg],
        hasImage && "touch-none",
        hasImage && (isPanning ? "cursor-grabbing" : "cursor-grab"),
        isDraggingFile && "ring-2 ring-primary ring-inset",
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
                draggable={false}
                style={transformStyle}
                className="pointer-events-none max-h-full max-w-full object-contain"
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
              draggable={false}
              style={transformStyle}
              className="pointer-events-none max-h-full max-w-full object-contain"
            />
          )}
        </div>
      )}

      {hasImage && !loading && (
        <p
          aria-hidden="true"
          className="pointer-events-none absolute bottom-3 left-4 hidden rounded-lg border border-studio-border/70 bg-studio-panel/80 px-2.5 py-1 font-mono text-xs text-studio-muted backdrop-blur-sm lg:block"
        >
          Scroll to zoom · Drag to pan · Double-click to reset
        </p>
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
