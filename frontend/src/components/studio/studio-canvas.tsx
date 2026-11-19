"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ImagePlus, LoaderCircle, Sparkles, Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  CANVAS_BG_CLASSES,
  MAX_ZOOM,
  MIN_ZOOM,
  type CanvasBg,
} from "./types";

export interface CanvasPan {
  x: number;
  y: number;
}

interface StudioCanvasProps {
  originalUrl: string | null;
  svgContent: string | null;
  loading: boolean;
  zoom: number;
  pan: CanvasPan;
  canvasBg: CanvasBg;
  onChooseImage: () => void;
  onFileDrop: (file: File) => void;
  onZoomChange: (nextZoom: number) => void;
  onPanChange: (nextPan: CanvasPan) => void;
  onResetView: () => void;
}

// Vectorizer.ai LoupeCutoffScale: lens appears automatically at 400% - 500% zoom
const LOUPE_CUTOFF_ZOOM = 4.0;
const LENS_RADIUS = 105;

export function StudioCanvas({
  originalUrl,
  svgContent,
  loading,
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
  const stageRef = useRef<HTMLDivElement | null>(null);

  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const [stageRect, setStageRect] = useState<{
    left: number;
    top: number;
    width: number;
    height: number;
  } | null>(null);

  // High performance animation refs: isolates interaction from React re-renders
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  const panStartRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  } | null>(null);

  const rafIdRef = useRef<number | null>(null);
  const syncTimeoutRef = useRef<number | null>(null);

  // Sync internal refs with props when props change externally (e.g. zoom buttons, reset view)
  useEffect(() => {
    zoomRef.current = zoom;
    panRef.current = pan;
    if (stageRef.current) {
      stageRef.current.style.transform = `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`;
    }
  }, [zoom, pan]);

  const hasImage = Boolean(originalUrl);
  const hasSvg = Boolean(svgContent);

  // Natural aspect ratio from SVG viewBox
  const aspectRatio = useMemo(() => {
    if (svgContent) {
      const vb = svgContent.match(
        /viewBox=["']([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)["']/,
      );
      if (vb) {
        const w = parseFloat(vb[3]);
        const h = parseFloat(vb[4]);
        if (w > 0 && h > 0) return `${w} / ${h}`;
      }
    }
    return "4 / 3";
  }, [svgContent]);

  // Update stage geometry for 500% zoom lens overlay alignment
  const updateStageGeometry = () => {
    const container = containerRef.current;
    const stage = stageRef.current;
    if (!container || !stage) {
      setStageRect(null);
      return;
    }
    const cRect = container.getBoundingClientRect();
    const sRect = stage.getBoundingClientRect();
    setStageRect({
      left: sRect.left - cRect.left,
      top: sRect.top - cRect.top,
      width: sRect.width,
      height: sRect.height,
    });
  };

  // Ultra-smooth mouse wheel zoom anchored at cursor
  // Directly modifies stage DOM transform in RAF for zero-latency 120 FPS
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
          ? 1.15
          : 1 / 1.15;

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

      zoomRef.current = nextZoom;
      panRef.current = nextPan;

      if (stageRef.current) {
        stageRef.current.style.transform = `translate3d(${nextPan.x}px, ${nextPan.y}px, 0) scale(${nextZoom})`;
      }

      updateStageGeometry();

      // Debounce notifying parent so React does not re-render on every micro-wheel tick
      if (syncTimeoutRef.current) {
        window.clearTimeout(syncTimeoutRef.current);
      }
      syncTimeoutRef.current = window.setTimeout(() => {
        onZoomChange(nextZoom);
        onPanChange(nextPan);
      }, 50);
    };

    container.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      container.removeEventListener("wheel", handleWheel);
      if (syncTimeoutRef.current) window.clearTimeout(syncTimeoutRef.current);
    };
  }, [hasImage, onZoomChange, onPanChange]);

  // Pointer event handlers for butter-smooth panning & lens tracking
  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!hasImage || loading) return;
    if (event.button !== 0 && event.button !== 1) return;

    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    panStartRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: panRef.current.x,
      originY: panRef.current.y,
    };
    setIsPanning(true);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    // 1. Pan dragging with RAF — ZERO React state changes or re-renders!
    const panStart = panStartRef.current;
    if (panStart && panStart.pointerId === event.pointerId) {
      const dx = event.clientX - panStart.startX;
      const dy = event.clientY - panStart.startY;
      const nextX = panStart.originX + dx;
      const nextY = panStart.originY + dy;

      panRef.current = { x: nextX, y: nextY };

      if (!rafIdRef.current) {
        rafIdRef.current = window.requestAnimationFrame(() => {
          rafIdRef.current = null;
          if (stageRef.current) {
            stageRef.current.style.transform = `translate3d(${nextX}px, ${nextY}px, 0) scale(${zoomRef.current})`;
          }
          updateStageGeometry();
        });
      }
      return;
    }

    // 2. Track mouse position for 500% zoom inspection lens
    if (containerRef.current) {
      const cRect = containerRef.current.getBoundingClientRect();
      const lx = event.clientX - cRect.left;
      const ly = event.clientY - cRect.top;
      setMousePos({ x: lx, y: ly });
      updateStageGeometry();
    }
  };

  const handlePointerEnd = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = panStartRef.current;
    if (!start || start.pointerId !== event.pointerId) return;

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    panStartRef.current = null;
    setIsPanning(false);

    // Commit final coordinates once upon release
    onPanChange({ ...panRef.current });
    updateStageGeometry();
  };

  const handlePointerLeave = () => {
    setMousePos(null);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDraggingFile(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) onFileDrop(dropped);
  };

  // Check if 500% zoom inspection lens should be active
  const showInspectionLens =
    hasSvg &&
    hasImage &&
    !loading &&
    !isPanning &&
    zoom >= LOUPE_CUTOFF_ZOOM &&
    mousePos !== null &&
    stageRect !== null;

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
      onPointerLeave={handlePointerLeave}
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
      {/* Top Vectorizer Result Badge */}
      {hasImage && !loading && (
        <div className="pointer-events-none absolute top-4 inset-x-0 z-20 flex justify-center">
          <div className="flex items-center gap-2 rounded-full border border-studio-border/70 bg-studio-panel/90 px-3.5 py-1 text-xs font-medium text-studio-foreground shadow-md backdrop-blur-md transition-all">
            <span className="size-2 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee] animate-pulse" />
            <span>
              {hasSvg ? "Vectorizer.AI Result" : "Original Image"}
            </span>
            {hasSvg && zoom >= LOUPE_CUTOFF_ZOOM && (
              <span className="rounded bg-cyan-950/60 px-1.5 py-0.5 text-[10px] font-mono text-cyan-300 border border-cyan-400/40">
                Pixel &amp; Line Inspection ({Math.round(zoom * 100)}%)
              </span>
            )}
          </div>
        </div>
      )}

      {/* Main Empty / Canvas Content */}
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
          {/* Main Stage with Hardware GPU Compositor Transform */}
          <div
            ref={stageRef}
            style={{
              aspectRatio: aspectRatio,
              transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
              transformOrigin: "center",
            }}
            className="relative flex h-auto max-h-[80vh] w-auto max-w-[85vw] items-center justify-center select-none"
          >
            {/* ONLY VECTOR VIEW: Clean, full-color mathematical vector graphics */}
            {hasSvg ? (
              <div
                className="vector-canvas-svg h-full w-full select-none"
                dangerouslySetInnerHTML={{ __html: svgContent }}
              />
            ) : (
              // If not yet vectorized: show uploaded original raster
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={originalUrl ?? ""}
                alt="Original"
                draggable={false}
                style={{
                  width: "100%",
                  height: "100%",
                }}
                className="pointer-events-none block h-full w-full object-contain"
              />
            )}
          </div>
        </div>
      )}

      {/* 500% ZOOM INSPECTION LENS: Exact replica of Vectorizer.ai (Screenshot 2) */}
      {/* Appears automatically at >= 400%-500% zoom, showing pixelated raster + cyan vector lines */}
      {showInspectionLens && mousePos && stageRect && (
        <div
          aria-hidden="true"
          style={{
            width: LENS_RADIUS * 2,
            height: LENS_RADIUS * 2,
            transform: `translate3d(${mousePos.x - LENS_RADIUS}px, ${mousePos.y - LENS_RADIUS}px, 0)`,
          }}
          className="pointer-events-none absolute top-0 left-0 z-30 select-none overflow-hidden rounded-full border-[2.5px] border-cyan-400 bg-background/90 shadow-[0_0_0_1px_rgba(0,242,254,0.6),0_0_24px_6px_rgba(0,242,254,0.55),inset_0_0_16px_rgba(0,242,254,0.25)]"
        >
          {/* Inner content wrapper aligned 100% seamlessly to the outer canvas coordinate */}
          <div
            style={{
              width: stageRect.width,
              height: stageRect.height,
              transform: `translate3d(${stageRect.left - (mousePos.x - LENS_RADIUS)}px, ${stageRect.top - (mousePos.y - LENS_RADIUS)}px, 0)`,
            }}
            className="relative h-full w-full"
          >
            {/* 1. Raw Original Raster Bitmap with crisp pixel blocks */}
            {originalUrl && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={originalUrl}
                alt=""
                draggable={false}
                style={{
                  imageRendering: "pixelated",
                }}
                className="pointer-events-none absolute inset-0 block h-full w-full object-contain"
              />
            )}

            {/* 2. Fitted Cyan Vector Lines Superimposed over the pixels */}
            {svgContent && (
              <div
                className="vector-canvas-svg vector-lens-lines pointer-events-none absolute inset-0 h-full w-full"
                dangerouslySetInnerHTML={{ __html: svgContent }}
              />
            )}
          </div>
        </div>
      )}

      {/* Status Bar / Hints */}
      {hasImage && !loading && (
        <p
          aria-hidden="true"
          className="pointer-events-none absolute bottom-3 left-4 hidden rounded-lg border border-studio-border/70 bg-studio-panel/80 px-2.5 py-1 font-mono text-xs text-studio-muted backdrop-blur-sm lg:block"
        >
          Scroll to zoom · Drag to pan · Double-click to reset · Zoom &gt; 400% reveals pixel &amp; vector lines
        </p>
      )}

      {/* Loading Overlay */}
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
