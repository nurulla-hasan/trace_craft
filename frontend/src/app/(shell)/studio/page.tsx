"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  MAX_ZOOM,
  MIN_ZOOM,
  StudioCanvas,
  StudioControls,
} from "@/components/studio";
import type {
  CanvasBg,
  CanvasPan,
  QuotaInfo,
  StudioView,
  TraceMode,
  VectorizeApiResponse,
  VectorizeStats,
} from "@/components/studio";
import apiClient from "@/lib/api-client";
import { getErrorMessage } from "@/lib/error";
import { WORKSPACE_HEIGHT } from "@/lib/layout";
import { errorToast, successToast } from "@/lib/toast";
import { cn } from "@/lib/utils";

export default function StudioPage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const copyTimeoutRef = useRef<number | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [mode, setMode] = useState<TraceMode>("auto");
  const [transparentBg, setTransparentBg] = useState<boolean>(false);
  const [canvasBg, setCanvasBg] = useState<CanvasBg>("surface");
  const [view, setView] = useState<StudioView>("original");
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<CanvasPan>({ x: 0, y: 0 });
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [svgOutput, setSvgOutput] = useState<string | null>(null);
  const [stats, setStats] = useState<VectorizeStats | null>(null);
  const [quota, setQuota] = useState<QuotaInfo | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  const svgDataUrl = useMemo(() => {
    if (!svgOutput) return null;
    const blob = new Blob([svgOutput], { type: "image/svg+xml;charset=utf-8" });
    return URL.createObjectURL(blob);
  }, [svgOutput]);

  useEffect(() => {
    if (!originalUrl) return;
    return () => URL.revokeObjectURL(originalUrl);
  }, [originalUrl]);

  useEffect(() => {
    if (!svgDataUrl) return;
    return () => URL.revokeObjectURL(svgDataUrl);
  }, [svgDataUrl]);

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) {
        window.clearTimeout(copyTimeoutRef.current);
      }
    };
  }, []);

  const handleFileSelect = useCallback((selected: File | null) => {
    if (!selected) return;
    setError(null);
    setSvgOutput(null);
    setStats(null);
    setQuota(null);
    setFile(selected);
    setOriginalUrl(URL.createObjectURL(selected));
    setView("original");
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const handleClearImage = useCallback(() => {
    setFile(null);
    setOriginalUrl(null);
    setSvgOutput(null);
    setStats(null);
    setError(null);
    setView("original");
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const handleChooseImage = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleZoomChange = useCallback((next: number) => {
    setZoom(Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Math.round(next * 100) / 100)));
  }, []);

  const handleResetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const handleVectorize = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("mode", mode);
      formData.append("includeBg", transparentBg ? "false" : "true");

      const response = await apiClient<VectorizeApiResponse>("/vectorize", {
        method: "POST",
        body: formData,
      });

      setSvgOutput(response.data.svg);
      setStats(response.data.stats);
      setQuota(response.data.quota ?? null);
      setView("vector");
    } catch (err: unknown) {
      const message = getErrorMessage(
        err,
        "Vectorization failed. Please check that the engine service is running.",
      );
      setError(message);
      errorToast("Vectorization failed", message);
    } finally {
      setLoading(false);
    }
  }, [file, mode, transparentBg]);

  const handleDownloadSvg = useCallback(() => {
    if (!svgOutput) return;
    const blob = new Blob([svgOutput], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const baseName = file?.name?.replace(/\.[^.]+$/, "") || "vectorized";
    anchor.href = url;
    anchor.download = `${baseName}.svg`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [svgOutput, file]);

  const handleDownloadPng4x = useCallback(() => {
    if (!svgOutput || !stats) return;
    const scale = 4;
    const image = new Image();
    const blob = new Blob([svgOutput], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);

    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = stats.width * scale;
      canvas.height = stats.height * scale;
      const context = canvas.getContext("2d");
      if (!context) {
        errorToast("PNG export failed", "Canvas rendering is unavailable.");
        URL.revokeObjectURL(url);
        return;
      }

      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((pngBlob) => {
        if (!pngBlob) {
          errorToast("PNG export failed", "Could not encode the canvas.");
          URL.revokeObjectURL(url);
          return;
        }

        const pngUrl = URL.createObjectURL(pngBlob);
        const anchor = document.createElement("a");
        const baseName = file?.name?.replace(/\.[^.]+$/, "") || "vectorized";
        anchor.href = pngUrl;
        anchor.download = `${baseName}-4x.png`;
        anchor.click();
        URL.revokeObjectURL(pngUrl);
        URL.revokeObjectURL(url);
      }, "image/png");
    };

    image.onerror = () => {
      errorToast("PNG export failed", "The SVG preview could not be rendered.");
      URL.revokeObjectURL(url);
    };

    image.src = url;
  }, [svgOutput, stats, file]);

  const handleCopy = useCallback(async () => {
    if (!svgOutput) return;
    try {
      await navigator.clipboard.writeText(svgOutput);
      setCopied(true);
      successToast("SVG copied to clipboard");
      if (copyTimeoutRef.current) {
        window.clearTimeout(copyTimeoutRef.current);
      }
      copyTimeoutRef.current = window.setTimeout(() => setCopied(false), 1800);
    } catch (err: unknown) {
      errorToast(
        "Copy failed",
        getErrorMessage(err, "Clipboard access was blocked."),
      );
    }
  }, [svgOutput]);

  return (
    <div
      className={cn(
        "relative flex min-h-0 flex-col bg-studio-surface lg:flex-row lg:overflow-hidden",
        WORKSPACE_HEIGHT,
      )}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/gif,image/bmp,image/tiff"
        className="hidden"
        onChange={(event) => {
          handleFileSelect(event.target.files?.[0] ?? null);
          event.target.value = "";
        }}
      />

      <StudioCanvas
        originalUrl={originalUrl}
        svgDataUrl={svgDataUrl}
        loading={loading}
        view={view}
        zoom={zoom}
        pan={pan}
        canvasBg={canvasBg}
        onChooseImage={handleChooseImage}
        onFileDrop={handleFileSelect}
        onZoomChange={handleZoomChange}
        onPanChange={setPan}
        onResetView={handleResetView}
      />

      <StudioControls
        file={file}
        quota={quota}
        error={error}
        loading={loading}
        mode={mode}
        transparentBg={transparentBg}
        canvasBg={canvasBg}
        view={view}
        zoom={zoom}
        copied={copied}
        stats={stats}
        hasSvg={Boolean(svgOutput)}
        onChooseImage={handleChooseImage}
        onClearImage={handleClearImage}
        onModeChange={setMode}
        onTransparentBgChange={setTransparentBg}
        onVectorize={handleVectorize}
        onViewChange={setView}
        onCanvasBgChange={setCanvasBg}
        onZoomChange={handleZoomChange}
        onResetView={handleResetView}
        onCopy={handleCopy}
        onDownloadSvg={handleDownloadSvg}
        onDownloadPng={handleDownloadPng4x}
      />
    </div>
  );
}
