"use client";

import { useState, useSyncExternalStore, type ReactNode } from "react";
import Link from "next/link";
import {
  Check,
  CircleAlert,
  Contrast,
  Copy,
  Crown,
  Download,
  Eraser,
  Image as ImageIcon,
  ImageDown,
  Info,
  PaintBucket,
  Palette,
  Shapes,
  Sparkles,
  Trash,
  Upload,
  WandSparkles,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  CANVAS_BGS,
  CANVAS_BG_CLASSES,
  CANVAS_BG_LABELS,
  TRACE_MODES,
  TRACE_MODE_LABELS,
  type CanvasBg,
  type QuotaInfo,
  type StudioView,
  type TraceMode,
  type VectorizeStats,
} from "./types";

const TRACE_MODE_ICONS = {
  auto: WandSparkles,
  color: Palette,
  binary: Contrast,
} as const;

const DESKTOP_MEDIA_QUERY = "(min-width: 1024px)";

function subscribeToDesktop(callback: () => void) {
  const media = window.matchMedia(DESKTOP_MEDIA_QUERY);
  media.addEventListener("change", callback);
  return () => media.removeEventListener("change", callback);
}

function getDesktopSnapshot() {
  return window.matchMedia(DESKTOP_MEDIA_QUERY).matches;
}

function getDesktopServerSnapshot() {
  return false;
}

function useIsDesktop() {
  return useSyncExternalStore(
    subscribeToDesktop,
    getDesktopSnapshot,
    getDesktopServerSnapshot,
  );
}

interface StudioControlsProps {
  file: File | null;
  quota: QuotaInfo | null;
  error: string | null;
  loading: boolean;
  mode: TraceMode;
  transparentBg: boolean;
  canvasBg: CanvasBg;
  view: StudioView;
  zoom: number;
  copied: boolean;
  stats: VectorizeStats | null;
  hasSvg: boolean;
  onChooseImage: () => void;
  onClearImage: () => void;
  onModeChange: (mode: TraceMode) => void;
  onTransparentBgChange: (value: boolean) => void;
  onVectorize: () => void;
  onViewChange: (view: StudioView) => void;
  onCanvasBgChange: (bg: CanvasBg) => void;
  onZoomChange: (zoom: number) => void;
  onCopy: () => void;
  onDownloadSvg: () => void;
  onDownloadPng: () => void;
}

interface DockButtonProps {
  label: string;
  children: ReactNode;
  onClick?: () => void;
  variant?: "ghost" | "outline" | "default" | "secondary" | "destructive";
  size?: "icon" | "icon-lg";
  disabled?: boolean;
  loading?: boolean;
  pressed?: boolean;
}

function DockButton({
  label,
  children,
  onClick,
  variant = "ghost",
  size = "icon",
  disabled,
  loading,
  pressed,
}: DockButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            variant={variant}
            size={size}
            onClick={onClick}
            disabled={disabled}
            loading={loading}
            loadingText=""
            aria-label={label}
            aria-pressed={pressed}
          />
        }
      >
        {children}
      </TooltipTrigger>
      <TooltipContent side="top">{label}</TooltipContent>
    </Tooltip>
  );
}

function DockDivider() {
  return (
    <span
      aria-hidden="true"
      className="hidden h-6 w-px shrink-0 bg-studio-muted/40 sm:block lg:h-px lg:w-6"
    />
  );
}

export function StudioControls({
  file,
  quota,
  error,
  loading,
  mode,
  transparentBg,
  canvasBg,
  view,
  zoom,
  copied,
  stats,
  hasSvg,
  onChooseImage,
  onClearImage,
  onModeChange,
  onTransparentBgChange,
  onVectorize,
  onViewChange,
  onCanvasBgChange,
  onZoomChange,
  onCopy,
  onDownloadSvg,
  onDownloadPng,
}: StudioControlsProps) {
  const [bgOpen, setBgOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [statsOpen, setStatsOpen] = useState(false);
  const isDesktop = useIsDesktop();

  return (
    <>
      {error && (
        <div
          role="alert"
          className="absolute inset-x-3 top-3 z-30 mx-auto flex max-w-md flex-col gap-2 rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive shadow-lg backdrop-blur-md lg:top-4"
        >
          <div className="flex items-start gap-2">
            <CircleAlert
              className="mt-0.5 size-4 shrink-0"
              aria-hidden="true"
            />
            <p className="leading-6">{error}</p>
          </div>
          {error.toLowerCase().includes("credit") && (
            <Button
              variant="outline"
              size="default"
              className="w-full"
              render={<Link href="/plans" />}
            >
              Upgrade plan
            </Button>
          )}
        </div>
      )}

      <aside
        aria-label="Studio controls"
        className={cn(
          "absolute inset-x-3 bottom-3 z-20 flex flex-wrap items-center justify-center gap-1 rounded-2xl border border-studio-border bg-studio-panel/95 p-1.5 shadow-xl backdrop-blur-md",
          "sm:flex-nowrap lg:inset-x-auto lg:top-1/2 lg:right-4 lg:bottom-auto lg:max-h-full lg:-translate-y-1/2 lg:flex-col lg:flex-nowrap lg:overflow-y-auto",
        )}
      >
        <DockButton
          label={file ? "Replace image" : "Upload image"}
          variant="outline"
          onClick={onChooseImage}
          disabled={loading}
        >
          <Upload />
        </DockButton>

        {file && (
          <DockButton
            label="Remove image"
            variant="destructive"
            onClick={onClearImage}
            disabled={loading}
          >
            <Trash />
          </DockButton>
        )}

        <DockDivider />

        <DockButton
          label="Vectorize"
          variant="default"
          size="icon-lg"
          onClick={onVectorize}
          disabled={!file}
          loading={loading}
        >
          <Sparkles />
        </DockButton>

        <DockDivider />

        {TRACE_MODES.map((option) => {
          const ModeIcon = TRACE_MODE_ICONS[option];
          return (
            <DockButton
              key={option}
              label={`Trace: ${TRACE_MODE_LABELS[option]}`}
              variant={mode === option ? "default" : "ghost"}
              onClick={() => onModeChange(option)}
              disabled={loading}
              pressed={mode === option}
            >
              <ModeIcon />
            </DockButton>
          );
        })}

        <DockButton
          label="Transparent background"
          variant={transparentBg ? "secondary" : "ghost"}
          onClick={() => onTransparentBgChange(!transparentBg)}
          disabled={loading}
          pressed={transparentBg}
        >
          <Eraser />
        </DockButton>

        <DockDivider />

        <DockButton
          label="Original view"
          variant={view === "original" ? "secondary" : "ghost"}
          onClick={() => onViewChange("original")}
          pressed={view === "original"}
        >
          <ImageIcon />
        </DockButton>

        <DockButton
          label="Vector view"
          variant={view === "vector" ? "secondary" : "ghost"}
          onClick={() => onViewChange("vector")}
          disabled={!hasSvg && !loading}
          pressed={view === "vector"}
        >
          <Shapes />
        </DockButton>

        <Popover open={bgOpen} onOpenChange={setBgOpen}>
          <PopoverTrigger
            render={
              <Button
                variant="ghost"
                size="icon"
                aria-label={`Canvas background: ${CANVAS_BG_LABELS[canvasBg]}`}
                title="Canvas background"
              />
            }
          >
            <PaintBucket />
          </PopoverTrigger>
          <PopoverContent
            side={isDesktop ? "left" : "top"}
            align={isDesktop ? "center" : "end"}
            sideOffset={isDesktop ? 12 : 8}
            className="w-auto"
          >
            <div className="flex items-center gap-1">
              {CANVAS_BGS.map((option) => {
                return (
                  <Tooltip key={option}>
                    <TooltipTrigger
                      render={
                        <Button
                          variant={canvasBg === option ? "secondary" : "ghost"}
                          size="icon"
                          aria-label={`${CANVAS_BG_LABELS[option]} canvas`}
                          aria-pressed={canvasBg === option}
                          onClick={() => {
                            onCanvasBgChange(option);
                            setBgOpen(false);
                          }}
                        />
                      }
                    >
                      <span
                        className={cn(
                          "size-6 rounded-md border border-border",
                          CANVAS_BG_CLASSES[option],
                        )}
                      />
                    </TooltipTrigger>
                    <TooltipContent side="top">
                      {CANVAS_BG_LABELS[option]} canvas
                    </TooltipContent>
                  </Tooltip>
                );
              })}
            </div>
          </PopoverContent>
        </Popover>

        <DockDivider />

        <Popover open={exportOpen} onOpenChange={setExportOpen}>
          <PopoverTrigger
            render={
              <Button
                variant="ghost"
                size="icon"
                aria-label="Export"
                title="Export"
              />
            }
          >
            <Download />
          </PopoverTrigger>
          <PopoverContent
            side={isDesktop ? "left" : "top"}
            align={isDesktop ? "center" : "end"}
            sideOffset={isDesktop ? 12 : 8}
            className="w-auto"
          >
            <div className="flex items-center gap-1">
              <DockButton
                label={copied ? "Copied" : "Copy SVG"}
                onClick={onCopy}
                disabled={!hasSvg}
              >
                {copied ? <Check /> : <Copy />}
              </DockButton>
              <DockButton
                label="Download SVG"
                onClick={onDownloadSvg}
                disabled={!hasSvg}
              >
                <Download />
              </DockButton>
              <DockButton
                label="Download PNG (4×)"
                onClick={onDownloadPng}
                disabled={!hasSvg || !stats}
              >
                <ImageDown />
              </DockButton>
            </div>
          </PopoverContent>
        </Popover>

        <Popover open={statsOpen} onOpenChange={setStatsOpen}>
          <PopoverTrigger
            render={
              <Button
                variant="ghost"
                size="icon"
                aria-label="Output details"
                title="Output details"
              />
            }
          >
            <Info />
          </PopoverTrigger>
          <PopoverContent
            side={isDesktop ? "left" : "top"}
            align={isDesktop ? "center" : "end"}
            sideOffset={isDesktop ? 12 : 8}
          >
            <div className="flex items-center justify-between gap-2">
              <p className="text-sm font-semibold text-popover-foreground">
                Output
              </p>
              {quota &&
                (quota.unlimited ? (
                  <Badge variant="default">
                    <Crown />
                    Unlimited
                  </Badge>
                ) : quota.remainingCredits !== null ? (
                  <Badge variant="outline">
                    {quota.remainingCredits} credits
                  </Badge>
                ) : null)}
            </div>

            {file && (
              <p className="truncate font-mono text-xs text-muted-foreground">
                {file.name}
              </p>
            )}

            {stats ? (
              <dl className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-lg bg-muted p-2">
                  <dt className="text-muted-foreground">Canvas</dt>
                  <dd className="mt-0.5 font-mono">
                    {stats.width} × {stats.height}
                  </dd>
                </div>
                <div className="rounded-lg bg-muted p-2">
                  <dt className="text-muted-foreground">Regions</dt>
                  <dd className="mt-0.5 font-mono">{stats.regions}</dd>
                </div>
                <div className="rounded-lg bg-muted p-2">
                  <dt className="text-muted-foreground">Paths</dt>
                  <dd className="mt-0.5 font-mono">{stats.paths}</dd>
                </div>
                <div className="rounded-lg bg-muted p-2">
                  <dt className="text-muted-foreground">File size</dt>
                  <dd className="mt-0.5 font-mono">
                    {(stats.bytes / 1024).toFixed(1)} KB
                  </dd>
                </div>
                <div className="rounded-lg bg-muted p-2">
                  <dt className="text-muted-foreground">Mode</dt>
                  <dd className="mt-0.5 font-mono">{stats.mode}</dd>
                </div>
                <div className="rounded-lg bg-muted p-2">
                  <dt className="text-muted-foreground">Time</dt>
                  <dd className="mt-0.5 font-mono">
                    {stats.duration_ms !== undefined
                      ? `${(stats.duration_ms / 1000).toFixed(2)}s`
                      : "—"}
                  </dd>
                </div>
              </dl>
            ) : (
              <p className="text-xs leading-5 text-muted-foreground">
                Vectorize an image to see output details here.
              </p>
            )}
          </PopoverContent>
        </Popover>
      </aside>

      <div
        className={cn(
          "absolute right-3 bottom-28 z-20 flex items-center gap-1 rounded-2xl border border-studio-border bg-studio-panel/95 p-1 shadow-xl backdrop-blur-md",
          "sm:bottom-20 lg:inset-x-0 lg:bottom-4 lg:mx-auto lg:w-fit",
        )}
      >
        <DockButton
          label="Zoom out"
          onClick={() => onZoomChange(zoom - 0.25)}
          disabled={zoom <= 0.25}
        >
          <ZoomOut />
        </DockButton>
        <span
          aria-live="polite"
          className="w-11 text-center font-mono text-xs text-studio-foreground select-none"
        >
          {Math.round(zoom * 100)}%
        </span>
        <DockButton
          label="Zoom in"
          onClick={() => onZoomChange(zoom + 0.25)}
          disabled={zoom >= 4}
        >
          <ZoomIn />
        </DockButton>
      </div>
    </>
  );
}
