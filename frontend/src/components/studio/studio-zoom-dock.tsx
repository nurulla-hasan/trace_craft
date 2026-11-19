"use client";

import { ZoomIn, ZoomOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { DockButton } from "./studio-dock-primitives";
import { MAX_ZOOM, MIN_ZOOM } from "./types";

interface StudioZoomDockProps {
  zoom: number;
  onZoomChange: (zoom: number) => void;
  onResetView: () => void;
}

export function StudioZoomDock({
  zoom,
  onZoomChange,
  onResetView,
}: StudioZoomDockProps) {
  return (
    <div
      className={cn(
        "absolute right-3 bottom-28 z-20 flex items-center gap-1 rounded-2xl border border-studio-border bg-studio-panel/95 p-1 shadow-xl backdrop-blur-md",
        "sm:bottom-20 lg:inset-x-0 lg:bottom-4 lg:mx-auto lg:w-fit",
      )}
    >
      <DockButton
        label="Zoom out"
        onClick={() => onZoomChange(zoom - 0.25)}
        disabled={zoom <= MIN_ZOOM}
      >
        <ZoomOut />
      </DockButton>
      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              type="button"
              variant="ghost"
              size="default"
              onClick={onResetView}
              aria-label="Reset zoom and pan"
            />
          }
        >
          <span
            aria-live="polite"
            className="w-12 text-center font-mono text-xs text-studio-foreground select-none"
          >
            {Math.round(zoom * 100)}%
          </span>
        </TooltipTrigger>
        <TooltipContent side="top">Reset view (100%)</TooltipContent>
      </Tooltip>
      <DockButton
        label="Zoom in"
        onClick={() => onZoomChange(zoom + 0.25)}
        disabled={zoom >= MAX_ZOOM}
      >
        <ZoomIn />
      </DockButton>
    </div>
  );
}
