"use client";

import { useState } from "react";
import { Crown, Info } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { QuotaInfo, VectorizeStats } from "./types";

interface StudioStatsPopoverProps {
  file: File | null;
  quota: QuotaInfo | null;
  stats: VectorizeStats | null;
  isDesktop: boolean;
}

export function StudioStatsPopover({
  file,
  quota,
  stats,
  isDesktop,
}: StudioStatsPopoverProps) {
  const [open, setOpen] = useState(false);

  return (
    <Popover open={open} onOpenChange={setOpen}>
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
              <Badge variant="outline">{quota.remainingCredits} credits</Badge>
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
  );
}
