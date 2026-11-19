"use client";

import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface DockButtonProps {
  label: string;
  children: ReactNode;
  onClick?: () => void;
  variant?: "ghost" | "outline" | "default" | "secondary" | "destructive";
  size?: "icon" | "icon-lg";
  disabled?: boolean;
  loading?: boolean;
  pressed?: boolean;
  className?: string;
}

export function DockButton({
  label,
  children,
  onClick,
  variant = "ghost",
  size = "icon",
  disabled,
  loading,
  pressed,
  className,
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
            className={className}
          />
        }
      >
        {children}
      </TooltipTrigger>
      <TooltipContent side="top">{label}</TooltipContent>
    </Tooltip>
  );
}

export function DockDivider() {
  return (
    <span
      aria-hidden="true"
      className="hidden h-6 w-px shrink-0 bg-studio-muted/40 sm:block lg:h-px lg:w-6"
    />
  );
}
