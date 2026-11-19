"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, Check, Crown, Zap } from "lucide-react";
import { UpgradePlanDialog } from "@/components/subscription/upgrade-plan-dialog";
import { cn } from "@/lib/utils";
import type { AuthUser, BillingCycle, PlanDefinition } from "@/services";

const EASE_OUT_EXPO: [number, number, number, number] = [0.16, 1, 0.3, 1];
const formatBdt = (amount: number) => `৳${amount.toLocaleString("en-US")}`;

interface PlanCardProps {
  plan: PlanDefinition;
  index: number;
  user: AuthUser | null | undefined;
  userLoading: boolean;
  activeProBillingCycle: BillingCycle | null;
  shouldReduceMotion: boolean | null;
}

export function PlanCard({
  plan,
  index,
  user,
  userLoading,
  activeProBillingCycle,
  shouldReduceMotion,
}: PlanCardProps) {
  const isFree = plan.code === "FREE";
  const isYearly = plan.billingCycle === "YEARLY";
  const isMonthly = plan.billingCycle === "MONTHLY";
  const isCurrentPlan = isFree
    ? user?.plan === "FREE"
    : user?.plan === "PRO" && plan.billingCycle === activeProBillingCycle;

  const userOnMonthly =
    user?.plan === "PRO" && activeProBillingCycle === "MONTHLY";
  const userOnYearly =
    user?.plan === "PRO" && activeProBillingCycle === "YEARLY";

  const isButtonDisabled =
    isCurrentPlan || userLoading || (userOnYearly && isMonthly);

  const buttonText = isCurrentPlan
    ? "Current plan"
    : userOnYearly && isMonthly
      ? "Included in Yearly"
      : userOnMonthly && isYearly
        ? "Upgrade to Pro Yearly"
        : isYearly
          ? "Choose Pro Yearly"
          : "Upgrade to Pro Monthly";

  return (
    <motion.div
      initial={shouldReduceMotion ? false : { opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 + index * 0.09, ease: EASE_OUT_EXPO }}
      whileHover={shouldReduceMotion ? undefined : { y: -4 }}
      className="relative h-full"
    >
      {plan.highlighted && (
        <>
          <motion.div
            aria-hidden="true"
            initial={shouldReduceMotion ? false : { opacity: 0, rotate: 0, x: 0, y: 0 }}
            animate={{ opacity: 1, rotate: 1, x: 8, y: 8 }}
            transition={{ duration: 0.55, delay: 0.25, ease: EASE_OUT_EXPO }}
            className="absolute inset-0 rounded-2xl border border-studio-border bg-studio-panel/60"
          />
          <motion.div
            aria-hidden="true"
            initial={shouldReduceMotion ? false : { opacity: 0, y: -10, rotate: -6 }}
            animate={{ opacity: 1, y: 0, rotate: -2 }}
            transition={{ type: "spring", stiffness: 280, damping: 20, delay: 0.3 }}
            className="absolute -top-3 left-1/2 z-20 h-6 w-24 -translate-x-1/2 bg-primary/80 shadow-sm"
          />
        </>
      )}

      <article
        className={cn(
          "relative flex h-full flex-col overflow-hidden rounded-2xl border p-6 transition-shadow duration-200",
          plan.highlighted
            ? "border-primary/45 bg-studio-panel shadow-2xl"
            : isYearly
              ? "border-dot-max/40 bg-studio-panel/90 shadow-lg"
              : "border-studio-border bg-studio-panel/85 shadow-lg",
        )}
      >
        {plan.highlighted && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute -top-16 -right-16 size-44 rounded-full bg-primary/10 blur-2xl"
          />
        )}

        <div className="relative flex items-start justify-between gap-3">
          <div>
            <span className="font-mono text-xs tracking-widest text-studio-muted uppercase">
              {plan.stepCode}
            </span>
            <h2 className="mt-1 font-heading text-xl font-bold tracking-tight text-studio-foreground">
              {plan.label}
            </h2>
          </div>

          {isCurrentPlan ? (
            <span className="inline-flex items-center gap-1 rounded-full border border-dot-max/40 bg-dot-max/10 px-2.5 py-0.5 font-mono text-xs tracking-wider text-dot-max uppercase">
              <Check className="size-3" />
              Current plan
            </span>
          ) : (
            plan.badge && (
              <span
                className={cn(
                  "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 font-mono text-xs tracking-wider uppercase",
                  isYearly
                    ? "border-dot-max/40 bg-dot-max/10 text-dot-max"
                    : "border-primary/50 bg-primary/15 text-primary",
                )}
              >
                {isYearly ? <Crown className="size-3" /> : <Zap className="size-3" />}
                {plan.badge}
              </span>
            )
          )}
        </div>

        <div className="relative mt-4">
          <div className="flex items-baseline gap-2">
            <span className="font-heading text-4xl font-bold tracking-tight text-studio-foreground">
              {formatBdt(plan.amountBdt)}
            </span>
            <span className="font-mono text-xs tracking-wider text-studio-muted uppercase">
              {plan.period}
            </span>
          </div>
          <p className={cn("mt-1 font-mono text-xs", isYearly ? "text-dot-max" : "text-studio-muted")}>
            {plan.subtext}
          </p>
        </div>

        <p className="mt-3 text-sm leading-5 text-studio-muted">{plan.description}</p>
        <div aria-hidden="true" className="my-4 border-t border-dashed border-studio-border" />

        <ul className="flex-1 space-y-2.5 text-sm">
          {plan.features.map((feature, featureIdx) => (
            <motion.li
              key={feature}
              initial={shouldReduceMotion ? false : { opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.35, delay: 0.25 + index * 0.08 + featureIdx * 0.04, ease: EASE_OUT_EXPO }}
              className="flex items-start gap-2.5"
            >
              <span
                className={cn(
                  "mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-full",
                  plan.highlighted ? "bg-primary/20 text-primary" : "bg-dot-max/15 text-dot-max",
                )}
              >
                <Check className="size-2.5" strokeWidth={3} />
              </span>
              <span className="text-studio-muted">{feature}</span>
            </motion.li>
          ))}
        </ul>

        {isFree ? (
          <Link
            href={user ? "/studio" : "/auth/register"}
            className="mt-6 inline-flex h-11 w-full items-center justify-center gap-2 rounded-full border border-studio-border bg-studio-surface px-5 text-sm font-medium text-studio-foreground transition-colors hover:border-studio-muted/60 hover:bg-accent"
          >
            <span>{user ? "Go to Studio" : "Get Started Free"}</span>
            <ArrowRight className="size-4" />
          </Link>
        ) : (
          <UpgradePlanDialog
            plan={plan}
            trigger={
              <button
                type="button"
                disabled={isButtonDisabled}
                className={cn(
                  "mt-6 inline-flex h-11 w-full cursor-pointer items-center justify-center gap-2 rounded-full px-5 text-sm font-semibold transition-colors",
                  isButtonDisabled
                    ? "cursor-default border border-studio-border bg-studio-surface text-studio-muted"
                    : plan.highlighted
                      ? "bg-primary text-primary-foreground shadow-md hover:bg-primary/90"
                      : "border border-dot-max/50 bg-dot-max/15 text-studio-foreground hover:bg-dot-max/25",
                )}
              >
                {!isButtonDisabled && plan.highlighted && (
                  <span className="size-1.5 rounded-full bg-primary-foreground" />
                )}
                <span>{buttonText}</span>
                {!isButtonDisabled && <ArrowRight className="size-4" />}
              </button>
            }
          />
        )}
      </article>
    </motion.div>
  );
}
