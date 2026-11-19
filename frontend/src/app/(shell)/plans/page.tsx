"use client";

import * as React from "react";
import { motion, useReducedMotion } from "framer-motion";
import {
  Clock3,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { PlanCard } from "@/components/subscription/plan-card";
import { UpgradePlanDialog } from "@/components/subscription/upgrade-plan-dialog";
import { Button } from "@/components/ui/button";
import { CONTAINER_GUTTER, CONTAINER_MAX_WIDTH } from "@/lib/layout";
import { cn } from "@/lib/utils";
import {
  useGetMe,
  useMyPaymentRequests,
  usePlans,
  type BillingCycle,
} from "@/services";

const formatBdt = (amount: number) => `৳${amount.toLocaleString("en-US")}`;

const BOTTOM_NOTES = [
  {
    num: "01",
    title: "Preview for free",
    desc: "Tune colors & curves — previews never spend credits",
    accent: "text-trace",
  },
  {
    num: "02",
    title: "1 credit per export",
    desc: "Starter credits included on sign-up, no card needed",
    accent: "text-primary",
  },
  {
    num: "03",
    title: "bKash · Nagad · Rocket",
    desc: "Fast local payments with instant review in Studio",
    accent: "text-dot-max",
  },
];

const EASE_OUT_EXPO: [number, number, number, number] = [0.16, 1, 0.3, 1];

export default function PlansPage() {
  const shouldReduceMotion = useReducedMotion();
  const { data: user, isLoading: userLoading } = useGetMe();
  const { data: plans, isLoading, isError, refetch } = usePlans();
  const { data: myRequests = [] } = useMyPaymentRequests({
    enabled: Boolean(user),
  });

  const pendingRequest = user
    ? myRequests.find((request) => request.status === "PENDING")
    : undefined;

  const approvedProRequest = user
    ? myRequests.find(
        (request) => request.status === "APPROVED" && request.plan === "PRO",
      )
    : undefined;

  const activeProBillingCycle: BillingCycle | null =
    user?.plan === "PRO"
      ? (approvedProRequest?.billingCycle ?? "MONTHLY")
      : null;

  const requestedPendingPlan = pendingRequest
    ? (plans?.find(
        (p) =>
          p.code === pendingRequest.plan &&
          p.billingCycle === pendingRequest.billingCycle,
      ) ?? plans?.find((p) => p.code === pendingRequest.plan) ?? null)
    : null;

  return (
    <section className="relative flex min-h-dvh flex-col justify-between overflow-hidden bg-studio-surface pt-24 pb-8">
      {/* Dot-grid notebook background */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 vk-dot-grid opacity-40"
      />

      <div
        className={cn(
          "relative mx-auto my-auto w-full",
          CONTAINER_MAX_WIDTH,
          CONTAINER_GUTTER,
        )}
      >
        {/* Compact Sketch Header */}
        <motion.div
          initial={shouldReduceMotion ? false : { opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: EASE_OUT_EXPO }}
          className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between"
        >
          <div>
            <div className="flex flex-wrap items-center gap-3">
              <p className="font-mono text-xs tracking-wider text-studio-muted">
                <span className="text-studio-muted/60">{"//"}</span> Plans &amp;
                credits · Vectorizer API
              </p>
              <motion.span
                whileHover={shouldReduceMotion ? undefined : { scale: 1.05 }}
                className="inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-0.5 font-mono text-xs font-medium text-primary"
              >
                <motion.span
                  animate={
                    shouldReduceMotion
                      ? undefined
                      : { rotate: [0, 15, -10, 0] }
                  }
                  transition={{
                    duration: 3.2,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                  className="inline-flex"
                >
                  <Sparkles className="size-3" />
                </motion.span>
                TRANSPARENT CREDITS
              </motion.span>
            </div>
            <h1 className="mt-2 font-heading text-3xl font-bold tracking-tight text-studio-foreground sm:text-4xl lg:text-5xl">
              Simple pricing,{" "}
              <motion.span
                initial={
                  shouldReduceMotion
                    ? false
                    : { opacity: 0, scale: 0.85, rotate: -6 }
                }
                animate={{ opacity: 1, scale: 1, rotate: -1 }}
                transition={{
                  type: "spring",
                  stiffness: 300,
                  damping: 18,
                  delay: 0.15,
                }}
                whileHover={
                  shouldReduceMotion ? undefined : { rotate: 0, scale: 1.03 }
                }
                className="inline-block -rotate-1 bg-primary px-2.5 py-0.5 text-primary-foreground shadow-md"
              >
                zero guesswork
              </motion.span>
              .
            </h1>
          </div>

          <div className="flex flex-col items-start gap-1 sm:items-end">
            <motion.span
              animate={
                shouldReduceMotion
                  ? undefined
                  : { y: [0, -2.5, 0], rotate: [-2, -0.5, -2] }
              }
              transition={{
                duration: 3.6,
                repeat: Infinity,
                ease: "easeInOut",
              }}
              className="-rotate-2 font-hand text-xl text-trace"
            >
              previews are always free ↗
            </motion.span>
            <p className="flex items-center gap-1.5 font-mono text-xs text-studio-muted">
              <ShieldCheck className="size-3.5 text-dot-max" />
              Credits are only spent when you export.
            </p>
          </div>
        </motion.div>

        {/* Pending Payment Banner */}
        {pendingRequest && (
          <motion.div
            initial={shouldReduceMotion ? false : { opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: EASE_OUT_EXPO }}
            className="mt-5 flex flex-col gap-3 rounded-xl border border-dashed border-primary/40 bg-primary/10 px-4 py-2.5 text-xs sm:flex-row sm:items-center sm:justify-between"
          >
            <p className="flex items-center gap-2 text-studio-foreground">
              <Clock3 className="size-4 shrink-0 text-primary" />
              <span>
                Your <strong>{pendingRequest.plan}</strong> payment (
                {formatBdt(pendingRequest.amount)}) is under review.
              </span>
            </p>
            {requestedPendingPlan && (
              <UpgradePlanDialog
                plan={requestedPendingPlan}
                trigger={
                  <Button variant="outline" size="default">
                    View status
                  </Button>
                }
              />
            )}
          </motion.div>
        )}

        {/* 3-Column Plan Cards */}
        {isLoading ? (
          <div className="mt-7 grid items-stretch gap-6 lg:grid-cols-3">
            {["FREE", "PRO-MONTHLY", "PRO-YEARLY"].map((code) => (
              <div
                key={code}
                className="h-96 animate-pulse rounded-2xl border border-studio-border bg-studio-panel/60"
              />
            ))}
          </div>
        ) : isError || !plans ? (
          <div className="mx-auto mt-8 flex max-w-xl flex-col items-center gap-3 rounded-2xl border border-dashed border-studio-border bg-studio-panel p-10 text-center">
            <p className="text-sm text-studio-muted">
              Pricing could not be loaded right now.
            </p>
            <Button variant="outline" size="default" onClick={() => refetch()}>
              <RefreshCw className="size-4" />
              Try again
            </Button>
          </div>
        ) : (
          <div className="mt-7 grid items-stretch gap-6 lg:grid-cols-3 lg:gap-7">
            {plans.map((plan, index) => (
              <PlanCard
                key={plan.id}
                plan={plan}
                index={index}
                user={user}
                userLoading={userLoading}
                activeProBillingCycle={activeProBillingCycle}
                shouldReduceMotion={shouldReduceMotion}
              />
            ))}
          </div>
        )}
      </div>

      {/* Bottom 3-step drafting strip */}
      <motion.div
        initial={shouldReduceMotion ? false : { opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4, ease: EASE_OUT_EXPO }}
        className={cn(
          "relative mx-auto mt-8 hidden w-full lg:block",
          CONTAINER_MAX_WIDTH,
          CONTAINER_GUTTER,
        )}
      >
        <div className="grid grid-cols-3 gap-6 border-t border-dashed border-studio-border/80 pt-4">
          {BOTTOM_NOTES.map((item) => (
            <motion.div
              key={item.num}
              whileHover={shouldReduceMotion ? undefined : { y: -2 }}
              className="flex cursor-default items-baseline gap-2.5"
            >
              <span className={cn("font-mono text-xs font-bold", item.accent)}>
                {item.num}
              </span>
              <div className="min-w-0">
                <p className="font-heading text-sm font-semibold text-studio-foreground">
                  {item.title}
                </p>
                <p className="truncate font-mono text-xs text-studio-muted">
                  {item.desc}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </section>
  );
}
