"use client";

import * as React from "react";
import Link from "next/link";
import { Check, ReceiptText, ShieldAlert, X } from "lucide-react";
import { RejectPaymentDialog } from "@/components/admin/reject-payment-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CONTAINER_GUTTER, CONTAINER_MAX_WIDTH } from "@/lib/layout";
import { cn } from "@/lib/utils";
import {
  type AdminPaymentRequest,
  type PaymentStatus,
  useAdminPaymentRequests,
  useApprovePaymentRequest,
  useGetMe,
} from "@/services";

const FILTERS: { label: string; value?: PaymentStatus }[] = [
  { label: "Pending", value: "PENDING" },
  { label: "Approved", value: "APPROVED" },
  { label: "Rejected", value: "REJECTED" },
  { label: "All" },
];

const STATUS_VARIANT: Record<
  PaymentStatus,
  "pending" | "success" | "rejected"
> = {
  PENDING: "pending",
  APPROVED: "success",
  REJECTED: "rejected",
};

const formatBdt = (amount: number) => `৳${amount.toLocaleString("en-US")}`;

const formatDate = (value: string) =>
  new Date(value).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });

export default function AdminPaymentsPage() {
  const { data: user, isLoading: userLoading } = useGetMe();
  const [filter, setFilter] = React.useState<PaymentStatus | undefined>(
    "PENDING",
  );
  const [rejectTarget, setRejectTarget] =
    React.useState<AdminPaymentRequest | null>(null);

  const isAdmin = user?.role === "ADMIN";
  const { data: requests = [], isLoading } = useAdminPaymentRequests(
    filter,
  );
  const { mutate: approve, isPending: isApproving, variables: approvingId } =
    useApprovePaymentRequest();

  if (userLoading) {
    return (
      <div className={cn("mx-auto w-full py-12", CONTAINER_MAX_WIDTH, CONTAINER_GUTTER)}>
        <Skeleton className="h-9 w-64" />
        <Skeleton className="mt-6 h-80 w-full" />
      </div>
    );
  }

  if (!isAdmin) {
    return (
      <div className="flex min-h-120 flex-col items-center justify-center gap-4 px-6 text-center">
        <ShieldAlert className="size-10 text-destructive" />
        <div>
          <h1 className="font-heading text-xl font-semibold">Admins only</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            You need an admin account to review payment requests.
          </p>
        </div>
        <Button render={<Link href={user ? "/studio" : "/auth/login"} />}>
          {user ? "Back to Studio" : "Sign in"}
        </Button>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "mx-auto w-full py-10 sm:py-14",
        CONTAINER_MAX_WIDTH,
        CONTAINER_GUTTER,
      )}
    >
      <div className="flex flex-col gap-2">
        <p className="flex items-center gap-2 font-mono text-xs tracking-widest text-primary uppercase">
          <ReceiptText className="size-3.5" />
          Admin
        </p>
        <h1 className="font-heading text-2xl font-bold tracking-tight sm:text-3xl">
          Payment requests
        </h1>
        <p className="text-sm text-muted-foreground">
          Verify bKash, Nagad, and Rocket payments. Approving a request
          activates that plan for the user immediately.
        </p>
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-2">
        {FILTERS.map((item) => (
          <Button
            key={item.label}
            size="default"
            variant={filter === item.value ? "default" : "outline"}
            onClick={() => setFilter(item.value)}
          >
            {item.label}
          </Button>
        ))}
      </div>

      <div className="mt-4 rounded-xl border border-border/70 bg-card">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>User</TableHead>
              <TableHead>Plan</TableHead>
              <TableHead>Amount</TableHead>
              <TableHead>Method</TableHead>
              <TableHead>Sender / TrxID</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Submitted</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              Array.from({ length: 3 }).map((_, index) => (
                <TableRow key={index}>
                  <TableCell colSpan={8}>
                    <Skeleton className="h-8 w-full" />
                  </TableCell>
                </TableRow>
              ))
            ) : requests.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={8}
                  className="h-24 text-center text-muted-foreground"
                >
                  No payment requests found.
                </TableCell>
              </TableRow>
            ) : (
              requests.map((request) => (
                <TableRow key={request.id}>
                  <TableCell>
                    <div className="flex flex-col">
                      <span className="font-medium">{request.user.name}</span>
                      <span className="text-xs text-muted-foreground">
                        {request.user.email}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-col">
                      <span className="font-medium">{request.plan}</span>
                      <span className="text-xs text-muted-foreground">
                        {request.billingCycle === "YEARLY"
                          ? "Yearly"
                          : "Monthly"}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="tabular-nums">
                    {formatBdt(request.amount)}
                  </TableCell>
                  <TableCell>{request.provider}</TableCell>
                  <TableCell>
                    <div className="flex flex-col text-xs">
                      <span className="tabular-nums">
                        {request.senderNumber}
                      </span>
                      <span className="break-all text-muted-foreground">
                        {request.transactionId}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant={STATUS_VARIANT[request.status]}>
                      {request.status}
                    </Badge>
                    {request.adminNote && (
                      <p className="mt-1 max-w-48 text-xs text-muted-foreground">
                        {request.adminNote}
                      </p>
                    )}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-muted-foreground">
                    {formatDate(request.createdAt)}
                  </TableCell>
                  <TableCell>
                    {request.status === "PENDING" ? (
                      <div className="flex justify-end gap-2">
                        <Button
                          size="default"
                          variant="outline"
                          onClick={() => setRejectTarget(request)}
                        >
                          <X className="size-3.5" />
                          Reject
                        </Button>
                        <Button
                          size="default"
                          loading={
                            isApproving && approvingId === request.id
                          }
                          loadingText="Approving"
                          onClick={() => approve(request.id)}
                        >
                          <Check className="size-3.5" />
                          Approve
                        </Button>
                      </div>
                    ) : (
                      <p className="text-right text-xs text-muted-foreground">
                        Reviewed {request.reviewedAt ? formatDate(request.reviewedAt) : ""}
                      </p>
                    )}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      <RejectPaymentDialog
        key={rejectTarget?.id ?? "closed"}
        request={rejectTarget}
        onClose={() => setRejectTarget(null)}
      />
    </div>
  );
}
