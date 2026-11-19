"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  type AdminPaymentRequest,
  useRejectPaymentRequest,
} from "@/services";

const formatBdt = (amount: number) => `৳${amount.toLocaleString("en-US")}`;

interface RejectPaymentDialogProps {
  request: AdminPaymentRequest | null;
  onClose: () => void;
}

export function RejectPaymentDialog({
  request,
  onClose,
}: RejectPaymentDialogProps) {
  const [note, setNote] = React.useState("");
  const { mutateAsync: reject, isPending } = useRejectPaymentRequest();

  const handleReject = async () => {
    if (!request) return;
    try {
      await reject({ id: request.id, note: note.trim() });
      setNote("");
      onClose();
    } catch {
      // Toast handled by the mutation hook.
    }
  };

  return (
    <Dialog
      open={request !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Reject payment request</DialogTitle>
          <DialogDescription>
            {request
              ? `Reject ${request.user.name}'s ${request.plan} payment of ${formatBdt(request.amount)}?`
              : null}
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-1.5">
          <Label htmlFor="reject-note">Note to the user (optional)</Label>
          <Textarea
            id="reject-note"
            value={note}
            maxLength={300}
            placeholder="e.g. Transaction ID not found in our statement."
            onChange={(event) => setNote(event.target.value)}
          />
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            loading={isPending}
            loadingText="Rejecting..."
            onClick={handleReject}
          >
            Reject request
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
