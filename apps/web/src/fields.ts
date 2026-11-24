// Pure helpers for the review screen, kept apart from React so they are easy to test.

import type { Field } from "./api";

const HEADER_ORDER = [
  "vendor_name",
  "vendor_tax_id",
  "invoice_number",
  "invoice_date",
  "po_number",
  "currency",
  "subtotal",
  "tax",
  "total",
];

function rowIndex(key: string): number {
  const m = key.match(/\[(\d+)\]/);
  return m ? Number(m[1]) : -1;
}

/** Natural document order: header fields, then table rows in order. */
export function naturalOrder(keys: string[]): string[] {
  const rank = (k: string) => {
    const i = HEADER_ORDER.indexOf(k);
    return i >= 0 ? i : 100 + rowIndex(k);
  };
  return [...keys].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

/** Fields needing review first (in document order), then everything else. */
export function reviewOrder(fields: Record<string, Field>): string[] {
  const keys = naturalOrder(Object.keys(fields));
  const pending = keys.filter((k) => fields[k].status === "needs_review");
  return [...pending, ...keys.filter((k) => fields[k].status !== "needs_review")];
}

/** The next field that still needs review after ``current`` (wrapping), or null. */
export function nextPending(order: string[], fields: Record<string, Field>, current: string | null, step = 1): string | null {
  const pending = order.filter((k) => fields[k]?.status === "needs_review");
  if (pending.length === 0) return null;
  const at = current ? order.indexOf(current) : -1;
  for (let i = 1; i <= order.length; i++) {
    const key = order[(((at + step * i) % order.length) + order.length) % order.length];
    if (fields[key]?.status === "needs_review") return key;
  }
  return null;
}

/** "line_items[2].unit_price" -> "Row 3 · unit price". */
export function label(key: string): string {
  const row = rowIndex(key);
  const name = key.split(".").pop()!.replace(/_/g, " ");
  return row >= 0 ? `Row ${row + 1} · ${name}` : name.replace(/^./, (c) => c.toUpperCase());
}

export function display(value: unknown): string {
  return value === null || value === undefined ? "—" : String(value);
}

/** The first suggestion any failed rule offers for this field. */
export function suggestion(field: Field): unknown {
  return field.checks.find((c) => !c.passed && c.suggestion != null)?.suggestion ?? null;
}

export function failedChecks(field: Field) {
  return field.checks.filter((c) => !c.passed);
}
