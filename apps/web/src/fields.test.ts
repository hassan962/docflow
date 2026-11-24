import { describe, expect, it } from "vitest";
import type { Field } from "./api";
import { label, naturalOrder, nextPending, reviewOrder, suggestion } from "./fields";

const f = (status: Field["status"], checks: Field["checks"] = []): Field => ({
  value: "x", raw: null, confidence: 0.9, page: 1, bbox: null, checks, source: "llm", status, signals: {},
});

describe("field ordering", () => {
  const fields = {
    "line_items[1].amount": f("needs_review"),
    total: f("auto_approved"),
    vendor_name: f("needs_review"),
    "line_items[0].amount": f("auto_approved"),
    invoice_date: f("auto_approved"),
  };

  it("puts header fields first, then table rows in order", () => {
    expect(naturalOrder(Object.keys(fields))).toEqual([
      "vendor_name", "invoice_date", "total", "line_items[0].amount", "line_items[1].amount",
    ]);
  });

  it("shows fields needing review first", () => {
    expect(reviewOrder(fields).slice(0, 2)).toEqual(["vendor_name", "line_items[1].amount"]);
  });

  it("Tab moves to the next field needing review and wraps around", () => {
    const order = reviewOrder(fields);
    expect(nextPending(order, fields, "vendor_name")).toBe("line_items[1].amount");
    expect(nextPending(order, fields, "line_items[1].amount")).toBe("vendor_name");
    expect(nextPending(order, fields, "vendor_name", -1)).toBe("line_items[1].amount");
    expect(nextPending(["total"], { total: f("reviewed") }, null)).toBeNull();
    expect(nextPending(["gone", "vendor_name"], fields, null)).toBe("vendor_name"); // stale key tolerated
  });
});

describe("labels and suggestions", () => {
  it("formats keys for people", () => {
    expect(label("line_items[2].unit_price")).toBe("Row 3 · unit price");
    expect(label("vendor_tax_id")).toBe("Vendor tax id");
  });

  it("finds a suggestion only on failed checks", () => {
    expect(suggestion(f("needs_review", [{ rule: "tax_id_format", passed: false, message: "bad", suggestion: "27AAPFU0939F1ZV" }]))).toBe("27AAPFU0939F1ZV");
    expect(suggestion(f("auto_approved", [{ rule: "x", passed: true, message: null, suggestion: "y" }]))).toBeNull();
  });
});
