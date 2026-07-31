"""MarketLoop company policy documents exposed as read-only MCP resources."""

from __future__ import annotations

RETURN_AND_REFUND_POLICY = """# MarketLoop Return & Refund Policy

## Return Window
- Most items can be returned within **30 calendar days** of the delivery date.
- Items received damaged, defective, or incorrect are eligible for a return up to **60 calendar days** after delivery.
- Returns requested after the applicable window are not accepted.

## Eligibility
- Items must be unused, in their original packaging, with all tags, manuals, and accessories included.
- Clearance and final-sale items, gift cards, and personalized products are not returnable.

## Restocking Fees
- **Electronics** (laptops, phones, appliances): a **15% restocking fee** applies to opened items. Unopened, sealed items are exempt from the fee.
- **Fashion and apparel**: no restocking fee if tags are still attached; a **10% fee** applies if tags have been removed.
- **Sports equipment and accessories**: no restocking fee.

## Refunds
- Approved refunds are issued to the original payment method.
- Refund processing takes **5-10 business days** after the returned item is received and inspected at the warehouse.
- Shipping costs are refunded only when the return is due to a MarketLoop error or a defective item.

## Return Process
1. Start a return from the order details page or contact Customer Support.
2. Receive a prepaid return label (available within 24 hours of approval).
3. Ship the item back within **14 days** of the label being issued.
4. Refund is issued after warehouse inspection per the timeline above.
"""

SHIPPING_SLA_POLICY = """# MarketLoop Shipping SLA & Guarantee

## Fulfillment
- Orders are picked, packed, and handed to the carrier within **2 business days** of order placement (by 2:00 PM local time on business days).
- Orders placed after the cutoff ship on the next business day.

## Delivery Estimates
- **Standard shipping**: **3-5 business days** after fulfillment (5-7 calendar days total).
- **Express shipping**: **1-2 business days** after fulfillment.
- **Same-day delivery**: available in Cairo and Alexandria for orders placed before 12:00 PM.

## Free Shipping
- Orders over **EGP 1,000** ship free with standard delivery.
- Express and same-day delivery are subject to a flat fee at checkout.

## Delivery Guarantee
- If a standard-shipping order is not delivered within **7 calendar days** of the order date, the shipping fee is **fully refunded**.
- If an express-shipping order is not delivered within **3 calendar days** of the order date, the express fee is refunded and the order qualifies for a **10% discount** on the next order.

## Delivery & Tracking
- A tracking number is emailed within **24 hours** of the order leaving the warehouse.
- If a package is marked delivered but the customer did not receive it, MarketLoop investigates with the carrier within **3 business days** and replaces or refunds the order if the carrier confirms loss.
"""


def return_and_refund_resource() -> str:
    """MarketLoop return window, restocking fee, and refund processing policy."""
    return RETURN_AND_REFUND_POLICY


return_and_refund_resource.uri = "marketloop://policies/return_and_refund"


def shipping_sla_resource() -> str:
    """MarketLoop shipping service-level agreement and delivery guarantees."""
    return SHIPPING_SLA_POLICY


shipping_sla_resource.uri = "marketloop://policies/shipping_sla"
