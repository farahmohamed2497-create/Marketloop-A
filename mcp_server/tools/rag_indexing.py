"""
Initialize and populate the MarketLoop knowledge store.

Indexes:
  - Database schema (tables, relationships)
  - MCP tools and their descriptions
  - Business policies (return, warranty, etc.)
  - Workflow documentation
  - FAQs and troubleshooting
"""

from __future__ import annotations

from mcp_server.tools.knowledge_store import KeywordStore


def index_marketloop_knowledge(store: KeywordStore) -> None:
    """
    Populate the knowledge store with MarketLoop domain knowledge.

    This is called once at server startup to prime the RAG index.
    """

    # =========================================================================
    # SECTION: Database Schema
    # =========================================================================

    store.upsert(
        payload=(
            "**Users Table**: Stores user accounts with role-based access. "
            "Columns: user_id (PK), full_name, email (unique), password, role_id (FK). "
            "Each user is assigned a single role (e.g., warehouse admin, customer support, manager). "
            "Passwords should be hashed; plaintext storage is a security risk."
        ),
        metadata={"section": "schema", "subsection": "Users", "entity_type": "table"},
    )

    store.upsert(
        payload=(
            "**Roles Table**: Defines available user roles (warehouse admin, customer support, manager, etc.). "
            "Columns: role_id (PK), role_name (unique). "
            "Used to filter tool visibility in the MCP server. "
            "Role-based access control (RBAC) prevents unauthorized users from calling sensitive tools."
        ),
        metadata={"section": "schema", "subsection": "Roles", "entity_type": "table"},
    )

    store.upsert(
        payload=(
            "**Customers Table**: Customer contact information. "
            "Columns: customer_id (PK), name, email (unique), phone, address. "
            "Linked to Orders via customer_id. "
            "Email and phone used for order notifications and customer support contact."
        ),
        metadata={"section": "schema", "subsection": "Customers", "entity_type": "table"},
    )

    store.upsert(
        payload=(
            "**Products Table**: Product catalog. "
            "Columns: product_id (PK), product_name, price, description, status (Active/Inactive), category_id (FK). "
            "Only active products appear in storefront. "
            "Category foreign key links to Categories table for organization."
        ),
        metadata={"section": "schema", "subsection": "Products", "entity_type": "table"},
    )

    store.upsert(
        payload=(
            "**Inventory Table**: Stock levels for each product. "
            "Columns: inventory_id (PK), quantity, warehouse_location, last_updated, product_id (FK, unique). "
            "One-to-one relationship with Products. "
            "Warehouse managers use this to track stock and reorder when quantity falls below threshold."
        ),
        metadata={
            "section": "schema",
            "subsection": "Inventory",
            "entity_type": "table",
        },
    )

    store.upsert(
        payload=(
            "**Orders Table**: Customer purchases. "
            "Columns: order_id (PK), order_date, total_amount, status (Pending/Processing/Shipped/Delivered/Cancelled), customer_id (FK). "
            "Status transitions follow a workflow: Pending -> Processing -> Shipped -> Delivered. "
            "Cancelled orders may revert inventory."
        ),
        metadata={"section": "schema", "subsection": "Orders", "entity_type": "table"},
    )

    store.upsert(
        payload=(
            "**Order_Items Table**: Line items in an order (many-to-many link). "
            "Columns: order_item_id (PK), quantity, price, order_id (FK), product_id (FK). "
            "Stores the quantity and price at purchase time (historical accuracy). "
            "Multiple items per order; prices may differ from current product prices."
        ),
        metadata={
            "section": "schema",
            "subsection": "Order_Items",
            "entity_type": "table",
        },
    )

    store.upsert(
        payload=(
            "**Shipments Table**: Tracking information for orders. "
            "Columns: shipment_id (PK), tracking_number (unique), carrier, created_at, "
            "shipment_status (Preparing/Shipped/In Transit/Delivered), order_id (FK, unique). "
            "One shipment per order. Carrier examples: FedEx, UPS, DHL."
        ),
        metadata={
            "section": "schema",
            "subsection": "Shipments",
            "entity_type": "table",
        },
    )

    store.upsert(
        payload=(
            "**Return_Requests Table**: Customer return and refund requests. "
            "Columns: return_id (PK), reason, request_date, status (Pending/Approved/Rejected), "
            "order_id (FK), customer_id (FK). "
            "Customer support reviews return reason and decides approval. "
            "Approved returns trigger inventory adjustment and refund processing."
        ),
        metadata={
            "section": "schema",
            "subsection": "Return_Requests",
            "entity_type": "table",
        },
    )

    store.upsert(
        payload=(
            "**Discounts Table**: Promotional pricing. "
            "Columns: discount_id (PK), discount_percent, start_date, end_date, "
            "status (Active/Expired/Scheduled), product_id (FK), created_by (FK to Users). "
            "Managers create discounts; system applies them at checkout based on active date range. "
            "Audit trail: created_by tracks which user created the discount."
        ),
        metadata={
            "section": "schema",
            "subsection": "Discounts",
            "entity_type": "table",
        },
    )

    store.upsert(
        payload=(
            "**Audit_Log Table**: System change tracking. "
            "Columns: log_id (PK), action, table_name, record_id, action_time, details, user_id (FK). "
            "Immutable log of all sensitive operations (refunds, inventory changes, user role changes). "
            "Compliance and troubleshooting tool. Details field stores JSON or free text."
        ),
        metadata={
            "section": "schema",
            "subsection": "Audit_Log",
            "entity_type": "table",
        },
    )

    # =========================================================================
    # SECTION: MCP Tools
    # =========================================================================

    store.upsert(
        payload=(
            "**Tool: list_inventory** - Warehouse managers and admins only. "
            "Returns all products with current stock levels, warehouse locations, and last_updated timestamps. "
            "Use this to track inventory health and identify low-stock items that need reordering. "
            "Output format: table with product_id, product_name, quantity, warehouse_location."
        ),
        metadata={"section": "tools", "subsection": "Inventory", "entity_type": "tool"},
    )

    store.upsert(
        payload=(
            "**Tool: update_inventory_quantity** - Warehouse admin and manager roles only. "
            "Adjust stock level for a product (e.g., after physical count or receiving shipment). "
            "Arguments: product_id, new_quantity, reason (optional). "
            "Triggers audit log entry. Refusing authorization for non-admin roles protects against accidental stock loss. "
            "Returns confirmation and updated inventory state."
        ),
        metadata={"section": "tools", "subsection": "Inventory", "entity_type": "tool"},
    )

    store.upsert(
        payload=(
            "**Tool: list_orders** - All roles. Retrieve orders with optional filtering. "
            "Arguments: status (optional, one of Pending/Processing/Shipped/Delivered/Cancelled), "
            "customer_id (optional). "
            "Returns order_id, order_date, customer name, total_amount, status. "
            "Customer support and warehouse use this daily to track fulfillment."
        ),
        metadata={"section": "tools", "subsection": "Orders", "entity_type": "tool"},
    )

    store.upsert(
        payload=(
            "**Tool: get_order_details** - All roles. Fetch complete order, including line items and shipment. "
            "Arguments: order_id. "
            "Returns order header, list of order_items (product, qty, price), shipment info if exists. "
            "Essential for customer support investigating order issues."
        ),
        metadata={"section": "tools", "subsection": "Orders", "entity_type": "tool"},
    )

    store.upsert(
        payload=(
            "**Tool: create_order** - Sales/order processing only. "
            "Arguments: customer_id, items (list of {product_id, quantity}), shipping_address. "
            "Creates order in Pending status, reserves inventory, and sends confirmation email. "
            "Returns order_id and order_date. Fails if inventory insufficient."
        ),
        metadata={"section": "tools", "subsection": "Orders", "entity_type": "tool"},
    )

    store.upsert(
        payload=(
            "**Tool: process_return_request** - Customer support and manager roles only. "
            "Arguments: return_id, decision (Approved or Rejected), reason. "
            "Approved: refunds customer, returns inventory to stock, sends refund notification. "
            "Rejected: logs decision, notifies customer. "
            "Requires elicitation (human confirmation) before processing."
        ),
        metadata={
            "section": "tools",
            "subsection": "Customer Service",
            "entity_type": "tool",
        },
    )

    store.upsert(
        payload=(
            "**Tool: generate_sales_audit_report** - Manager role only. "
            "Arguments: start_date, end_date. "
            "Generates summary of all orders, revenue, returns, and discounts applied in date range. "
            "Reports progress as it processes. Output: markdown table with totals. "
            "Used for business metrics and compliance audits."
        ),
        metadata={"section": "tools", "subsection": "Reports", "entity_type": "tool"},
    )

    store.upsert(
        payload=(
            "**Tool: switch_active_user_role** - All roles (session tool). "
            "Arguments: user_id. "
            "Changes the session role for multi-role testing or demonstration. "
            "After switch, server sends tools/list_changed notification; client refreshes tool list. "
            "Example: manager switches to warehouse_admin view to check inventory operations."
        ),
        metadata={
            "section": "tools",
            "subsection": "Session",
            "entity_type": "tool",
        },
    )

    # =========================================================================
    # SECTION: Policies & Workflows
    # =========================================================================

    store.upsert(
        payload=(
            "**Return Policy**: Customers may initiate return requests within 30 days of delivery. "
            "Valid reasons: defective, wrong item, changed mind. "
            "Support staff review reason and photos (if provided). "
            "Approved: full refund minus shipping (unless defect). "
            "Rejected: refund denied with explanation. "
            "Processing time: 5-7 business days after approval."
        ),
        metadata={
            "section": "policies",
            "subsection": "Returns",
            "entity_type": "policy",
        },
    )

    store.upsert(
        payload=(
            "**Order Fulfillment Workflow**: "
            "1. Order placed -> status=Pending, inventory reserved. "
            "2. Payment confirmed -> status=Processing, warehouse notified. "
            "3. Items picked and packed -> shipment created. "
            "4. Carrier picks up -> status=Shipped, tracking sent to customer. "
            "5. Delivery -> status=Delivered, completion email sent. "
            "Cancelled orders (before Shipped) refund inventory and customer payment."
        ),
        metadata={
            "section": "policies",
            "subsection": "Order Fulfillment",
            "entity_type": "workflow",
        },
    )

    store.upsert(
        payload=(
            "**Inventory Reorder Rules**: Warehouse managers set a reorder threshold per product (e.g., 20 units). "
            "When quantity drops below threshold, system alerts. "
            "Reorder process: manager submits purchase order (PO), supplier ships, warehouse receives and updates Inventory table. "
            "No automatic reordering; manual approval required."
        ),
        metadata={
            "section": "policies",
            "subsection": "Inventory Management",
            "entity_type": "workflow",
        },
    )

    store.upsert(
        payload=(
            "**Role-Based Access Control (RBAC)**: "
            "- **Customer Support**: Can list/view orders, process returns, generate sales reports. Cannot modify inventory. "
            "- **Warehouse Admin**: Can list/update inventory, cannot process returns or generate reports. "
            "- **Manager**: Can do everything (master role). "
            "- **Front Desk**: Read-only access to customer and order info for inquiries. "
            "Authorization is enforced server-side; tools visible only if user's role matches tool's requirement."
        ),
        metadata={
            "section": "policies",
            "subsection": "Access Control",
            "entity_type": "policy",
        },
    )

    store.upsert(
        payload=(
            "**Audit Logging Requirement**: All sensitive operations must log to Audit_Log: "
            "refunds, inventory changes, role switches, discount creation. "
            "Immutable log used for compliance, dispute resolution, and security review. "
            "Log entry includes: user_id, action, affected table, record_id, timestamp, optional details JSON."
        ),
        metadata={
            "section": "policies",
            "subsection": "Compliance",
            "entity_type": "policy",
        },
    )

    # =========================================================================
    # SECTION: FAQs & Troubleshooting
    # =========================================================================

    store.upsert(
        payload=(
            "**FAQ: How do I check if an item is in stock?** "
            "Use the list_inventory tool (warehouse admin/manager only) to see all stock levels. "
            "Customers can check product status (Active/Inactive) from the storefront. "
            "If a product is Inactive, it's out of stock or discontinued."
        ),
        metadata={
            "section": "workflows",
            "subsection": "FAQ",
            "entity_type": "faq",
        },
    )

    store.upsert(
        payload=(
            "**FAQ: Why did my tool disappear after switching roles?** "
            "Some tools (e.g., update_inventory_quantity, process_return_request) are restricted by role. "
            "When you switch user roles via switch_active_user_role, the server sends a tools/list_changed notification. "
            "Your client must re-fetch the tool list to see the new set of available tools for that role."
        ),
        metadata={
            "section": "workflows",
            "subsection": "FAQ",
            "entity_type": "faq",
        },
    )

    store.upsert(
        payload=(
            "**Troubleshooting: Inventory Update Failed** "
            "Common causes: (1) Product not found (invalid product_id), (2) User role is not warehouse admin/manager, "
            "(3) Database is locked. "
            "Check the tool's response message for the specific error. "
            "If role is correct and product exists, restart the server to release any lock."
        ),
        metadata={
            "section": "workflows",
            "subsection": "Troubleshooting",
            "entity_type": "troubleshooting",
        },
    )

    store.upsert(
        payload=(
            "**Troubleshooting: Return Request Stuck in Pending** "
            "Support staff must explicitly approve or reject the return. "
            "Use process_return_request with decision=Approved or Rejected. "
            "If no action is taken, it remains Pending indefinitely (no auto-expiration). "
            "Check Audit_Log to see if another user has already processed it."
        ),
        metadata={
            "section": "workflows",
            "subsection": "Troubleshooting",
            "entity_type": "troubleshooting",
        },
    )
