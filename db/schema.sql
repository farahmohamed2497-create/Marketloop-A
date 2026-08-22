-- =========================
-- Roles 
-- =========================
CREATE TABLE Roles (
    role_id INTEGER PRIMARY KEY AUTOINCREMENT,
    role_name TEXT NOT NULL UNIQUE
);

-- =========================
-- Users
-- =========================
CREATE TABLE Users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    role_id INTEGER NOT NULL,
    FOREIGN KEY (role_id) REFERENCES Roles(role_id)
);

-- =========================
-- Customers
-- =========================
CREATE TABLE Customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    address TEXT
);


-- =========================
-- Subscriptions
-- =========================
CREATE TABLE Subscriptions (
    subscription_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    monthly_value DECIMAL(10,2) NOT NULL,
    discount_pct DECIMAL(5,4) NOT NULL DEFAULT 0.0,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);
-- =========================
-- Categories
-- =========================
CREATE TABLE Categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
);

-- =========================
-- Products
-- =========================
CREATE TABLE Products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    description TEXT,
    status TEXT CHECK(status IN ('Active','Inactive')),
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES Categories(category_id)
);

-- =========================
-- Inventory
-- =========================
CREATE TABLE Inventory (
    inventory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    quantity INTEGER NOT NULL,
    warehouse_location TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    product_id INTEGER UNIQUE NOT NULL,
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
);

-- =========================
-- Orders
-- =========================
CREATE TABLE Orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_date DATE NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    status TEXT CHECK(status IN
    ('Pending','Processing','Shipped','Delivered','Cancelled')),
    customer_id INTEGER NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);

-- =========================
-- Order_Items
-- =========================
CREATE TABLE Order_Items (
    order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    FOREIGN KEY (order_id) REFERENCES Orders(order_id),
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
);

-- =========================
-- Shipments
-- =========================
CREATE TABLE Shipments (
    shipment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tracking_number TEXT UNIQUE,
    carrier TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    shipment_status TEXT CHECK(shipment_status IN
    ('Preparing','Shipped','In Transit','Delivered')),
    order_id INTEGER UNIQUE NOT NULL,
    FOREIGN KEY (order_id) REFERENCES Orders(order_id)
);

-- =========================
-- Return_Requests
-- =========================
CREATE TABLE Return_Requests (
    return_id INTEGER PRIMARY KEY AUTOINCREMENT,
    reason TEXT NOT NULL,
    request_date DATE NOT NULL,
    status TEXT CHECK(status IN
    ('Pending','Approved','Rejected')),
    order_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    FOREIGN KEY (order_id) REFERENCES Orders(order_id),
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);

-- =========================
-- Discounts
-- =========================
CREATE TABLE Discounts (
    discount_id INTEGER PRIMARY KEY AUTOINCREMENT,
    discount_percent DECIMAL(5,2) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    status TEXT CHECK(status IN
    ('Active','Expired','Scheduled')),
    product_id INTEGER NOT NULL,
    created_by INTEGER NOT NULL,
    FOREIGN KEY (product_id) REFERENCES Products(product_id),
    FOREIGN KEY (created_by) REFERENCES Users(user_id)
);

-- =========================
-- Audit_Log
-- =========================
CREATE TABLE Audit_Log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    table_name TEXT NOT NULL,
    record_id INTEGER NOT NULL,
    action_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details TEXT,
    user_id INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);
