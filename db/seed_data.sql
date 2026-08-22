-- =========================
-- Roles 
-- =========================
INSERT INTO Roles (role_name) VALUES
('Admin'),
('Customer Support'),
('Inventory Manager'),
('Marketing Manager'),
('Finance Manager');

-- =========================
-- Users
-- =========================
INSERT INTO Users (full_name,email,password,role_id) VALUES
('Ahmed Hassan','ahmed@ecommerce.com','Adm!n2026#A',1),
('Sara Ali','sara@ecommerce.com','Supp0rt#Sara7',2),
('Omar Khaled','omar@ecommerce.com','Inv_Mgr@2026',3),
('Mona Adel','mona@ecommerce.com','Mkt!Mona88',4),
('Youssef Nabil','youssef@ecommerce.com','Fin#Rep2026',5);

-- =========================
-- Customers
-- =========================
INSERT INTO Customers (name,email,phone,address) VALUES
('Ali Mahmoud','ali@gmail.com','01011111111','Alexandria'),
('Mariam Ahmed','mariam@gmail.com','01022222222','Cairo'),
('Mostafa Samir','mostafa@gmail.com','01033333333','Giza'),
('Nour Hany','nour@gmail.com','01044444444','Mansoura'),
('Salma Tarek','salma@gmail.com','01055555555','Tanta');


-- =========================
-- Subscriptions
-- =========================
INSERT INTO Subscriptions
(customer_id, status, monthly_value, discount_pct)
VALUES
(1, 'active', 500.00, 0.0),
(2, 'active', 750.00, 0.0),
(3, 'active', 300.00, 0.0);

-- =========================
-- Categories
-- =========================
INSERT INTO Categories (category_name) VALUES
('Electronics'),
('Fashion'),
('Home Appliances'),
('Sports');

-- =========================
-- Products
-- =========================
INSERT INTO Products
(product_name,price,description,status,category_id)
VALUES
('Dell Laptop',35000,'Core i7 Laptop','Active',1),
('Samsung Phone',18000,'Android Smartphone','Active',1),
('Running Shoes',2200,'Sports Shoes','Active',4),
('Air Fryer',4500,'Kitchen Appliance','Active',3),
('T-Shirt',500,'Cotton T-Shirt','Active',2);

-- =========================
-- Inventory
-- =========================
INSERT INTO Inventory
(quantity,warehouse_location,product_id)
VALUES
(20,'Warehouse A',1),
(35,'Warehouse A',2),
(50,'Warehouse B',3),
(15,'Warehouse B',4),
(100,'Warehouse C',5);

-- =========================
-- Orders
-- =========================
INSERT INTO Orders
(order_date,total_amount,status,customer_id)
VALUES
('2026-07-01',35000,'Delivered',1),
('2026-07-05',18000,'Shipped',2),
('2026-07-10',2700,'Processing',3);

-- =========================
-- Order_Items
-- =========================
INSERT INTO Order_Items
(quantity,price,order_id,product_id)
VALUES
(1,35000,1,1),
(1,18000,2,2),
(1,2200,3,3),
(1,500,3,5);

-- =========================
-- Shipments
-- =========================
INSERT INTO Shipments
(tracking_number,carrier,shipment_status,order_id)
VALUES
('TRK1001','DHL','Delivered',1),
('TRK1002','Aramex','In Transit',2),
('TRK1003','FedEx','Preparing',3);

-- =========================
-- Return_Requests
-- =========================
INSERT INTO Return_Requests
(reason,request_date,status,order_id,customer_id)
VALUES
('Wrong size','2026-07-15','Pending',1,1);

-- =========================
-- Discounts
-- =========================
INSERT INTO Discounts
(discount_percent,start_date,end_date,status,product_id,created_by)
VALUES
(15,'2026-07-01','2026-08-01','Active',1,4),
(10,'2026-07-10','2026-08-10','Active',3,4);

-- =========================
-- Audit_Log
-- =========================
INSERT INTO Audit_Log
(action,table_name,record_id,details,user_id)
VALUES
('INSERT','Products',1,'New product added',1),
('UPDATE','Inventory',2,'Stock updated',3),
('CREATE_DISCOUNT','Discounts',1,'Summer campaign',4);
