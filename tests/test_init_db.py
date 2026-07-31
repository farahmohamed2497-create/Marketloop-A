import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.init_db import initialize_database


class InitDatabaseTests(unittest.TestCase):
    def test_initialize_database_creates_schema_and_seed_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "marketloop.db"
            result_path = initialize_database(db_path)

            self.assertEqual(result_path, db_path)
            self.assertTrue(db_path.exists())

            with sqlite3.connect(db_path) as connection:
                tables = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                table_names = {row[0] for row in tables}
                self.assertIn("Users", table_names)
                self.assertIn("Products", table_names)
                self.assertIn("Orders", table_names)

                roles = connection.execute("SELECT role_name FROM Roles ORDER BY role_id").fetchall()
                self.assertIn(("Admin",), roles)

                products = connection.execute(
                    "SELECT product_name FROM Products WHERE product_name = 'Dell Laptop'"
                ).fetchall()
                self.assertEqual(products, [("Dell Laptop",)])


if __name__ == "__main__":
    unittest.main()
