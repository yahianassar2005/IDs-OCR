import sqlite3
import os

class IDDatabase:
    def __init__(self, db_path: str = "national_ids.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS extracted_ids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_name TEXT,
                    full_name TEXT,
                    address_1 TEXT,
                    address_2 TEXT,
                    national_id TEXT,
                    source_image TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def insert_record(self, data: dict, source_image: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO extracted_ids (first_name, full_name, address_1, address_2, national_id, source_image)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data.get("first_name"),
                data.get("full_name"),
                data.get("address_1"),
                data.get("address_2"),
                data.get("national_id"),
                os.path.basename(source_image)
            ))
            conn.commit()
