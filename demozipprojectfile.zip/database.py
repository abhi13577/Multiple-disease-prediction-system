# database.py

import sqlite3
import os

DB_NAME = 'user_reports.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            disease TEXT NOT NULL,
            prediction TEXT NOT NULL,
            input_data TEXT NOT NULL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
    ''')
    
    try:
        # Create a default user if one doesn't exist
        cursor.execute("INSERT INTO users (username) VALUES (?)", ("Current_User",))
    except sqlite3.IntegrityError:
        pass
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database 'user_reports.db' initialized with tables.")