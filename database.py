import sqlite3
import hashlib

DB_NAME = 'gait_analysis.db'


def create_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn


def initialize_db():
    conn = create_connection()
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')

    # Create gait data table (optional for later analysis saving)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gait_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            video_file TEXT,
            parameters TEXT,
            diagnosis TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()


def add_user(username, password, role="patient"):
    conn = create_connection()
    cursor = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    try:
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, hashed_password, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists
    finally:
        conn.close()


def validate_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    result = cursor.fetchone()

    conn.close()

    if result:
        return result  # Return user info
    else:
        return None
