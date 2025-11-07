#!/usr/bin/env python3
"""
User Management System for Knowledge Base CLI
Handles user authentication, authorization, and management
"""

import json
import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re

class UserRole(Enum):
    """User roles with different permission levels"""
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    GUEST = "guest"

class Permission(Enum):
    """System permissions"""
    READ_DOCUMENTS = "read_documents"
    WRITE_DOCUMENTS = "write_documents"
    DELETE_DOCUMENTS = "delete_documents"
    MANAGE_USERS = "manage_users"
    SYSTEM_ADMIN = "system_admin"
    VIEW_ANALYTICS = "view_analytics"
    BACKUP_RESTORE = "backup_restore"
    BULK_INGEST = "bulk_ingest"

@dataclass
class User:
    """User data model"""
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    password_hash: str = ""
    role: str = UserRole.VIEWER.value
    created_at: Optional[str] = None
    last_login: Optional[str] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)"""
        data = asdict(self)
        data.pop('password_hash', None)
        return data

class AuthenticationError(Exception):
    """Authentication related errors"""
    pass

class AuthorizationError(Exception):
    """Authorization related errors"""
    pass

class UserManager:
    """Comprehensive user management system"""

    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.session_timeout = timedelta(hours=24)
        self._init_database()
        self._init_default_admin()

    def _init_database(self):
        """Initialize user database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                metadata TEXT
            )
        ''')

        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')

        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                resource TEXT,
                details TEXT,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
            )
        ''')

        conn.commit()
        conn.close()

    def _init_default_admin(self):
        """Create default admin user if no users exist"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]

            if user_count == 0:
                # Create default admin user
                admin_user = User(
                    username="admin",
                    email="admin@localhost",
                    password_hash=self._hash_password("admin123"),
                    role=UserRole.ADMIN.value
                )

                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES (?, ?, ?, ?)
                ''', (admin_user.username, admin_user.email, admin_user.password_hash, admin_user.role))

                conn.commit()

                # Log the action separately to avoid the locking issue
                try:
                    self._log_action(None, "CREATE_DEFAULT_ADMIN", "user", {"username": "admin"})
                except:
                    pass  # Ignore logging errors during initialization

                print("⚠️  Default admin user created:")
                print("   Username: admin")
                print("   Password: admin123")
                print("   Please change the password after first login!")

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print("⚠️  Database is busy, skipping default admin creation")
            else:
                raise e
        finally:
            try:
                conn.close()
            except:
                pass

    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex = password_hash.split(':')
            computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return computed_hash.hex() == hash_hex
        except:
            return False

    def _validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength"""
        errors = []

        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")

        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")

        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")

        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")

        return len(errors) == 0, errors

    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def _validate_username(self, username: str) -> Tuple[bool, List[str]]:
        """Validate username"""
        errors = []

        if len(username) < 3:
            errors.append("Username must be at least 3 characters long")

        if len(username) > 30:
            errors.append("Username must be no more than 30 characters long")

        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            errors.append("Username can only contain letters, numbers, underscores, and hyphens")

        return len(errors) == 0, errors

    def _log_action(self, user_id: Optional[int], action: str, resource: str, details: Dict[str, Any] = None):
        """Log user action to audit trail"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO audit_log (user_id, action, resource, details)
            VALUES (?, ?, ?, ?)
        ''', (user_id, action, resource, json.dumps(details) if details else None))

        conn.commit()
        conn.close()

    def create_user(self, username: str, email: str, password: str, role: str = UserRole.VIEWER.value,
                   created_by: Optional[int] = None) -> Tuple[bool, str, Optional[User]]:
        """Create a new user"""
        # Validate inputs
        valid_username, username_errors = self._validate_username(username)
        if not valid_username:
            return False, f"Invalid username: {', '.join(username_errors)}", None

        if not self._validate_email(email):
            return False, "Invalid email format", None

        valid_password, password_errors = self._validate_password(password)
        if not valid_password:
            return False, f"Invalid password: {', '.join(password_errors)}", None

        try:
            role_enum = UserRole(role.lower())
        except ValueError:
            return False, f"Invalid role. Must be one of: {[r.value for r in UserRole]}", None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if username or email already exists
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            conn.close()
            return False, "Username or email already exists", None

        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=self._hash_password(password),
            role=role_enum.value,
            created_at=datetime.now().isoformat()
        )

        cursor.execute('''
            INSERT INTO users (username, email, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (user.username, user.email, user.password_hash, user.role, user.created_at))

        user.id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Log action
        self._log_action(created_by, "CREATE_USER", "user", {
            "target_username": username,
            "target_email": email,
            "role": role
        })

        return True, "User created successfully", user

    def authenticate(self, username: str, password: str, ip_address: str = None) -> Tuple[bool, str, Optional[str], Optional[User]]:
        """Authenticate user and return session token"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, username, email, password_hash, role, created_at, last_login,
                   is_active, failed_login_attempts, locked_until, metadata
            FROM users WHERE username = ?
        ''', (username,))

        user_data = cursor.fetchone()
        if not user_data:
            conn.close()
            self._log_action(None, "LOGIN_FAILED", "auth", {
                "username": username,
                "reason": "user_not_found",
                "ip_address": ip_address
            })
            return False, "Invalid username or password", None, None

        user = User(
            id=user_data[0],
            username=user_data[1],
            email=user_data[2],
            password_hash=user_data[3],
            role=user_data[4],
            created_at=user_data[5],
            last_login=user_data[6],
            is_active=bool(user_data[7]),
            failed_login_attempts=user_data[8],
            locked_until=user_data[9],
            metadata=json.loads(user_data[10]) if user_data[10] else None
        )

        # Check if user is locked
        if user.locked_until:
            lock_time = datetime.fromisoformat(user.locked_until)
            if datetime.now() < lock_time:
                conn.close()
                return False, f"Account locked. Try again after {lock_time.strftime('%Y-%m-%d %H:%M:%S')}", None, None
            else:
                # Lock expired, reset failed attempts
                cursor.execute("UPDATE users SET failed_login_attempts = 0, locked_until = NULL WHERE id = ?", (user.id,))
                user.failed_login_attempts = 0
                user.locked_until = None

        # Check if user is active
        if not user.is_active:
            conn.close()
            return False, "Account is disabled", None, None

        # Verify password
        if not self._verify_password(password, user.password_hash):
            # Increment failed attempts
            user.failed_login_attempts += 1

            if user.failed_login_attempts >= self.max_failed_attempts:
                # Lock account
                lock_time = datetime.now() + self.lockout_duration
                cursor.execute('''
                    UPDATE users SET failed_login_attempts = ?, locked_until = ?
                    WHERE id = ?
                ''', (user.failed_login_attempts, lock_time.isoformat(), user.id))

                conn.commit()
                conn.close()

                self._log_action(user.id, "ACCOUNT_LOCKED", "auth", {
                    "failed_attempts": user.failed_login_attempts,
                    "locked_until": lock_time.isoformat(),
                    "ip_address": ip_address
                })

                return False, f"Account locked due to too many failed attempts. Try again after {lock_time.strftime('%Y-%m-%d %H:%M:%S')}", None, None
            else:
                cursor.execute("UPDATE users SET failed_login_attempts = ? WHERE id = ?",
                             (user.failed_login_attempts, user.id))
                conn.commit()
                conn.close()

                self._log_action(user.id, "LOGIN_FAILED", "auth", {
                    "failed_attempts": user.failed_login_attempts,
                    "ip_address": ip_address
                })

                return False, "Invalid username or password", None, None

        # Authentication successful - reset failed attempts
        cursor.execute('''
            UPDATE users SET failed_login_attempts = 0, last_login = ? WHERE id = ?
        ''', (datetime.now().isoformat(), user.id))

        # Create session token
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + self.session_timeout

        cursor.execute('''
            INSERT INTO sessions (token, user_id, expires_at, ip_address)
            VALUES (?, ?, ?, ?)
        ''', (session_token, user.id, expires_at.isoformat(), ip_address))

        conn.commit()
        conn.close()

        user.last_login = datetime.now().isoformat()

        # Log successful login
        self._log_action(user.id, "LOGIN_SUCCESS", "auth", {"ip_address": ip_address})

        return True, "Authentication successful", session_token, user

    def verify_session(self, session_token: str, ip_address: str = None) -> Tuple[bool, Optional[User]]:
        """Verify session token and return user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT s.user_id, s.expires_at, s.last_accessed,
                   u.id, u.username, u.email, u.role, u.created_at, u.last_login,
                   u.is_active, u.metadata
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
        ''', (session_token,))

        session_data = cursor.fetchone()
        if not session_data:
            conn.close()
            return False, None

        # Check if session expired
        expires_at = datetime.fromisoformat(session_data[1])
        if datetime.now() > expires_at:
            # Remove expired session
            cursor.execute("DELETE FROM sessions WHERE token = ?", (session_token,))
            conn.commit()
            conn.close()
            return False, None

        # Update last accessed time
        cursor.execute("UPDATE sessions SET last_accessed = ? WHERE token = ?",
                     (datetime.now().isoformat(), session_token))

        conn.commit()

        user = User(
            id=session_data[3],
            username=session_data[4],
            email=session_data[5],
            role=session_data[6],
            created_at=session_data[7],
            last_login=session_data[8],
            is_active=bool(session_data[9]),
            metadata=json.loads(session_data[10]) if session_data[10] else None
        )

        conn.close()

        if not user.is_active:
            return False, None

        return True, user

    def logout(self, session_token: str) -> bool:
        """Logout user by removing session token"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM sessions WHERE token = ?", (session_token,))
        affected_rows = cursor.rowcount

        conn.commit()
        conn.close()

        return affected_rows > 0

    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, username, email, role, created_at, last_login,
                   is_active, failed_login_attempts, locked_until, metadata
            FROM users WHERE id = ?
        ''', (user_id,))

        user_data = cursor.fetchone()
        conn.close()

        if not user_data:
            return None

        return User(
            id=user_data[0],
            username=user_data[1],
            email=user_data[2],
            role=user_data[3],
            created_at=user_data[4],
            last_login=user_data[5],
            is_active=bool(user_data[6]),
            failed_login_attempts=user_data[7],
            locked_until=user_data[8],
            metadata=json.loads(user_data[9]) if user_data[9] else None
        )

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, username, email, role, created_at, last_login,
                   is_active, failed_login_attempts, locked_until, metadata
            FROM users WHERE username = ?
        ''', (username,))

        user_data = cursor.fetchone()
        conn.close()

        if not user_data:
            return None

        return User(
            id=user_data[0],
            username=user_data[1],
            email=user_data[2],
            role=user_data[3],
            created_at=user_data[4],
            last_login=user_data[5],
            is_active=bool(user_data[6]),
            failed_login_attempts=user_data[7],
            locked_until=user_data[8],
            metadata=json.loads(user_data[9]) if user_data[9] else None
        )

    def list_users(self, include_inactive: bool = False) -> List[User]:
        """List all users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
            SELECT id, username, email, role, created_at, last_login,
                   is_active, failed_login_attempts, locked_until, metadata
            FROM users
        '''
        params = []

        if not include_inactive:
            query += " WHERE is_active = 1"

        query += " ORDER BY created_at DESC"

        cursor.execute(query, params)
        users_data = cursor.fetchall()
        conn.close()

        users = []
        for user_data in users_data:
            users.append(User(
                id=user_data[0],
                username=user_data[1],
                email=user_data[2],
                role=user_data[3],
                created_at=user_data[4],
                last_login=user_data[5],
                is_active=bool(user_data[6]),
                failed_login_attempts=user_data[7],
                locked_until=user_data[8],
                metadata=json.loads(user_data[9]) if user_data[9] else None
            ))

        return users

    def update_user(self, user_id: int, **kwargs) -> Tuple[bool, str]:
        """Update user information"""
        user = self.get_user(user_id)
        if not user:
            return False, "User not found"

        allowed_fields = ['email', 'role', 'is_active']
        update_fields = []
        update_values = []

        for field, value in kwargs.items():
            if field in allowed_fields:
                if field == 'email' and not self._validate_email(value):
                    return False, "Invalid email format"

                if field == 'role':
                    try:
                        UserRole(value.lower())
                    except ValueError:
                        return False, f"Invalid role. Must be one of: {[r.value for r in UserRole]}"

                update_fields.append(f"{field} = ?")
                update_values.append(value)

        if not update_fields:
            return False, "No valid fields to update"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        update_values.append(user_id)
        cursor.execute(f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?", update_values)

        conn.commit()
        conn.close()

        self._log_action(user_id, "UPDATE_USER", "user", {"updated_fields": update_fields})

        return True, "User updated successfully"

    def change_password(self, user_id: int, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password"""
        user = self.get_user(user_id)
        if not user:
            return False, "User not found"

        # Verify old password
        if not self._verify_password(old_password, user.password_hash):
            return False, "Current password is incorrect"

        # Validate new password
        valid_password, password_errors = self._validate_password(new_password)
        if not valid_password:
            return False, f"Invalid password: {', '.join(password_errors)}"

        # Update password
        new_password_hash = self._hash_password(new_password)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_password_hash, user_id))
        conn.commit()
        conn.close()

        self._log_action(user_id, "CHANGE_PASSWORD", "user", {})

        return True, "Password changed successfully"

    def delete_user(self, user_id: int, deleted_by: Optional[int] = None) -> Tuple[bool, str]:
        """Delete user (soft delete by deactivating)"""
        user = self.get_user(user_id)
        if not user:
            return False, "User not found"

        # Don't allow deletion of the last admin
        if user.role == UserRole.ADMIN.value:
            admin_count = len([u for u in self.list_users() if u.role == UserRole.ADMIN.value])
            if admin_count <= 1:
                return False, "Cannot delete the last admin user"

        # Soft delete (deactivate)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))

        # Remove all sessions for this user
        cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

        conn.commit()
        conn.close()

        self._log_action(deleted_by, "DELETE_USER", "user", {"target_username": user.username})

        return True, "User deleted successfully"

    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        role_permissions = {
            UserRole.GUEST: [Permission.READ_DOCUMENTS],
            UserRole.VIEWER: [Permission.READ_DOCUMENTS, Permission.VIEW_ANALYTICS],
            UserRole.EDITOR: [
                Permission.READ_DOCUMENTS, Permission.WRITE_DOCUMENTS,
                Permission.VIEW_ANALYTICS, Permission.BULK_INGEST
            ],
            UserRole.ADMIN: [p for p in Permission]  # All permissions
        }

        user_role = UserRole(user.role)
        return permission in role_permissions.get(user_role, [])

    def require_permission(self, user: User, permission: Permission):
        """Require user to have specific permission or raise exception"""
        if not self.has_permission(user, permission):
            raise AuthorizationError(f"Permission required: {permission.value}")

    def get_user_statistics(self) -> Dict[str, Any]:
        """Get user statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total users
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]

        # Active users
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
        active_users = cursor.fetchone()[0]

        # Users by role
        cursor.execute("SELECT role, COUNT(*) FROM users WHERE is_active = 1 GROUP BY role")
        users_by_role = dict(cursor.fetchall())

        # Recent logins (last 7 days)
        cursor.execute('''
            SELECT COUNT(*) FROM users
            WHERE last_login >= datetime('now', '-7 days') AND is_active = 1
        ''')
        recent_logins = cursor.fetchone()[0]

        # Active sessions
        cursor.execute("SELECT COUNT(*) FROM sessions WHERE expires_at > datetime('now')")
        active_sessions = cursor.fetchone()[0]

        conn.close()

        return {
            "total_users": total_users,
            "active_users": active_users,
            "users_by_role": users_by_role,
            "recent_logins": recent_logins,
            "active_sessions": active_sessions
        }

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM sessions WHERE expires_at < datetime('now')")
        deleted_count = cursor.rowcount

        conn.commit()
        conn.close()

        return deleted_count

    def get_audit_log(self, limit: int = 100, user_id: Optional[int] = None,
                     action: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
            SELECT al.id, al.user_id, u.username, al.action, al.resource,
                   al.details, al.ip_address, al.timestamp
            FROM audit_log al
            LEFT JOIN users u ON al.user_id = u.id
            WHERE 1=1
        '''
        params = []

        if user_id:
            query += " AND al.user_id = ?"
            params.append(user_id)

        if action:
            query += " AND al.action = ?"
            params.append(action)

        query += " ORDER BY al.timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        log_entries = cursor.fetchall()
        conn.close()

        return [
            {
                "id": entry[0],
                "user_id": entry[1],
                "username": entry[2],
                "action": entry[3],
                "resource": entry[4],
                "details": json.loads(entry[5]) if entry[5] else None,
                "ip_address": entry[6],
                "timestamp": entry[7]
            }
            for entry in log_entries
        ]

# CLI User Management Commands
def create_user_manager():
    """Create and return a UserManager instance"""
    return UserManager()

def create_user(username: str, email: str, password: str, role: str = "viewer") -> Tuple[bool, str]:
    """CLI helper to create user"""
    user_manager = create_user_manager()

    # Find admin user for created_by field
    admin_user = user_manager.get_user_by_username("admin")
    created_by = admin_user.id if admin_user else None

    success, message, user = user_manager.create_user(username, email, password, role, created_by)
    return success, message

def delete_user(username: str) -> Tuple[bool, str]:
    """CLI helper to delete user"""
    user_manager = create_user_manager()

    user = user_manager.get_user_by_username(username)
    if not user:
        return False, "User not found"

    # Find admin user for deleted_by field
    admin_user = user_manager.get_user_by_username("admin")
    deleted_by = admin_user.id if admin_user else None

    success, message = user_manager.delete_user(user.id, deleted_by)
    return success, message

def list_users() -> List[Dict[str, Any]]:
    """CLI helper to list users"""
    user_manager = create_user_manager()
    users = user_manager.list_users()
    return [user.to_dict() for user in users]

def get_user_stats() -> Dict[str, Any]:
    """CLI helper to get user statistics"""
    user_manager = create_user_manager()
    return user_manager.get_user_statistics()

if __name__ == '__main__':
    # Simple CLI interface for testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python user_manager.py <command> [args]")
        print("Commands: create-user, delete-user, list-users, stats")
        sys.exit(1)

    command = sys.argv[1]

    if command == "create-user" and len(sys.argv) >= 5:
        username, email, password = sys.argv[2:5]
        role = sys.argv[5] if len(sys.argv) > 5 else "viewer"
        success, message = create_user(username, email, password, role)
        print(f"{'✅' if success else '❌'} {message}")

    elif command == "delete-user" and len(sys.argv) >= 3:
        username = sys.argv[2]
        success, message = delete_user(username)
        print(f"{'✅' if success else '❌'} {message}")

    elif command == "list-users":
        users = list_users()
        if users:
            print(f"Found {len(users)} users:")
            for user in users:
                print(f"  • {user['username']} ({user['email']}) - {user['role']}")
        else:
            print("No users found")

    elif command == "stats":
        stats = get_user_stats()
        print("User Statistics:")
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

    else:
        print("Invalid command or insufficient arguments")
        sys.exit(1)