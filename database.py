import json
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
import time

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.cursor = None
        
    def verify_connection(self):
        """Verify database connection and permissions."""
        try:
            # Basic connection test
            self.cursor.execute("SELECT version();")
            version = self.cursor.fetchone()
            
            # Get user permissions
            self.cursor.execute("""
                SELECT 
                    current_user,
                    session_user,
                    current_database()
            """)
            user_info = self.cursor.fetchone()
            
            # Get all tables user can see
            self.cursor.execute("""
                SELECT schemaname, tablename 
                FROM pg_catalog.pg_tables 
                WHERE tableowner = current_user
                ORDER BY schemaname, tablename;
            """)
            owned_tables = self.cursor.fetchall()
            
            return {
                'version': version['version'],
                'current_user': user_info['current_user'],
                'session_user': user_info['session_user'],
                'database': user_info['current_database'],
                'owned_tables': [f"{t['schemaname']}.{t['tablename']}" for t in owned_tables]
            }
        except Exception as e:
            return None
        
    def connect(self, connection_string):
        """Establish database connection using connection string."""
        try:
            # Handle both URL format and direct parameters
            if '://' in connection_string:
                # Parse connection string
                parsed = urlparse(connection_string)
                dbname = parsed.path[1:]  # Remove leading slash
                user = parsed.username
                password = parsed.password
                host = parsed.hostname
                port = parsed.port or 5432
            else:
                # Parse traditional format: host:port/dbname:username:password
                parts = connection_string.split(':')
                if len(parts) != 3:
                    return False, "Invalid connection string format. Use either URL format or host:port/dbname:username:password"
                host_port_db = parts[0].split('/')
                if len(host_port_db) != 2:
                    return False, "Invalid host/database format"
                host_port = host_port_db[0].split(':')
                host = host_port[0]
                port = int(host_port[1]) if len(host_port) > 1 else 5432
                dbname = host_port_db[1]
                user = parts[1]
                password = parts[2]

            # Validate connection parameters
            if not all([host, dbname, user, password]):
                return False, "Missing required connection parameters"

            # Attempt connection with application_name
            self.conn = psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                application_name='NLtoSQL_Tool',
                options='-c search_path=public,pg_catalog'
            )
            
            # Set autocommit for better table visibility
            self.conn.autocommit = True
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # Small delay to ensure connection is fully established
            time.sleep(1)
            
            # Verify connection and get debug info
            conn_info = self.verify_connection()
            if not conn_info:
                return False, "Connected but unable to verify database access"
            
            if not conn_info['owned_tables']:
                # If no owned tables, try to list all accessible tables
                self.cursor.execute("""
                    SELECT table_schema, table_name 
                    FROM information_schema.tables 
                    WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                    AND table_type = 'BASE TABLE';
                """)
                accessible_tables = self.cursor.fetchall()
                if not accessible_tables:
                    return False, f"Connected as {conn_info['current_user']} to {conn_info['database']} but no accessible tables found. Please verify database permissions."
                
                tables_list = [f"{t['table_schema']}.{t['table_name']}" for t in accessible_tables]
                return True, f"Connected successfully. Found {len(tables_list)} accessible tables."
            
            return True, f"Connected successfully as {conn_info['current_user']}. Found {len(conn_info['owned_tables'])} owned tables."
            
        except Exception as e:
            return False, f"Error connecting to database: {str(e)}"

    def extract_schema(self):
        """Extract database schema information."""
        try:
            if not self.conn:
                return None, "Database not connected"

            schema_info = {}
            
            # First try to get all accessible tables
            self.cursor.execute("""
                SELECT 
                    t.table_schema,
                    t.table_name,
                    (SELECT COUNT(*) FROM information_schema.columns c 
                     WHERE c.table_schema = t.table_schema 
                     AND c.table_name = t.table_name) as column_count
                FROM information_schema.tables t
                WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
                AND t.table_type = 'BASE TABLE'
                ORDER BY t.table_schema, t.table_name;
            """)
            
            tables = self.cursor.fetchall()
            
            if not tables:
                # Try pg_catalog as fallback
                self.cursor.execute("""
                    SELECT schemaname as table_schema, tablename as table_name
                    FROM pg_catalog.pg_tables
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema');
                """)
                tables = self.cursor.fetchall()
            
            if not tables:
                return None, "No tables found. Please verify database permissions."
            
            for table in tables:
                schema_name = table['table_schema']
                table_name = table['table_name']
                full_table_name = f"{schema_name}.{table_name}"
                
                try:
                    # Get column information
                    self.cursor.execute("""
                        SELECT 
                            column_name,
                            data_type,
                            is_nullable,
                            column_default,
                            col_description(
                                (SELECT oid FROM pg_class WHERE relname = %s AND relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = %s)),
                                ordinal_position
                            ) as column_description
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position;
                    """, (table_name, schema_name, schema_name, table_name))
                    
                    columns = self.cursor.fetchall()
                    
                    if columns:
                        schema_info[full_table_name] = {
                            'schema': schema_name,
                            'columns': [
                                {
                                    'name': col['column_name'],
                                    'type': col['data_type'],
                                    'nullable': col['is_nullable'] == 'YES',
                                    'default': col['column_default'],
                                    'description': col['column_description'] or ''
                                }
                                for col in columns
                            ]
                        }
                except Exception as table_error:
                    print(f"Warning: Error getting columns for {full_table_name}: {str(table_error)}")
                    continue

            if not schema_info:
                return None, "Could not extract schema information from any tables."

            # Save schema information to file
            with open('schema_info.json', 'w') as f:
                json.dump(schema_info, f, indent=4)

            return schema_info, f"Successfully extracted schema from {len(schema_info)} tables"
                
        except Exception as e:
            return None, f"Error during schema extraction: {str(e)}"

    def execute_query(self, query):
        """Execute SQL query and return results."""
        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return True, results
        except Exception as e:
            return False, f"Error executing query: {str(e)}"

    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close() 