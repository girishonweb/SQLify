from database import DatabaseManager
from embedding_manager import EmbeddingManager
from query_generator import QueryGenerator
import pandas as pd
from dotenv import load_dotenv
import streamlit as st  # Ensure you have Streamlit imported

class NLToSQL:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.embedding_manager = EmbeddingManager()
        self.query_generator = QueryGenerator()

    def initialize(self, connection_string):
        """Initialize the system by connecting to DB and preparing embeddings."""
        print("Initializing NL to SQL system...")
        
        # Connect to database
        if not self.db_manager.connect(connection_string):
            return False

        # Extract schema
        print("Extracting database schema...")
        self.db_manager.extract_schema()

        # Load schema and build embeddings
        print("Building table embeddings...")
        self.embedding_manager.load_schema()
        self.embedding_manager.create_table_descriptions()
        self.embedding_manager.build_index()

        print("System initialized successfully!")
        return True

    def process_query(self, natural_query):
        """Process a natural language query and return results."""
        try:
            # Find relevant tables
            print("\nFinding relevant tables...")
            relevant_tables = self.embedding_manager.find_relevant_tables(natural_query)
            
            if not relevant_tables:
                return "No relevant tables found for your query."

            print(f"Relevant tables found: {', '.join(t['table_name'] for t in relevant_tables)}")

            # Generate SQL query
            print("\nGenerating SQL query...")
            sql_query = self.query_generator.generate_sql_query(natural_query, relevant_tables)
            print(f"\nGenerated SQL query:\n{sql_query}")

            # Execute query
            print("\nExecuting query...")
            results = self.db_manager.execute_query(sql_query)

            if results is None:
                return "Error executing query."

            # Convert results to pandas DataFrame for better display
            df = pd.DataFrame(results)
            return df

        except Exception as e:
            return f"Error processing query: {str(e)}"

    def close(self):
        """Close database connection."""
        self.db_manager.close()


def main():
    # Initialize the system
    nl_to_sql = NLToSQL()
    
    # Streamlit input for database connection string
    connection_string = st.text_input("Enter your database connection string:", "").strip()

    if connection_string and not nl_to_sql.initialize(connection_string):
        st.error("Failed to initialize the system.")
        return

    st.write("\nNatural Language to SQL Query System")
    st.write("====================================")
    st.write("Type 'exit' to quit the system.")

    while True:
        try:
            # Get user input
            query = st.text_input("Enter your question:").strip()
            
            if query.lower() == 'exit':
                break

            if not query:
                continue

            # Process the query
            results = nl_to_sql.process_query(query)
            
            # Display results
            st.write("\nResults:")
            st.write(results)

        except KeyboardInterrupt:
            break
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Cleanup
    nl_to_sql.close()
    st.write("\nThank you for using the NL to SQL system!")


if __name__ == "__main__":
    main() 