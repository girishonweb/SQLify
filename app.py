import streamlit as st
from database import DatabaseManager
from embedding_manager import EmbeddingManager
from query_generator import QueryGenerator
import pandas as pd
import json
import os

class NLToSQL:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.embedding_manager = None
        self.query_generator = None
        
    def initialize_nlp_components(self):
        """Initialize NLP components after database connection."""
        try:
            self.embedding_manager = EmbeddingManager()
            self.query_generator = QueryGenerator()
            return True, "NLP components initialized successfully!"
        except Exception as e:
            return False, f"Error initializing NLP components: {str(e)}"
        
    def connect_db(self, connection_string):
        """Connect to database and initialize components."""
        try:
            # Step 1: Connect to database
            success, message = self.db_manager.connect(connection_string)
            if not success:
                return False, message

            # Step 2: Extract schema
            try:
                schema_info, message = self.db_manager.extract_schema()
                if not schema_info:
                    return False, f"Schema extraction failed: {message}"
            except Exception as e:
                return False, f"Error extracting schema: {str(e)}"

            # Step 3: Initialize NLP components
            success, message = self.initialize_nlp_components()
            if not success:
                return False, message

            # Step 4: Load schema and build embeddings
            try:
                if not self.embedding_manager.load_schema():
                    return False, "Failed to load schema for embeddings."
                self.embedding_manager.create_table_descriptions()
                self.embedding_manager.build_index()
            except Exception as e:
                return False, f"Error building embeddings: {str(e)}"

            return True, "Database connected and system initialized successfully!"
        except Exception as e:
            return False, f"Error during initialization: {str(e)}"

    def process_query(self, natural_query):
        """Process a natural language query and return results."""
        try:
            if not self.embedding_manager or not self.query_generator:
                return False, "System not properly initialized. Please reconnect to the database."

            # Find relevant tables
            relevant_tables = self.embedding_manager.find_relevant_tables(natural_query)
            
            if not relevant_tables:
                return False, "No relevant tables found for your query."

            # Generate SQL query
            sql_query = self.query_generator.generate_sql_query(natural_query, relevant_tables)

            # Execute query
            success, results = self.db_manager.execute_query(sql_query)
            
            if not success:
                return False, results

            return True, {
                'relevant_tables': [t['table_name'] for t in relevant_tables],
                'sql_query': sql_query,
                'results': results
            }

        except Exception as e:
            return False, f"Error processing query: {str(e)}"

def main():
    st.set_page_config(
        page_title="Natural Language to SQL Query System",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Natural Language to SQL Query System")
    
    # Initialize session state
    if 'nl_to_sql' not in st.session_state:
        st.session_state.nl_to_sql = NLToSQL()
    if 'connected' not in st.session_state:
        st.session_state.connected = False

    # Database Connection Section
    st.header("1️⃣ Database Connection")
    
    with st.expander("Database Connection", expanded=not st.session_state.connected):
        connection_string = st.text_input(
            "PostgreSQL Connection String",
            placeholder="postgresql://username:password@host:port/database",
            type="password"
        )
        
        if st.button("Connect to Database"):
            if connection_string:
                with st.spinner("Connecting to database and initializing components..."):
                    success, message = st.session_state.nl_to_sql.connect_db(connection_string)
                    if success:
                        st.session_state.connected = True
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.warning("Please enter a connection string")

    # Query Section
    if st.session_state.connected:
        st.header("2️⃣ Natural Language Query")
        
        # Example queries
        st.markdown("### Example Queries")
        examples = [
            "Show all employees earning more than 50K",
            "List all customers from New York",
            "Find all orders placed in the last month"
        ]
        for example in examples:
            st.markdown(f"- {example}")
            
        # Query input
        query = st.text_area(
            "Enter your question",
            placeholder="Type your question here...",
            height=100
        )
        
        if st.button("Generate and Execute Query"):
            if query:
                with st.spinner("Processing your query..."):
                    success, result = st.session_state.nl_to_sql.process_query(query)
                    
                    if success:
                        # Display relevant tables
                        st.markdown("### 📊 Relevant Tables")
                        st.write(", ".join(result['relevant_tables']))
                        
                        # Display generated SQL
                        st.markdown("### 🔍 Generated SQL Query")
                        st.code(result['sql_query'], language='sql')
                        
                        # Display results
                        st.markdown("### 📈 Query Results")
                        df = pd.DataFrame(result['results'])
                        st.dataframe(df)
                    else:
                        st.error(result)
            else:
                st.warning("Please enter a query")

if __name__ == "__main__":
    main() 