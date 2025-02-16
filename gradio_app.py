import gradio as gr
import pandas as pd
from database import DatabaseManager
from embedding_manager import EmbeddingManager
from query_generator import QueryGenerator
import os
import socket
from dotenv import load_dotenv

def find_free_port(start_port=7860, max_port=7960):
    """Find a free port in the given range."""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Cannot find empty port in range: {start_port}-{max_port}")

class NLToSQL:
    def __init__(self):
        self.db_manager = None
        self.embedding_manager = None
        self.query_generator = None
        self.is_connected = False
        
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
            # Initialize database manager
            self.db_manager = DatabaseManager()
            
            # Connect to database
            success, message = self.db_manager.connect(connection_string)
            if not success:
                return f"❌ Connection error: {message}"

            # Extract schema
            schema_info, message = self.db_manager.extract_schema()
            if not schema_info:
                return f"❌ Schema extraction failed: {message}"

            # Initialize NLP components
            success, message = self.initialize_nlp_components()
            if not success:
                return f"❌ NLP initialization failed: {message}"

            # Load schema and build embeddings
            if not self.embedding_manager.load_schema():
                return "❌ Failed to load schema for embeddings."
            self.embedding_manager.create_table_descriptions()
            self.embedding_manager.build_index()

            self.is_connected = True
            return "✅ Connected! You can now enter your queries."

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def process_query(self, query):
        """Process a natural language query and return results."""
        try:
            if not self.is_connected:
                return "❌ Please connect to the database first."

            # Find relevant tables
            relevant_tables = self.embedding_manager.find_relevant_tables(query)
            if not relevant_tables:
                return "❌ No relevant tables found for your query."

            # Generate SQL query
            sql_query = self.query_generator.generate_sql_query(query, relevant_tables)

            # Execute query
            success, results = self.db_manager.execute_query(sql_query)
            if not success:
                return f"❌ Error executing query: {results}"

            # Format results
            df = pd.DataFrame(results)
            tables_used = ", ".join([t['table_name'] for t in relevant_tables])
            
            return {
                'tables': f"Tables used: {tables_used}",
                'query': sql_query,
                'results': df
            }

        except Exception as e:
            return f"❌ Error: {str(e)}"

def main():
    try:
        load_dotenv()
        nl_to_sql = NLToSQL()

        def connect_to_db(connection_string):
            if not connection_string:
                return "❌ Please provide a connection string.", gr.update(visible=False)
            result = nl_to_sql.connect_db(connection_string)
            if "✅" in result:
                return result, gr.update(visible=True)
            return result, gr.update(visible=False)

        def process_query(query):
            if not query:
                return "Please enter a query", None, None
            
            result = nl_to_sql.process_query(query)
            if isinstance(result, str):
                return result, None, None
            
            return result['tables'], result['query'], result['results']

        # Create two-step interface
        with gr.Blocks(title="NL to SQL") as iface:
            gr.Markdown("## Natural Language to SQL Query")
            
            with gr.Column() as connection_section:
                gr.Markdown("### Step 1: Connect to Database")
                connection_input = gr.Textbox(
                    label="PostgreSQL Connection String",
                    placeholder="postgresql://username:password@host:port/database",
                    type="password"
                )
                connect_btn = gr.Button("Connect to Database", variant="primary")
                connection_status = gr.Markdown()

            with gr.Column(visible=False) as query_section:
                gr.Markdown("### Step 2: Ask Your Question")
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Enter your question here...",
                    lines=3
                )
                query_btn = gr.Button("Generate SQL & Get Results", variant="primary")
                
                with gr.Column() as results_section:
                    tables_output = gr.Markdown(label="Tables Used")
                    query_output = gr.Code(language="sql", label="Generated SQL")
                    results_output = gr.DataFrame(label="Results")

            # Set up event handlers
            connect_btn.click(
                fn=connect_to_db,
                inputs=[connection_input],
                outputs=[connection_status, query_section]
            )

            query_btn.click(
                fn=process_query,
                inputs=[query_input],
                outputs=[tables_output, query_output, results_output]
            )

        # Launch the interface
        iface.launch(
            share=True
        )
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        print("Please check your network connection and try again.")

if __name__ == "__main__":
    main() 