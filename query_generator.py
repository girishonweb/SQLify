import json
import re
import anthropic
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

class QueryGenerator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Anthropic client with API key from environment
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.client = anthropic.Client(api_key=api_key)
        
    def _extract_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract intent from natural language query using Claude."""
        system_prompt = """You are an expert at understanding database queries.
        Analyze the user's question and extract:
        1. What specific information they want (columns)
        2. Any conditions or filters
        3. The main subject they're asking about
        
        Format your response as JSON with these keys:
        {
            "target_columns": [], # List of columns they want to see
            "conditions": [], # List of conditions to filter by
            "subject": "", # Main entity being queried
            "output_columns": [] # Final columns to show in result
        }"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Analyze this database query: {query}"
                    }
                ]
            )

            # Parse the response into intent dictionary
            try:
                # Get the response text and try to parse it as JSON
                response_text = response.content[0].text
                # Remove any markdown formatting if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1]
                
                intent = json.loads(response_text.strip())
                return intent
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                print(f"Error parsing Claude response: {e}")
                print(f"Raw response: {response_text}")
                return {
                    "target_columns": [],
                    "conditions": [],
                    "subject": None,
                    "output_columns": ["name", "price", "category"]
                }
                
        except Exception as e:
            print(f"Error in query intent extraction: {e}")
            return {
                "target_columns": [],
                "conditions": [],
                "subject": None,
                "output_columns": ["name", "price", "category"]  # Default columns
            }

    def _generate_sql_with_claude(self, query: str, table_info: Dict, intent: Dict) -> str:
        """Generate SQL query using Claude with schema context."""
        try:
            # Create schema description
            schema_desc = []
            for table_name, info in table_info.items():
                cols = [f"{col['name']} ({col['type']})" for col in info['columns']]
                schema_desc.append(f"Table {table_name} columns: {', '.join(cols)}")
            
            schema_context = "\n".join(schema_desc)

            system_prompt = """You are an expert PostgreSQL query generator.
            Given a database schema and a natural language query:
            1. Generate a precise SQL query that gets exactly what's asked
            2. Use specific column names instead of SELECT *
            3. Include proper WHERE clauses for filtering
            4. Use ILIKE for text matching to handle case-insensitivity
            5. Return only the SQL query, nothing else
            6. Do not include any explanations or markdown, just the SQL query"""

            message_content = f"""Database Schema:
            {schema_context}

            Natural Language Query: {query}

            Intent Analysis:
            - Looking for: {', '.join(intent['output_columns'])}
            - Subject: {intent['subject']}
            - Conditions: {intent['conditions']}

            Generate a precise SQL query that gets exactly what's asked."""

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": message_content
                    }
                ]
            )

            # Extract SQL query from response
            sql_query = response.content[0].text.strip()
            
            # Clean up the query
            sql_query = self._clean_sql_query(sql_query)
            
            return sql_query
        except Exception as e:
            print(f"Error in SQL generation: {e}")
            raise

    def _clean_sql_query(self, sql: str) -> str:
        """Clean and validate the generated SQL query."""
        try:
            # Remove any comments
            sql = re.sub(r'--.*$|/\*.*?\*/', '', sql, flags=re.MULTILINE)
            
            # Remove any markdown code block syntax
            if "```sql" in sql:
                sql = sql.split("```sql")[1].split("```")[0]
            elif "```" in sql:
                sql = sql.split("```")[1].split("```")[0]
            
            # Extract only the SQL query
            sql_pattern = r'SELECT.*?;'
            matches = re.findall(sql_pattern, sql, re.IGNORECASE | re.DOTALL)
            
            if matches:
                sql = matches[0]
            else:
                # If no complete query found, try to clean up what we have
                sql = sql.strip()
                if not sql.lower().startswith('select'):
                    raise ValueError("Invalid SQL query: must start with SELECT")
                if not sql.endswith(';'):
                    sql += ';'
            
            # Clean up whitespace
            sql = ' '.join(sql.split())
            
            return sql
        except Exception as e:
            print(f"Error cleaning SQL query: {e}")
            print(f"Original SQL: {sql}")
            raise

    def generate_sql_query(self, natural_query: str, relevant_tables: List[Dict]) -> str:
        """Generate SQL query from natural language and relevant table information."""
        try:
            # Load schema information
            with open('schema_info.json', 'r') as f:
                full_schema = json.load(f)
            
            # Filter schema to only relevant tables
            relevant_schema = {
                table['table_name']: full_schema[table['table_name']]
                for table in relevant_tables
            }
            
            # Extract query intent using Claude
            intent = self._extract_query_intent(natural_query)
            
            # Generate SQL query using Claude
            sql_query = self._generate_sql_with_claude(natural_query, relevant_schema, intent)
            
            # Validate basic SQL structure
            if not sql_query.lower().startswith('select'):
                raise ValueError("Generated query does not start with SELECT")
            
            return sql_query
            
        except Exception as e:
            raise Exception(f"Error generating SQL query: {str(e)}") 