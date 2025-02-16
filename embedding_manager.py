import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class EmbeddingManager:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.table_descriptions = []
        self.schema_info = None
        self.embeddings = None

    def load_schema(self):
        """Load schema information from file."""
        try:
            with open('schema_info.json', 'r') as f:
                self.schema_info = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading schema: {str(e)}")
            return False

    def _create_semantic_descriptions(self, table_name, table_info):
        """Create semantic descriptions for tables and columns."""
        # Table description with purpose
        table_purpose = f"This table named {table_name} stores information about {table_name.lower().replace('_', ' ')}"
        
        # Column descriptions with semantic meaning
        column_descriptions = []
        semantic_columns = []
        for col in table_info['columns']:
            col_name = col['name'].lower()
            col_type = col['type'].lower()
            
            # Create semantic description based on common column patterns
            if 'id' in col_name and col_name.endswith('id'):
                semantic_desc = f"unique identifier for {col_name.replace('_id', '')}"
            elif col_name in ['salary', 'wage', 'payment', 'amount', 'price']:
                semantic_desc = f"monetary value representing {col_name}"
            elif 'date' in col_name or col_type.startswith('date') or col_type.startswith('timestamp'):
                semantic_desc = f"date/time information for {col_name.replace('_date', '')}"
            elif col_name in ['name', 'title', 'label']:
                semantic_desc = f"name or title field"
            else:
                semantic_desc = f"{col_name.replace('_', ' ')} information"

            column_descriptions.append(f"{col['name']} ({col['type']}): {semantic_desc}")
            semantic_columns.append({
                'name': col['name'],
                'type': col['type'],
                'semantic_desc': semantic_desc
            })

        return {
            'table_purpose': table_purpose,
            'column_descriptions': column_descriptions,
            'semantic_columns': semantic_columns
        }

    def create_table_descriptions(self):
        """Create enhanced natural language descriptions of tables and their columns."""
        if not self.schema_info:
            raise Exception("Schema not loaded")

        self.table_descriptions = []
        
        for table_name, table_info in self.schema_info.items():
            semantic_info = self._create_semantic_descriptions(table_name, table_info)
            
            descriptions = [
                f"Table {table_name} contains columns: {', '.join(semantic_info['column_descriptions'])}",
                semantic_info['table_purpose'],
                *[f"Information about {col['semantic_desc']}" for col in semantic_info['semantic_columns']]
            ]
            
            self.table_descriptions.append({
                'table_name': table_name,
                'descriptions': ' '.join(descriptions),
                'semantic_info': semantic_info
            })

    def build_index(self):
        """Build TF-IDF index for table descriptions."""
        if not self.table_descriptions:
            self.create_table_descriptions()

        # Create corpus of descriptions
        corpus = [desc['descriptions'] for desc in self.table_descriptions]
        
        # Create TF-IDF embeddings
        self.embeddings = self.vectorizer.fit_transform(corpus)

    def find_relevant_tables(self, query, top_k=2):
        """Find most relevant tables for a given query using TF-IDF similarity."""
        if self.embeddings is None:
            raise Exception("Index not built")

        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.embeddings).flatten()
        
        # Get top-k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_tables = [
            {
                'table_name': self.table_descriptions[idx]['table_name'],
                'similarity_score': float(similarities[idx])
            }
            for idx in top_indices
            if similarities[idx] > 0.1  # Minimum similarity threshold
        ]

        return relevant_tables 