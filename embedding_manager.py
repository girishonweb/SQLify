import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.table_descriptions = []
        self.schema_info = None
        self.column_embeddings = {}
        self.table_embeddings = {}

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
            elif col_name in ['salary', 'wage', 'payment', 'amount']:
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
        
        # Create descriptions for each table
        for table_name, table_info in self.schema_info.items():
            semantic_info = self._create_semantic_descriptions(table_name, table_info)
            
            # Create multiple description variants for better matching
            descriptions = [
                # Basic description
                f"Table {table_name} contains columns: {', '.join(semantic_info['column_descriptions'])}",
                # Purpose-focused description
                semantic_info['table_purpose'],
                # Column-focused descriptions
                *[f"Information about {col['semantic_desc']}" for col in semantic_info['semantic_columns']]
            ]
            
            # Add common query patterns
            numeric_columns = [col for col in table_info['columns'] 
                             if any(t in col['type'].lower() for t in ['int', 'decimal', 'numeric', 'float'])]
            text_columns = [col for col in table_info['columns']
                          if any(t in col['type'].lower() for t in ['char', 'text'])]
            date_columns = [col for col in table_info['columns']
                          if any(t in col['type'].lower() for t in ['date', 'timestamp'])]
            
            # Add pattern-based descriptions
            if numeric_columns:
                descriptions.extend([
                    f"Find {table_name} with {col['name']} greater than or less than a value"
                    for col in numeric_columns
                ])
            if text_columns:
                descriptions.extend([
                    f"Search {table_name} by {col['name']}"
                    for col in text_columns
                ])
            if date_columns:
                descriptions.extend([
                    f"Filter {table_name} by {col['name']} date range"
                    for col in date_columns
                ])

            self.table_descriptions.append({
                'table_name': table_name,
                'descriptions': descriptions,
                'semantic_info': semantic_info
            })

    def build_index(self):
        """Build enhanced FAISS index for table and column descriptions."""
        if not self.table_descriptions:
            self.create_table_descriptions()

        # Create embeddings for all descriptions
        all_embeddings = []
        embedding_map = []  # Map to track which embedding belongs to which table

        for table_desc in self.table_descriptions:
            # Encode all descriptions for this table
            table_embeddings = self.model.encode(table_desc['descriptions'])
            
            # Normalize embeddings
            table_embeddings = table_embeddings.astype(np.float32)
            faiss.normalize_L2(table_embeddings)
            
            # Add to collection
            all_embeddings.append(table_embeddings)
            embedding_map.extend([table_desc['table_name']] * len(table_desc['descriptions']))

        # Combine all embeddings
        all_embeddings = np.vstack(all_embeddings)

        # Create and train FAISS index
        self.index = faiss.IndexFlatIP(all_embeddings.shape[1])
        self.index.add(all_embeddings)
        
        # Store the mapping
        self.embedding_map = embedding_map

    def _preprocess_query(self, query):
        """Preprocess the query to enhance matching."""
        # Convert to lowercase
        query = query.lower()
        
        # Extract key components
        amount_pattern = r'(\d+(?:,\d{3})*(?:\.\d{2})?)'
        comparison_pattern = r'(more than|less than|greater than|higher than|lower than|at least|at most|equal to)'
        
        # Find amounts and comparisons
        amounts = re.findall(amount_pattern, query)
        comparisons = re.findall(comparison_pattern, query)
        
        # Create query variants
        variants = [query]
        
        # Add normalized variants
        if amounts and comparisons:
            for amount in amounts:
                for comp in comparisons:
                    variants.append(f"records where value is {comp} {amount}")
                    variants.append(f"entries with values {comp} {amount}")
        
        return variants

    def find_relevant_tables(self, query, top_k=2):
        """Find most relevant tables for a given query using enhanced matching."""
        if not self.index:
            raise Exception("Index not built")

        # Preprocess query
        query_variants = self._preprocess_query(query)
        
        # Get embeddings for all query variants
        query_embeddings = self.model.encode(query_variants)
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)

        # Find matches for each variant
        all_matches = {}
        for q_embed in query_embeddings:
            scores, indices = self.index.search(q_embed.reshape(1, -1), top_k * 2)  # Get more candidates
            
            for score, idx in zip(scores[0], indices[0]):
                if score > 0.3:  # Lower threshold for better recall
                    table_name = self.embedding_map[idx]
                    current_score = all_matches.get(table_name, 0.0)
                    all_matches[table_name] = max(current_score, float(score))

        # Convert to sorted list of results
        relevant_tables = [
            {'table_name': table, 'similarity_score': score}
            for table, score in sorted(all_matches.items(), key=lambda x: x[1], reverse=True)
        ][:top_k]

        return relevant_tables 