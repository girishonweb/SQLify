# Natural Language to SQL Query System

This system allows users to query a PostgreSQL database using natural language questions. It automatically identifies relevant tables and generates appropriate SQL queries.

## Features

- 🔌 PostgreSQL database connection
- 🤖 Natural language query processing
- 🎯 Automatic table selection using RAG (Retrieval-Augmented Generation)
- 💡 SQL query generation using CodeLlama
- 📊 Clean results display using pandas

## Prerequisites

- Python 3.8+
- PostgreSQL database
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file:
Copy the `.env.example` file to `.env` and fill in your database credentials:
```bash
cp .env.example .env
```

Edit the `.env` file with your PostgreSQL database credentials:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Enter your questions in natural language. For example:
- "Show all employees earning more than 50K"
- "List all customers from New York"
- "Find all orders placed in the last month"

3. Type 'exit' to quit the application.

## How It Works

1. **Database Connection**: The system connects to your PostgreSQL database using the credentials provided in the `.env` file.

2. **Schema Extraction**: Upon startup, the system extracts and stores information about all tables and columns in your database.

3. **Query Processing**:
   - Your natural language query is processed to find relevant tables using FAISS similarity search
   - The query and schema information are used to generate an appropriate SQL query
   - The SQL query is executed, and results are displayed in a formatted table

## Error Handling

- The system provides clear error messages for:
  - Database connection issues
  - Invalid queries
  - Schema extraction problems
  - Query execution errors

## Limitations

- The system works best with clear, well-structured questions
- Complex queries involving multiple joins might require more specific phrasing
- Performance depends on the size of your database and the complexity of the query

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 