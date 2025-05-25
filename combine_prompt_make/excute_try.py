import sqlite3
import jsonlines
import re
import json

def create_database(data):
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    orders = data.split('\n\n')[0].split('\n')
    for order in orders:
        cursor.execute(order)

    conn.commit()
    return conn

def execute_query(conn, query):
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        results = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"SQLite error: {e}")
        results = None
    except Exception as e:
        print(f"Error: {e}")
        results = None

    return results

data_path = 'prompt_task_1.json'
data = json.load(open(data_path,'r'))
# Assume the JSON file contains a list of SQL commands under the key "sql_commands"
sql_commands = data[0]

# Create the database and execute the SQL commands
conn = create_database(sql_commands)

# Example query to fetch data from the Championship_review table
query = "SELECT * FROM Championship_review"

# Execute the query and print the results
results = execute_query(conn, query)

if results:
    for row in results:
        print(row)

# Close the database connection
conn.close()
