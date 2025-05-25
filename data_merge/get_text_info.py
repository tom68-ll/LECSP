import sqlite3
import json

class get_text:
    def __init__(self,db_path,question,conn=None):
        self.db_path = db_path
        self.question = question
        self.conn = conn

    def get_db_schema(self):
        """
        Extracts the database schema (tables and columns) from a SQLite database.

        Returns:
        dict: A dictionary with table names as keys and lists of column names as values.
        """
        schema = {}
        if not self.conn:
            # Connect to the SQLite database
            conn = sqlite3.connect(self.db_path)
        else:
            conn = self.conn
        cursor = conn.cursor()
        
        # Query for all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Loop through tables and get column details
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            schema[table_name] = ', '.join(column[1] for column in columns)
        
        cursor.close()
        if not self.conn:
            conn.close()
        return schema

    def generate_output(self):
        """
        Generates formatted text combining database schema and a specific question.

        Parameters:
        schema (dict): Database schema as a dictionary.
        question (str): A specific question related to the database.

        Returns:
        str: Formatted text combining the schema and the question.
        """
        # Format the schema into text
        schema_text = "; ".join([f"{table}: {columns}" for table, columns in self.schema.items()])
        
        # Combine the schema text and the question to form the output
        output_text = f"{schema_text} | {self.question}"
        return output_text

    def get_text(self):
        self.schema = self.get_db_schema()
        output = self.generate_output()
        return output
    

class get_text_forwiki:
    def __init__(self,db_id,question):
        self.db_id = db_id
        self.question = question

    def get_db_schema(self):
        wikisql_table_path = '../data/wikisql/tables.json'
        with open(wikisql_table_path,'r') as fp:
            tables = json.load(fp)

        table_now = None
        for table in tables:
            if table["db_id"] == self.db_id:
                table_now = table
                break
        
        column_list = []
        for tmp_list in table_now["column_names"][1:]:
            column_list.append(tmp_list[1])
        return column_list

    def generate_output(self):
        """
        Generates formatted text combining database schema and a specific question.

        Parameters:
        schema (dict): Database schema as a dictionary.
        question (str): A specific question related to the database.

        Returns:
        str: Formatted text combining the schema and the question.
        """
        # Format the schema into text
        schema_text = ", ".join(self.columnlist)
        
        # Combine the schema text and the question to form the output
        output_text = f"{schema_text} | {self.question}"
        return output_text

    def get_text(self):
        self.columnlist = self.get_db_schema()
        output = self.generate_output()
        return output

