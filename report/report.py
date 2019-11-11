import socket

class DbReport:
    "Initialize and submit reports to MySQL database"

    __predefined_fields = {
        'id': 'BIGINT(20) UNSIGNED NOT NULL AUTO_INCREMENT',
        'ServerName': 'VARCHAR(500) NOT NULL',
        'ScriptName': 'VARCHAR(500) NOT NULL',
        'CommitHash': 'VARCHAR(500) NOT NULL',
        'date': 'TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP',
        # Add more stuff here for caches, CPU clock, etc
    }

    __predefined_field_values = {}

    def __init_predefined_field_values(self, initial_values):
        if initial_values is not None:
            self.__predefined_field_values.update(initial_values)
        self.__predefined_field_values['ServerName'] = socket.gethostname()
        # Add more stuff here which initializes fields for caches, CPU clock, etc

    def __init__(self, database, table_name, benchmark_specific_fields, initial_values=None):
        self.__table_name = table_name
        self.__init_predefined_field_values(initial_values)
        all_fields = self.__predefined_fields
        all_fields.update(benchmark_specific_fields)
        sql_statement = "CREATE TABLE IF NOT EXISTS %s (" % table_name
        for field, spec in all_fields.items():
            sql_statement += field + " " + spec + ","
        sql_statement += "PRIMARY KEY (id));"
        print("Executing statement", sql_statement)
        database.cursor().execute(sql_statement)
        self.__database = database

    def __quote_string(self, n):
        if type(n) is str:
            return "'" + n + "'"
        elif type(n) is float:
            if n == float("inf"):
                return '4294967295'
        return str(n)

    def submit(self, benchmark_specific_values):
        sql_statement = "INSERT INTO %s (" % self.__table_name
        all_fields = self.__predefined_field_values
        all_fields.update(benchmark_specific_values)
        for n in list(all_fields.keys())[:-1]:
            sql_statement += n + ','
        sql_statement += list(all_fields)[-1] + ") VALUES("
        for n in list(all_fields.values())[:-1]:
            sql_statement += self.__quote_string(n) + ','
        n = list(all_fields.values())[-1]
        sql_statement += self.__quote_string(n) + ");"
        print("Executing statement", sql_statement)
        self.__database.cursor().execute(sql_statement)
        self.__database.commit()

