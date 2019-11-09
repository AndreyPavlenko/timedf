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

    def __init_predefined_field_values(self, script_name):
        self.__predefined_field_values['ServerName'] = socket.gethostname()
        self.__predefined_field_values['ScriptName'] = script_name
        # Add more stuff here which initializes fields for caches, CPU clock, etc

    def __init__(self, database, table_name, script_name, benchmark_specific_fields):
        self.__table_name = table_name
        self.__init_predefined_field_values(script_name)
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
        for n in self.__predefined_field_values.keys():
            sql_statement += n + ','
        for n in list(benchmark_specific_values.keys())[:-1]:
            sql_statement += n + ','
        sql_statement += list(benchmark_specific_values.keys())[-1] + ") VALUES("
        for n in self.__predefined_field_values.values():
            sql_statement += self.__quote_string(n) + ','
        for n in list(benchmark_specific_values.values())[:-1]:
            sql_statement += self.__quote_string(n) + ','
        n = list(benchmark_specific_values.values())[-1]
        sql_statement += self.__quote_string(n) + ");"
        print("Executing statement", sql_statement)
        self.__database.cursor().execute(sql_statement)
        self.__database.commit()

