import attr
import pyodbc
from dotenv import dotenv_values


@attr.s
class DataBaseConnector:
    dotenvPrefix: str = attr.ib()

    def __attrs_post_init__(self):
        self.user = dotenv_values().get(str(self.dotenvPrefix) + '_' + 'UID')
        self.driver = dotenv_values().get(str(self.dotenvPrefix) + '_' + 'DRIVER')
        self.password = dotenv_values().get(str(self.dotenvPrefix) + '_' + 'PWD')
        self.host = dotenv_values().get(str(self.dotenvPrefix) + '_' + 'SERVER')
        self.port = dotenv_values().get(str(self.dotenvPrefix) + '_' + 'PORT')
        self.database = dotenv_values().get(str(self.dotenvPrefix) + '_' + 'DB')

    def connect(self):
        pass


@attr.s
class MicrosoftSQLDBConnector(DataBaseConnector):
    def connect(self):

        connstring=f"Driver={self.driver};Server={self.host};Database={self.database};UID={self.user};PWD={self.password}"

        return pyodbc.connect(connstring)

if __name__ == '__main__':
    connector = MicrosoftSQLDBConnector(dotenvPrefix = 'ITDWH')
    print(connector.connect())
