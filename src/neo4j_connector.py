from neo4j import GraphDatabase, RoutingControl


class Neo4jConnector:
    def __init__(self, uri="neo4j://127.0.0.1:7687", user="neo4j", password="password", database="neo4j"):
        self.uri = uri
        self.auth = (user, password)
        self.database = database
        self.driver = GraphDatabase.driver(self.uri, auth=self.auth)

    def close(self):
        if self.driver:
            self.driver.close()

    def verify_connection(self):
        self.driver.verify_connectivity()
        return True

    def execute_read(self, query, parameters=None):
        records, _, _ = self.driver.execute_query(
            query,
            parameters_=parameters or {},
            database_=self.database,
            routing_=RoutingControl.READ
        )
        return [record.data() for record in records]

    def execute_write(self, query, parameters=None):
        records, _, _ = self.driver.execute_query(
            query,
            parameters_=parameters or {},
            database_=self.database,
            routing_=RoutingControl.WRITE
        )
        return [record.data() for record in records]
