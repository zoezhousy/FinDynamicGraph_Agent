"""Environment-only connection settings and exception-safe store initialization."""
from dataclasses import dataclass, field
import os

@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str = field(repr=False)
    database: str = "neo4j"

    @classmethod
    def from_env(cls, database=None, *, required=True):
        values = [os.getenv(k) for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")]
        if not all(values):
            if required:
                raise RuntimeError("NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD must be set")
            return None
        return cls(*values, database=database if database is not None else os.getenv("NEO4J_DATABASE", "neo4j"))

    def create(self, factory):
        return factory(self.uri, self.user, self.password, database=self.database)

def initialized_store(settings, factory, *, optional=False):
    if settings is None:
        return None
    store = settings.create(factory)
    try:
        if not store.health_check():
            if optional:
                store.close()
                return None
            raise RuntimeError("Neo4j health check failed")
        store.init_constraints()
        return store
    except Exception:
        store.close()
        raise
