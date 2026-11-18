"""Shared graph ingestion primitives; builders retain their existing semantics."""
from src.kg.schema import Entity

def company_entity(ticker):
    return Entity(entity_id=f"company:{ticker}", type="Company", properties={"ticker": ticker})

def write_graph_batch(store, entities, evidences, relations):
    store.upsert_entities(entities)
    if evidences is not None:
        store.upsert_evidences(evidences)
    store.upsert_relations(relations)

def ingest_frame(store, builder, *args, **kwargs):
    entities, evidences, relations = builder(*args, **kwargs)
    write_graph_batch(store, entities, evidences, relations)
    return len(entities), len(evidences), len(relations)
