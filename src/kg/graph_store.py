# graph store for the financial knowledge graph

# In-memory graph for MVP
# Neo4j for future development

import networkx as nx
from src.kg.schema import Evidence, FinancialSignal, GraphUpdate


class GraphStore:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity_id: str, **attrs) -> None:
        self.graph.add_node(entity_id, node_type="entity", **attrs)

    def add_evidence(self, evidence: Evidence) -> None:
        self.graph.add_node(
            evidence.evidence_id,
            node_type="evidence",
            **evidence.model_dump(),
        )

    def add_signal(self, signal: FinancialSignal) -> None:
        self.graph.add_node(
            signal.signal_id,
            node_type="signal",
            **signal.model_dump(),
        )

        if not self.graph.has_node(signal.ticker):
            self.add_entity(
                signal.ticker,
                name=signal.ticker,
                entity_type="stock",
            )

        self.graph.add_edge(
            signal.ticker,
            signal.signal_id,
            relation_type="HAS_SIGNAL",
        )

        self.graph.add_edge(
            signal.signal_id,
            signal.evidence_id,
            relation_type="SUPPORTED_BY",
        )

    def apply_update(self, update: GraphUpdate) -> None:
        self.add_evidence(update.evidence)
        self.add_signal(update.signal)

    def get_signals_for_ticker(self, ticker: str) -> list[dict]:
        signals = []

        if not self.graph.has_node(ticker):
            return signals

        for _, signal_id, attrs in self.graph.out_edges(ticker, data=True):
            if attrs.get("relation_type") == "HAS_SIGNAL":
                node_data = self.graph.nodes[signal_id]
                signals.append(dict(node_data))

        return signals

    def get_evidence(self, evidence_id: str) -> dict | None:
        if not self.graph.has_node(evidence_id):
            return None
        return dict(self.graph.nodes[evidence_id])

    def get_graph_summary(self) -> dict:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
        }