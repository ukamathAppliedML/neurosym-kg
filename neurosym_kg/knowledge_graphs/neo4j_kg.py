"""
Neo4j Knowledge Graph Backend.

Connects to a Neo4j graph database for production-scale knowledge graph operations.
Supports both Neo4j Community and Enterprise editions.

Requirements:
    pip install neo4j

Example:
    from neurosym_kg.knowledge_graphs import Neo4jKG
    
    kg = Neo4jKG(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j"
    )
    
    # Query neighbors
    neighbors = kg.get_neighbors("Albert_Einstein")
    
    # Get relations
    relations = kg.get_relations("Albert_Einstein")
    
    # Execute Cypher query
    results = kg.execute_cypher(
        "MATCH (n:Person)-[r]->(m) WHERE n.name = $name RETURN m.name, type(r)",
        {"name": "Albert Einstein"}
    )
"""

from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass
import logging

from .base import BaseKnowledgeGraph, BaseMutableKnowledgeGraph
from ..core.types import Entity, Triple, Relation, Subgraph

logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    encrypted: bool = False
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0


class Neo4jKG(BaseMutableKnowledgeGraph):
    """
    Neo4j-backed Knowledge Graph.
    
    Provides a KnowledgeGraph interface over a Neo4j database, enabling
    integration with existing enterprise graph deployments.
    
    Node Mapping:
        - Entities become Neo4j nodes with label :Entity
        - Entity IDs stored as 'id' property
        - Entity names stored as 'name' property
        - Additional properties supported
    
    Relationship Mapping:
        - Triples become Neo4j relationships
        - Relationship type = predicate
        - Properties can be attached to relationships
    
    Example Cypher representation:
        Triple("Einstein", "born_in", "Germany")
        →  (:Entity {id: "Einstein"})-[:born_in]->(:Entity {id: "Germany"})
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        config: Optional[Neo4jConfig] = None,
        name: str = "Neo4j KG",
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j bolt URI (e.g., "bolt://localhost:7687")
            username: Neo4j username
            password: Neo4j password
            database: Database name (default: "neo4j")
            config: Optional Neo4jConfig for advanced settings
            name: Human-readable name for this KG instance
        """
        self._name = name
        self.config = config or Neo4jConfig(
            uri=uri,
            username=username,
            password=password,
            database=database,
        )
        
        self._driver = None
        self._connected = False
        
        # Try to import neo4j
        try:
            import neo4j
            self._neo4j = neo4j
        except ImportError:
            raise ImportError(
                "neo4j package not installed. Install with: pip install neo4j"
            )
        
        # Establish connection
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self._driver = self._neo4j.GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                encrypted=self.config.encrypted,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_timeout,
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            self._connected = True
            logger.info(f"Connected to Neo4j at {self.config.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Cannot connect to Neo4j at {self.config.uri}: {e}")
    
    def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._connected = False
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a raw Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        if not self._connected:
            raise ConnectionError("Not connected to Neo4j")
        
        with self._driver.session(database=self.config.database) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        query = """
        MATCH (e:Entity {id: $id})
        RETURN e.id as id, e.name as name, e.description as description, labels(e) as labels
        """
        results = self.execute_cypher(query, {"id": entity_id})
        
        if not results:
            return None
        
        record = results[0]
        return Entity(
            id=record["id"],
            name=record.get("name") or record["id"],  # Fallback to id if name is None
            description=record.get("description") or "",  # Fallback to empty string
        )
    
    def get_neighbors(
        self,
        entity_id: str,
        relation: Optional[str] = None,
        direction: str = "both",
        limit: int = 100,
    ) -> List[Entity]:
        """
        Get neighboring entities.
        
        Args:
            entity_id: Source entity ID
            relation: Filter by relation type (optional)
            direction: "outgoing", "incoming", or "both"
            limit: Maximum neighbors to return
            
        Returns:
            List of neighboring entities
        """
        # Build direction-specific pattern
        if direction == "outgoing":
            pattern = "(e)-[r]->(n)"
        elif direction == "incoming":
            pattern = "(e)<-[r]-(n)"
        else:
            pattern = "(e)-[r]-(n)"
        
        # Build relation filter
        rel_filter = f":{relation}" if relation else ""
        pattern = pattern.replace("[r]", f"[r{rel_filter}]")
        
        query = f"""
        MATCH {pattern}
        WHERE e.id = $id
        RETURN DISTINCT n.id as id, n.name as name, n.description as description
        LIMIT $limit
        """
        
        results = self.execute_cypher(query, {"id": entity_id, "limit": limit})
        
        return [
            Entity(
                id=r["id"],
                name=r.get("name") or r["id"],  # Fallback to id if name is None
                description=r.get("description") or "",  # Fallback to empty string
            )
            for r in results
        ]
    
    def get_relations(
        self,
        entity_id: str,
        direction: str = "both",
    ) -> List[Relation]:
        """
        Get all relations for an entity.
        
        Args:
            entity_id: Entity ID
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            List of relations with their target entities
        """
        if direction == "outgoing":
            query = """
            MATCH (e:Entity {id: $id})-[r]->(n)
            RETURN type(r) as relation, n.id as target, 'outgoing' as direction
            """
        elif direction == "incoming":
            query = """
            MATCH (e:Entity {id: $id})<-[r]-(n)
            RETURN type(r) as relation, n.id as source, 'incoming' as direction
            """
        else:
            query = """
            MATCH (e:Entity {id: $id})-[r]-(n)
            RETURN type(r) as relation, n.id as other, 
                   CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END as direction
            """
        
        results = self.execute_cypher(query, {"id": entity_id})
        
        relations = []
        for r in results:
            rel_name = r["relation"]
            if r["direction"] == "outgoing":
                source = entity_id
                target = r.get("target", r.get("other"))
            else:
                source = r.get("source", r.get("other"))
                target = entity_id
            
            rel = Relation(
                id=rel_name,  # Use relation name as id
                name=rel_name,
                description=f"{source} -> {rel_name} -> {target}",
            )
            relations.append(rel)
        
        return relations
    
    def get_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Triple]:
        """
        Query triples with optional filters.
        
        Args:
            subject: Filter by subject entity ID
            predicate: Filter by relation type
            object: Filter by object entity ID
            limit: Maximum triples to return
            
        Returns:
            List of matching triples
        """
        conditions = []
        params = {"limit": limit}
        
        if subject:
            conditions.append("s.id = $subject")
            params["subject"] = subject
        
        if object:
            conditions.append("o.id = $object")
            params["object"] = object
        
        rel_pattern = f":{predicate}" if predicate else ""
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        query = f"""
        MATCH (s:Entity)-[r{rel_pattern}]->(o:Entity)
        {where_clause}
        RETURN s.id as subject, type(r) as predicate, o.id as object
        LIMIT $limit
        """
        
        results = self.execute_cypher(query, params)
        
        return [
            Triple(
                subject=r["subject"],
                predicate=r["predicate"],
                object=r["object"],
            )
            for r in results
        ]
    
    def add_entity(
        self,
        entity: Entity,
        labels: Optional[List[str]] = None,
    ) -> bool:
        """
        Add an entity to the graph.
        
        Args:
            entity: Entity to add
            labels: Additional Neo4j labels (default: ["Entity"])
            
        Returns:
            True if successful
        """
        all_labels = ["Entity"] + (labels or [])
        label_str = ":".join(all_labels)
        
        query = f"""
        MERGE (e:{label_str} {{id: $id}})
        SET e.name = $name, e.description = $description
        """
        
        try:
            self.execute_cypher(query, {
                "id": entity.id,
                "name": entity.name or entity.id,
                "description": entity.description,
            })
            return True
        except Exception:
            return False
    
    def add_triple(self, triple: Triple) -> bool:
        """
        Add a triple (relationship) to the graph.
        
        Args:
            triple: Triple to add
            
        Returns:
            True if successful
        """
        # Sanitize predicate for Neo4j (must be valid identifier)
        safe_predicate = triple.predicate.replace(" ", "_").replace("-", "_")
        
        query = f"""
        MERGE (s:Entity {{id: $subject}})
        ON CREATE SET s.name = $subject
        MERGE (o:Entity {{id: $object}})
        ON CREATE SET o.name = $object
        MERGE (s)-[r:{safe_predicate}]->(o)
        """
        
        try:
            self.execute_cypher(query, {
                "subject": str(triple.subject),
                "object": str(triple.object),
            })
            return True
        except Exception:
            return False
    
    def add_triples(self, triples: List[Triple]) -> int:
        """
        Bulk add triples (more efficient than individual adds).
        
        Args:
            triples: List of triples to add
            
        Returns:
            Number of triples added
        """
        # Group by predicate for efficient batch insertion
        by_predicate: Dict[str, List[Tuple[str, str]]] = {}
        for t in triples:
            pred = t.predicate.replace(" ", "_").replace("-", "_")
            if pred not in by_predicate:
                by_predicate[pred] = []
            by_predicate[pred].append((str(t.subject), str(t.object)))
        
        # Batch insert per predicate type
        for predicate, pairs in by_predicate.items():
            query = f"""
            UNWIND $pairs AS pair
            MERGE (s:Entity {{id: pair[0]}})
            ON CREATE SET s.name = pair[0]
            MERGE (o:Entity {{id: pair[1]}})
            ON CREATE SET o.name = pair[1]
            MERGE (s)-[r:{predicate}]->(o)
            """
            self.execute_cypher(query, {"pairs": pairs})
        
        logger.info(f"Added {len(triples)} triples to Neo4j")
        return len(triples)
    
    def delete_triple(self, triple: Triple) -> None:
        """Delete a specific triple."""
        safe_predicate = triple.predicate.replace(" ", "_").replace("-", "_")
        
        query = f"""
        MATCH (s:Entity {{id: $subject}})-[r:{safe_predicate}]->(o:Entity {{id: $object}})
        DELETE r
        """
        
        self.execute_cypher(query, {
            "subject": str(triple.subject),
            "object": str(triple.object),
        })
    
    def search_entities(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Entity]:
        """
        Search entities by name (fuzzy match).
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            Matching entities
        """
        # Use CONTAINS for simple substring search
        # For production, consider Neo4j full-text indexes
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($query) 
           OR toLower(e.id) CONTAINS toLower($query)
        RETURN e.id as id, e.name as name, e.description as description
        LIMIT $limit
        """
        
        results = self.execute_cypher(cypher, {"query": query, "limit": limit})
        
        return [
            Entity(
                id=r["id"],
                name=r.get("name") or r["id"],  # Fallback to id if name is None
                description=r.get("description") or "",  # Fallback to empty string
            )
            for r in results
        ]
    
    def get_entity_by_name(self, name: str, limit: int = 10) -> List[Entity]:
        """Search for entities by name (required by base class)."""
        return self.search_entities(name, limit)
    
    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and all its relationships."""
        try:
            self.execute_cypher(
                "MATCH (e:Entity {id: $id}) DETACH DELETE e",
                {"id": entity_id}
            )
            return True
        except Exception:
            return False
    
    def remove_triple(self, triple: Triple) -> bool:
        """Remove a specific triple (alias for delete_triple)."""
        try:
            self.delete_triple(triple)
            return True
        except Exception:
            return False
    
    def get_shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> Optional[List[Triple]]:
        """
        Find shortest path between two entities.
        
        Args:
            source_id: Starting entity ID
            target_id: Target entity ID
            max_depth: Maximum path length
            
        Returns:
            List of triples forming the path, or None if no path exists
        """
        # Note: max_depth must be interpolated into query string (Neo4j limitation)
        query = f"""
        MATCH path = shortestPath(
            (s:Entity {{id: $source}})-[*1..{max_depth}]-(t:Entity {{id: $target}})
        )
        RETURN [r IN relationships(path) | {{
            subject: startNode(r).id,
            predicate: type(r),
            object: endNode(r).id
        }}] as triples
        """
        
        results = self.execute_cypher(query, {
            "source": source_id,
            "target": target_id,
        })
        
        if not results or not results[0]["triples"]:
            return None
        
        return [
            Triple(
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
            )
            for t in results[0]["triples"]
        ]
    
    def get_subgraph(
        self,
        entity_ids: List[str],
        max_hops: int = 1,
    ) -> List[Triple]:
        """
        Extract subgraph around given entities.
        
        Args:
            entity_ids: Center entities
            max_hops: How many hops to expand
            
        Returns:
            All triples within the subgraph
        """
        # Note: max_hops must be interpolated into query string (Neo4j limitation)
        query = f"""
        MATCH (e:Entity)
        WHERE e.id IN $ids
        CALL {{
            WITH e
            MATCH path = (e)-[*1..{max_hops}]-(n)
            UNWIND relationships(path) as r
            RETURN DISTINCT startNode(r).id as subject, 
                   type(r) as predicate, 
                   endNode(r).id as object
        }}
        RETURN DISTINCT subject, predicate, object
        """
        
        results = self.execute_cypher(query, {
            "ids": entity_ids,
        })
        
        return [
            Triple(
                subject=r["subject"],
                predicate=r["predicate"],
                object=r["object"],
            )
            for r in results
        ]
    
    @property
    def num_entities(self) -> int:
        """Count of entities in the graph."""
        results = self.execute_cypher("MATCH (e:Entity) RETURN count(e) as count")
        return results[0]["count"] if results else 0
    
    @property
    def num_triples(self) -> int:
        """Count of relationships in the graph."""
        results = self.execute_cypher("MATCH ()-[r]->() RETURN count(r) as count")
        return results[0]["count"] if results else 0
    
    def get_relation_types(self) -> List[str]:
        """Get all distinct relation types in the graph."""
        results = self.execute_cypher(
            "MATCH ()-[r]->() RETURN DISTINCT type(r) as rel_type"
        )
        return [r["rel_type"] for r in results]
    
    def clear(self) -> None:
        """Delete all nodes and relationships (use with caution!)."""
        self.execute_cypher("MATCH (n) DETACH DELETE n")
        logger.warning("Cleared all data from Neo4j database")
    
    def create_indexes(self) -> None:
        """Create recommended indexes for performance."""
        indexes = [
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        ]
        
        for idx_query in indexes:
            try:
                self.execute_cypher(idx_query)
            except Exception as e:
                logger.warning(f"Index creation skipped: {e}")
        
        logger.info("Created Neo4j indexes")
