"""
Wikidata Knowledge Graph connector.

Provides access to Wikidata via SPARQL queries.
"""

from __future__ import annotations

import time
from typing import Any
from functools import lru_cache

import httpx

from neurosym_kg.core.config import get_config
from neurosym_kg.core.exceptions import KGConnectionError, KGQueryError, KGTimeoutError
from neurosym_kg.core.types import Entity, EntityType, Relation, RelationType, Subgraph, Triple
from neurosym_kg.knowledge_graphs.base import BaseKnowledgeGraph


# Wikidata property to RelationType mapping
WIKIDATA_RELATION_TYPES: dict[str, RelationType] = {
    "P31": RelationType.INSTANCE_OF,  # instance of
    "P279": RelationType.PART_OF,  # subclass of
    "P361": RelationType.PART_OF,  # part of
    "P527": RelationType.PART_OF,  # has part
    "P17": RelationType.SPATIAL,  # country
    "P131": RelationType.SPATIAL,  # located in
    "P625": RelationType.SPATIAL,  # coordinate location
    "P569": RelationType.TEMPORAL,  # date of birth
    "P570": RelationType.TEMPORAL,  # date of death
    "P580": RelationType.TEMPORAL,  # start time
    "P582": RelationType.TEMPORAL,  # end time
    "P26": RelationType.SOCIAL,  # spouse
    "P22": RelationType.SOCIAL,  # father
    "P25": RelationType.SOCIAL,  # mother
    "P40": RelationType.SOCIAL,  # child
}


class WikidataKG(BaseKnowledgeGraph):
    """
    Wikidata Knowledge Graph connector via SPARQL.

    Features:
    - Entity lookup by QID or name search
    - Neighbor retrieval with relation filtering
    - Path finding between entities
    - Caching of query results

    Example:
        >>> kg = WikidataKG()
        >>> entity = kg.get_entity("Q42")  # Douglas Adams
        >>> neighbors = kg.get_neighbors("Q42", direction="outgoing", limit=10)
    """

    # Common namespaces
    ENTITY_PREFIX = "http://www.wikidata.org/entity/"
    PROP_PREFIX = "http://www.wikidata.org/prop/direct/"

    def __init__(
        self,
        endpoint: str | None = None,
        user_agent: str | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
    ) -> None:
        super().__init__(name="Wikidata")

        config = get_config()
        self.endpoint = endpoint or config.kg.wikidata_endpoint
        self.user_agent = user_agent or config.kg.wikidata_user_agent
        self.timeout = timeout or config.kg.timeout_seconds
        self.max_retries = max_retries

        self._client = httpx.Client(
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
        )

    def _execute_sparql(self, query: str) -> list[dict[str, Any]]:
        """Execute a SPARQL query and return results."""
        self._increment_stat("queries")

        for attempt in range(self.max_retries):
            try:
                response = self._client.get(
                    self.endpoint,
                    params={"query": query, "format": "json"},
                )
                response.raise_for_status()
                data = response.json()
                return data.get("results", {}).get("bindings", [])

            except httpx.TimeoutException:
                if attempt == self.max_retries - 1:
                    raise KGTimeoutError("SPARQL query", self.timeout)
                time.sleep(1 * (attempt + 1))

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    time.sleep(5 * (attempt + 1))
                    continue
                raise KGQueryError(query[:200], str(e))

            except httpx.RequestError as e:
                raise KGConnectionError("Wikidata", str(e))

        raise KGQueryError(query[:200], "Max retries exceeded")

    def _parse_entity_uri(self, uri: str) -> str:
        """Extract QID from Wikidata URI."""
        if uri.startswith(self.ENTITY_PREFIX):
            return uri[len(self.ENTITY_PREFIX) :]
        return uri

    def _parse_property_uri(self, uri: str) -> str:
        """Extract PID from Wikidata property URI."""
        if uri.startswith(self.PROP_PREFIX):
            return uri[len(self.PROP_PREFIX) :]
        if "/prop/direct/" in uri:
            return uri.split("/prop/direct/")[-1]
        if "/entity/" in uri:
            return uri.split("/entity/")[-1]
        return uri

    def get_entity(self, entity_id: str) -> Entity | None:
        """Retrieve an entity by its Wikidata QID."""
        # Normalize QID
        if not entity_id.startswith("Q"):
            entity_id = f"Q{entity_id}"

        query = f"""
        SELECT ?label ?description ?altLabel WHERE {{
            wd:{entity_id} rdfs:label ?label .
            FILTER(LANG(?label) = "en")
            OPTIONAL {{
                wd:{entity_id} schema:description ?description .
                FILTER(LANG(?description) = "en")
            }}
            OPTIONAL {{
                wd:{entity_id} skos:altLabel ?altLabel .
                FILTER(LANG(?altLabel) = "en")
            }}
        }}
        """

        try:
            results = self._execute_sparql(query)
        except Exception:
            return None

        if not results:
            return None

        # Aggregate results
        labels = set()
        descriptions = set()
        aliases = set()

        for row in results:
            if "label" in row:
                labels.add(row["label"]["value"])
            if "description" in row:
                descriptions.add(row["description"]["value"])
            if "altLabel" in row:
                aliases.add(row["altLabel"]["value"])

        name = next(iter(labels)) if labels else entity_id
        description = next(iter(descriptions)) if descriptions else ""

        # Try to determine entity type from instance_of
        entity_type = self._get_entity_type(entity_id)

        return Entity(
            id=entity_id,
            name=name,
            aliases=list(aliases - {name}),
            description=description,
            entity_type=entity_type,
        )

    def _get_entity_type(self, entity_id: str) -> EntityType:
        """Determine entity type from Wikidata instance_of (P31)."""
        query = f"""
        SELECT ?type WHERE {{
            wd:{entity_id} wdt:P31 ?type .
        }} LIMIT 5
        """

        try:
            results = self._execute_sparql(query)
        except Exception:
            return EntityType.UNKNOWN

        # Map Wikidata types to EntityType
        type_mapping = {
            "Q5": EntityType.PERSON,  # human
            "Q43229": EntityType.ORGANIZATION,  # organization
            "Q515": EntityType.LOCATION,  # city
            "Q6256": EntityType.LOCATION,  # country
            "Q35120": EntityType.THING,  # entity
        }

        for row in results:
            type_qid = self._parse_entity_uri(row["type"]["value"])
            if type_qid in type_mapping:
                return type_mapping[type_qid]

        return EntityType.UNKNOWN

    def get_entity_by_name(self, name: str, limit: int = 10) -> list[Entity]:
        """Search for entities by name using Wikidata's search API."""
        # Use MediaWiki API for better search
        search_url = "https://www.wikidata.org/w/api.php"

        try:
            response = self._client.get(
                search_url,
                params={
                    "action": "wbsearchentities",
                    "search": name,
                    "language": "en",
                    "limit": limit,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []

        entities = []
        for item in data.get("search", []):
            entities.append(
                Entity(
                    id=item["id"],
                    name=item.get("label", item["id"]),
                    description=item.get("description", ""),
                    aliases=item.get("aliases", []),
                )
            )

        return entities

    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",
        relation_filter: list[str] | None = None,
        limit: int = 100,
    ) -> list[Triple]:
        """Get neighboring triples for an entity."""
        if not entity_id.startswith("Q"):
            entity_id = f"Q{entity_id}"

        triples: list[Triple] = []

        # Build relation filter clause
        filter_clause = ""
        if relation_filter:
            pids = [f"wdt:{r if r.startswith('P') else 'P' + r}" for r in relation_filter]
            filter_clause = f"FILTER(?p IN ({', '.join(pids)}))"

        # Outgoing edges
        if direction in ("outgoing", "both"):
            query = f"""
            SELECT ?p ?pLabel ?o ?oLabel WHERE {{
                wd:{entity_id} ?p ?o .
                ?prop wikibase:directClaim ?p .
                {filter_clause}
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }} LIMIT {limit}
            """

            try:
                results = self._execute_sparql(query)
                for row in results:
                    pred_id = self._parse_property_uri(row["p"]["value"])
                    obj_id = self._parse_entity_uri(row["o"]["value"])

                    triples.append(
                        Triple(
                            subject=entity_id,
                            predicate=pred_id,
                            object=obj_id if obj_id.startswith("Q") else row["o"]["value"],
                        )
                    )
            except Exception:
                pass

        # Incoming edges
        if direction in ("incoming", "both") and len(triples) < limit:
            query = f"""
            SELECT ?s ?sLabel ?p ?pLabel WHERE {{
                ?s ?p wd:{entity_id} .
                ?prop wikibase:directClaim ?p .
                {filter_clause}
            }} LIMIT {limit - len(triples)}
            """

            try:
                results = self._execute_sparql(query)
                for row in results:
                    subj_id = self._parse_entity_uri(row["s"]["value"])
                    pred_id = self._parse_property_uri(row["p"]["value"])

                    if subj_id.startswith("Q"):
                        triples.append(
                            Triple(
                                subject=subj_id,
                                predicate=pred_id,
                                object=entity_id,
                            )
                        )
            except Exception:
                pass

        return triples[:limit]

    def get_relations(self, entity_id: str) -> list[Relation]:
        """Get all relations connected to an entity."""
        if not entity_id.startswith("Q"):
            entity_id = f"Q{entity_id}"

        query = f"""
        SELECT DISTINCT ?p ?pLabel ?pDescription WHERE {{
            {{
                wd:{entity_id} ?p ?o .
            }} UNION {{
                ?s ?p wd:{entity_id} .
            }}
            ?prop wikibase:directClaim ?p .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """

        try:
            results = self._execute_sparql(query)
        except Exception:
            return []

        relations = []
        seen = set()

        for row in results:
            pred_id = self._parse_property_uri(row["p"]["value"])
            if pred_id in seen:
                continue
            seen.add(pred_id)

            relations.append(
                Relation(
                    id=pred_id,
                    name=row.get("pLabel", {}).get("value", pred_id),
                    description=row.get("pDescription", {}).get("value", ""),
                    relation_type=WIKIDATA_RELATION_TYPES.get(pred_id, RelationType.OTHER),
                )
            )

        return relations

    def get_triples(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        limit: int = 100,
    ) -> list[Triple]:
        """Query triples with optional filters."""
        # Build query parts
        subj_part = f"wd:{subject}" if subject else "?s"
        pred_part = f"wdt:{predicate}" if predicate else "?p"
        obj_part = f"wd:{obj}" if obj else "?o"

        # Need at least one filter
        if not any([subject, predicate, obj]):
            return []

        query = f"""
        SELECT {"?s" if not subject else ""} {"?p" if not predicate else ""} {"?o" if not obj else ""}
        WHERE {{
            {subj_part} {pred_part} {obj_part} .
            {"?prop wikibase:directClaim ?p ." if not predicate else ""}
        }} LIMIT {limit}
        """

        try:
            results = self._execute_sparql(query)
        except Exception:
            return []

        triples = []
        for row in results:
            s = subject or self._parse_entity_uri(row["s"]["value"])
            p = predicate or self._parse_property_uri(row["p"]["value"])
            o = obj or (
                self._parse_entity_uri(row["o"]["value"])
                if row["o"]["type"] == "uri"
                else row["o"]["value"]
            )

            triples.append(Triple(subject=s, predicate=p, object=o))

        return triples

    def find_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
        max_paths: int = 10,
    ) -> list[list[Triple]]:
        """Find paths between two entities using property paths."""
        # Normalize QIDs
        if not source.startswith("Q"):
            source = f"Q{source}"
        if not target.startswith("Q"):
            target = f"Q{target}"

        paths: list[list[Triple]] = []

        # Try to find paths of increasing length
        for path_length in range(1, max_hops + 1):
            if len(paths) >= max_paths:
                break

            # Build path query
            path_vars = " ".join([f"?p{i} ?e{i}" for i in range(path_length)])
            path_pattern = []

            for i in range(path_length):
                subj = f"?e{i-1}" if i > 0 else f"wd:{source}"
                obj = f"?e{i}" if i < path_length - 1 else f"wd:{target}"
                path_pattern.append(f"{subj} ?p{i} {obj} .")
                path_pattern.append(f"?prop{i} wikibase:directClaim ?p{i} .")

            query = f"""
            SELECT {path_vars} WHERE {{
                {' '.join(path_pattern)}
            }} LIMIT {max_paths - len(paths)}
            """

            try:
                results = self._execute_sparql(query)
                for row in results:
                    path = []
                    current = source
                    for i in range(path_length):
                        pred = self._parse_property_uri(row[f"p{i}"]["value"])
                        next_entity = (
                            self._parse_entity_uri(row[f"e{i}"]["value"])
                            if i < path_length - 1
                            else target
                        )
                        path.append(Triple(subject=current, predicate=pred, object=next_entity))
                        current = next_entity
                    paths.append(path)
            except Exception:
                continue

        return paths[:max_paths]

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "WikidataKG":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
