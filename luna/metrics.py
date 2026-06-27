"""
luna.metrics
~~~~~~~~~~~~~
Metrics API for Luna pipeline — performance tracking and benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricsReport:
    """
    Immutable performance metrics report for a Luna pipeline run.
    
    All times are in seconds. Use `to_dict()` for JSON serialization.
    Use `print(metrics)` for a Rich-formatted table output.
    
    Fields are grouped by pipeline stage:
    - Ingest: Tile embedding and index compilation
    - Pithos Search: Vector database query performance  
    - Candidates: NMS filtering statistics
    - ESSA: Second-stage refinement
    - Total: End-to-end timing
    """
    
    # Ingest metrics
    ingest_s: float = 0.0
    ingest_tiles: int = 0
    ingest_tiles_per_s: float = 0.0
    index_compile_s: float = 0.0
    index_size_bytes: int = 0
    
    # Pithos Search metrics
    index_load_s: float = 0.0
    knn_search_s: float = 0.0
    knn_ms_per_query: float = 0.0
    n_queries: int = 0
    index_vectors: int = 0
    index_tiers: list[int] = field(default_factory=list)
    
    # Candidates metrics
    n_candidates_raw: int = 0
    nms_s: float = 0.0
    n_candidates_nms: int = 0
    
    # ESSA metrics
    essa_s: float = 0.0
    essa_hits_in: int = 0
    essa_hits_out: int = 0
    essa_refinement_ratio: float = 0.0
    
    # Total metrics
    total_scan_s: float = 0.0
    total_s: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return {
            # Ingest
            "ingest_s": self.ingest_s,
            "ingest_tiles": self.ingest_tiles,
            "ingest_tiles_per_s": self.ingest_tiles_per_s,
            "index_compile_s": self.index_compile_s,
            "index_size_bytes": self.index_size_bytes,
            
            # Pithos Search
            "index_load_s": self.index_load_s,
            "knn_search_s": self.knn_search_s,
            "knn_ms_per_query": self.knn_ms_per_query,
            "n_queries": self.n_queries,
            "index_vectors": self.index_vectors,
            "index_tiers": self.index_tiers,
            
            # Candidates
            "n_candidates_raw": self.n_candidates_raw,
            "nms_s": self.nms_s,
            "n_candidates_nms": self.n_candidates_nms,
            
            # ESSA
            "essa_s": self.essa_s,
            "essa_hits_in": self.essa_hits_in,
            "essa_hits_out": self.essa_hits_out,
            "essa_refinement_ratio": self.essa_refinement_ratio,
            
            # Total
            "total_scan_s": self.total_scan_s,
            "total_s": self.total_s,
        }
    
    @classmethod
    def from_trace(cls, trace: dict, **overrides: Any) -> MetricsReport:
        """
        Build a MetricsReport from an internal trace dictionary.
        
        The trace dictionary contains low-level timing keys that are assembled
        into the structured MetricsReport format.
        """
        # Extract all trace values
        trace_values = {k: v for k, v in trace.items() if isinstance(v, (int, float))}
        
        return cls(
            # Ingest - extracted from trace
            ingest_s=trace_values.get("ingest_s", 0.0),
            ingest_tiles=trace_values.get("ingest_tiles", 0),
            ingest_tiles_per_s=trace_values.get("ingest_tiles_per_s", 0.0),
            index_compile_s=trace_values.get("index_compile_s", 0.0),
            index_size_bytes=trace_values.get("index_size_bytes", 0),
            
            # Pithos Search
            index_load_s=trace_values.get("p1_pithos_index_scan", 0.0),
            knn_search_s=trace_values.get("p1_pithos_index_scan", 0.0),
            knn_ms_per_query=trace_values.get("knn_ms_per_query", 0.0),
            n_queries=trace_values.get("n_queries", 0),
            index_vectors=trace_values.get("index_vectors", 0),
            index_tiers=trace_values.get("index_tiers", []),
            
            # Candidates
            n_candidates_raw=trace_values.get("n_candidates_raw", 0),
            nms_s=trace_values.get("p1_cpu_nms_filtering", 0.0),
            n_candidates_nms=trace_values.get("n_candidates_nms", 0),
            
            # ESSA
            essa_s=trace_values.get("p2_essa_inference", 0.0),
            essa_hits_in=trace_values.get("essa_hits_in", 0),
            essa_hits_out=trace_values.get("essa_hits_out", 0),
            essa_refinement_ratio=trace_values.get("essa_refinement_ratio", 0.0),
            
            # Total
            total_scan_s=trace_values.get("total_scan_s", 0.0),
            total_s=trace_values.get("total_s", 0.0),
            
            **overrides
        )
    
    def __repr__(self) -> str:
        """Rich console representation."""
        lines = [
            "MetricsReport:",
            "─" * 50,
            "",
            "  INGEST:",
            f"    Tiles: {self.ingest_tiles} ({self.ingest_tiles_per_s:.1f}/s)",
            f"    Index compile: {self.index_compile_s:.3f}s",
            f"    Index size: {self._format_bytes(self.index_size_bytes)}",
            "",
            "  PITHOS SEARCH:",
            f"    Index load: {self.index_load_s:.3f}s",
            f"    KNN search: {self.knn_search_s:.3f}s ({self.knn_ms_per_query:.2f}ms/query)",
            f"    Queries: {self.n_queries}, Vectors: {self.index_vectors}, Tiers: {self.index_tiers}",
            "",
            "  CANDIDATES:",
            f"    Raw candidates: {self.n_candidates_raw}",
            f"    NMS: {self.nms_s:.3f}s → {self.n_candidates_nms} retained",
            "",
            "  ESSA:",
            f"    Refinement: {self.essa_s:.3f}s",
            f"    Hits: {self.essa_hits_in} → {self.essa_hits_out} ({self.essa_refinement_ratio:.1%} ratio)",
            "",
            "  TOTAL:",
            f"    Scan: {self.total_scan_s:.3f}s",
            f"    End-to-end: {self.total_s:.3f}s",
        ]
        return "\n".join(lines)
    
    def _repr_html_(self) -> str:
        """Jupyter notebook HTML representation."""
        html = ['<table style="font-family: monospace; border-collapse: collapse;">']
        html.append('<tr><th colspan="2" style="text-align: left; border-bottom: 2px solid #ccc;"><b>MetricsReport</b></th></tr>')
        
        sections = [
            ("INGEST", [
                ("Tiles", f"{self.ingest_tiles} ({self.ingest_tiles_per_s:.1f}/s)"),
                ("Index compile", f"{self.index_compile_s:.3f}s"),
                ("Index size", self._format_bytes(self.index_size_bytes)),
            ]),
            ("PITHOS SEARCH", [
                ("Index load", f"{self.index_load_s:.3f}s"),
                ("KNN search", f"{self.knn_search_s:.3f}s ({self.knn_ms_per_query:.2f}ms/query)"),
                ("Queries/Vectors/Tiers", f"{self.n_queries}/{self.index_vectors}/{self.index_tiers}"),
            ]),
            ("CANDIDATES", [
                ("Raw candidates", str(self.n_candidates_raw)),
                ("NMS", f"{self.nms_s:.3f}s → {self.n_candidates_nms} retained"),
            ]),
            ("ESSA", [
                ("Refinement", f"{self.essa_s:.3f}s"),
                ("Hits", f"{self.essa_hits_in} → {self.essa_hits_out} ({self.essa_refinement_ratio:.1%} ratio)"),
            ]),
            ("TOTAL", [
                ("Scan", f"{self.total_scan_s:.3f}s"),
                ("End-to-end", f"{self.total_s:.3f}s"),
            ]),
        ]
        
        for section, rows in sections:
            html.append(f'<tr><th colspan="2" style="text-align: left; background: #f0f0f0;">{section}</th></tr>')
            for label, value in rows:
                html.append(f'<tr><td style="padding: 2px 10px;">{label}</td><td style="padding: 2px 10px;">{value}</td></tr>')
        
        html.append('</table>')
        return "".join(html)
    
    @staticmethod
    def _format_bytes(n: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(n) < 1024.0:
                return f"{n:.1f} {unit}"
            n /= 1024.0
        return f"{n:.1f} PB"
