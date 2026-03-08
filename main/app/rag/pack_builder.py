"""
Build citation-only intelligence packs from retrieval + reranking + classifiers.

All URLs and issue IDs in the output MUST come from the retrieved set—no hallucinated links.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.retrieval.reranker import RankedHit
from app.ml.classifiers import predict_dict


@dataclass
class SimilarIssueEntry:
    id: str
    url: str
    title: str
    score: float
    rerank_score: float
    snippet: str


@dataclass
class IntelligencePack:
    query_text: str
    query_issue_id: str | None
    similar_issues: list[SimilarIssueEntry]
    predictions: dict
    suggested_next_action: str
    citation_issue_ids: list[str]


def _pick_action(predictions: dict, similar_issues: list[SimilarIssueEntry]) -> str:
    """
    Determine suggested_next_action from classifier output.
    If the classifier is low confidence and many high-scoring similar issues exist,
    hint at 'duplicate' or 'triage'.
    """
    action = predictions.get("action_recommendation", "triage")
    action_conf = predictions.get("action_confidence", 0.0)

    if action_conf >= 0.5:
        return action

    if similar_issues and similar_issues[0].rerank_score > 5.0:
        return "duplicate"

    return "triage"


def build_pack(
    query_text: str,
    ranked_hits: list[RankedHit],
    query_issue_id: str | None = None,
    labels: list[str] | None = None,
) -> IntelligencePack:
    """
    Assemble the full intelligence pack:
      - similar_issues  (from reranked hits)
      - predictions     (from classifiers on query text)
      - suggested_next_action
      - citation_issue_ids (only IDs present in retrieval)
    """
    similar: list[SimilarIssueEntry] = []
    citation_ids: list[str] = []

    for hit in ranked_hits:
        entry = SimilarIssueEntry(
            id=hit.issue_id,
            url=hit.url,
            title=hit.title,
            score=hit.rrf_score,
            rerank_score=hit.rerank_score,
            snippet=hit.snippet,
        )
        similar.append(entry)
        citation_ids.append(hit.issue_id)

    preds = predict_dict(query_text, labels=labels)
    action = _pick_action(preds, similar)

    return IntelligencePack(
        query_text=query_text,
        query_issue_id=query_issue_id,
        similar_issues=similar,
        predictions=preds,
        suggested_next_action=action,
        citation_issue_ids=citation_ids,
    )


def pack_to_dict(pack: IntelligencePack) -> dict:
    """Serialize an IntelligencePack to a JSON-friendly dict."""
    return {
        "query_text": pack.query_text,
        "query_issue_id": pack.query_issue_id,
        "similar_issues": [
            {
                "id": si.id,
                "url": si.url,
                "title": si.title,
                "score": round(si.score, 4),
                "rerank_score": round(si.rerank_score, 4),
                "snippet": si.snippet,
            }
            for si in pack.similar_issues
        ],
        "predictions": pack.predictions,
        "suggested_next_action": pack.suggested_next_action,
        "citation_issue_ids": pack.citation_issue_ids,
    }
