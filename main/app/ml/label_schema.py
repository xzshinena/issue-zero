"""Canonical label sets shared by classifier serving and training."""

URGENCY_LABELS = ["critical_bug", "high", "medium", "low", "enhancement", "question"]
ISSUE_TYPE_LABELS = ["bug", "feature_request", "docs", "refactor", "regression", "question"]
ACTION_LABELS = ["triage", "assign_to_area", "need_more_info", "duplicate", "close"]
IS_REGRESSION_LABELS = ["true", "false"]
