"""
Agent 1 Matching Component: Sentence-BERT (SBERT) for Resume-Job Matching.

Encodes candidate profiles and job descriptions into dense vector embeddings
using a pretrained Sentence-BERT model, then computes cosine similarity to
determine match quality.

Architecture:
  SBERT (all-MiniLM-L6-v2) → Fixed-length 384-dim embeddings → Cosine similarity

Why SBERT: Creates semantically meaningful embeddings where similar meanings
cluster together in vector space. Unlike keyword matching, SBERT understands
that "P&ID review" and "piping and instrumentation diagram analysis" are
related concepts.

Course Connection: Module 05 - Transformers (sentence-level embeddings)
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import SBERT_MODEL_NAME, MATCH_THRESHOLD


class SBERTMatcher:
    """Match candidates to job descriptions using SBERT embeddings."""

    def __init__(self, model_name=SBERT_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.threshold = MATCH_THRESHOLD

    def encode(self, texts):
        """Encode a list of texts into dense embeddings."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def compute_match_score(self, candidate_text, job_text):
        """
        Compute cosine similarity between a candidate profile and job description.

        Returns a score in [0, 1] where 1 = perfect semantic match.
        """
        embeddings = self.encode([candidate_text, job_text])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(score)

    def rank_candidates(self, candidates, job_description):
        """
        Rank multiple candidates against a single job description.

        Args:
            candidates: List of dicts with 'id', 'name', 'text' keys
            job_description: Dict with 'id', 'title', 'text' keys

        Returns:
            Sorted list of dicts with match scores and skill gap analysis.
        """
        job_embedding = self.encode([job_description["text"]])[0]
        candidate_texts = [c["text"] for c in candidates]
        candidate_embeddings = self.encode(candidate_texts)

        results = []
        for i, candidate in enumerate(candidates):
            score = float(cosine_similarity(
                [candidate_embeddings[i]], [job_embedding]
            )[0][0])

            results.append({
                "candidate_id": candidate["id"],
                "candidate_name": candidate["name"],
                "job_id": job_description["id"],
                "job_title": job_description["title"],
                "match_score": round(score, 4),
                "is_match": score >= self.threshold,
                "match_tier": self._score_to_tier(score),
            })

        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results

    def batch_match(self, candidates, job_descriptions):
        """
        Match all candidates against all job descriptions.
        Returns a matrix of results organized by job.
        """
        all_results = {}
        for jd in job_descriptions:
            rankings = self.rank_candidates(candidates, jd)
            all_results[jd["id"]] = {
                "title": jd["title"],
                "rankings": rankings,
            }
        return all_results

    @staticmethod
    def _score_to_tier(score):
        """Convert numeric score to human-readable tier."""
        if score >= 0.80:
            return "Excellent Match"
        elif score >= 0.65:
            return "Strong Match"
        elif score >= 0.50:
            return "Moderate Match"
        elif score >= 0.35:
            return "Weak Match"
        else:
            return "Poor Match"

    def create_candidate_profile(self, entities, raw_text=""):
        """
        Build a structured candidate profile string from NER-extracted entities.
        Used to create the embedding input for matching.
        """
        parts = []
        if entities.get("SKILL"):
            parts.append(f"Skills: {', '.join(entities['SKILL'])}")
        if entities.get("CERT"):
            parts.append(f"Certifications: {', '.join(entities['CERT'])}")
        if entities.get("DEGREE"):
            parts.append(f"Education: {', '.join(entities['DEGREE'])}")
        if entities.get("EMPLOYER"):
            parts.append(f"Experience at: {', '.join(entities['EMPLOYER'])}")
        if entities.get("YEARS_EXP"):
            parts.append(f"Experience: {', '.join(entities['YEARS_EXP'])}")

        profile = ". ".join(parts)
        # Combine structured profile with raw text for richer embeddings
        if raw_text:
            return f"{profile}\n\n{raw_text}"
        return profile
