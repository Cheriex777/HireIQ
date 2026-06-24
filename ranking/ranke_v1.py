import json

SEARCH_KEYWORDS = [
    "retrieval",
    "ranking",
    "search",
    "embedding",
    "vector",
    "recommendation",
    "information retrieval",
    "semantic search",
    "rag"
]

GOOD_TITLES = [
    "ai",
    "machine learning",
    "ml engineer",
    "nlp",
    "recommendation",
    "data scientist",
    "search"
]


def title_score(title):
    title = title.lower()

    score = 0

    for keyword in GOOD_TITLES:
        if keyword in title:
            score += 20

    return score


def retrieval_score(text):
    text = text.lower()

    score = 0

    for keyword in SEARCH_KEYWORDS:
        if keyword in text:
            score += 10

    return score


def behavior_score(signals):

    score = 0

    score += signals["recruiter_response_rate"] * 30

    score += signals["interview_completion_rate"] * 20

    score += min(signals["saved_by_recruiters_30d"], 20)

    score += min(signals["github_activity_score"], 100) * 0.2

    return score


candidates = []

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        candidate = json.loads(line)

        title = candidate["profile"]["current_title"]

        summary = candidate["profile"]["summary"]

        headline = candidate["profile"]["headline"]

        text = headline + " " + summary

        score = 0

        score += title_score(title)

        score += retrieval_score(text)

        score += behavior_score(
            candidate["redrob_signals"]
        )

        score += candidate["profile"]["years_of_experience"]

        candidates.append(
            (
                score,
                candidate["candidate_id"],
                title
            )
        )

candidates.sort(reverse=True)

print("\nTOP 20\n")

for score, cid, title in candidates[:20]:

    print(
        round(score, 2),
        cid,
        title
    )