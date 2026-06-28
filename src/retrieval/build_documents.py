import json

documents = []

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        candidate = json.loads(line)

        profile = candidate["profile"]

        text = f"""
Title: {profile['current_title']}

Headline:
{profile['headline']}

Summary:
{profile['summary']}

Skills:
"""

        for skill in candidate["skills"]:
            text += skill["name"] + ", "

        text += "\n\nCareer History:\n"

        for job in candidate["career_history"]:

            text += f"""
Title: {job['title']}
Company: {job['company']}
Description:
{job['description']}
"""

        documents.append(
            {
                "candidate_id": candidate["candidate_id"],
                "text": text
            }
        )

print("Documents created:", len(documents))

with open(
    "output/candidate_documents.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        documents,
        f,
        indent=2
    )

print("Saved to output/candidate_documents.json")