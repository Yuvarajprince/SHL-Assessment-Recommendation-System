def classify_query_intent(query: str):
    query = query.lower()

    technical_keywords = [
        "java", "python", "sql", "developer", "programming",
        "coding", "framework", "engineer", "technical"
    ]

    behavioral_keywords = [
        "collaboration", "communication", "teamwork",
        "personality", "behavior", "leadership", "stakeholder"
    ]

    tech = any(k in query for k in technical_keywords)
    beh = any(k in query for k in behavioral_keywords)

    if tech and beh:
        return "mixed"
    elif tech:
        return "technical"
    elif beh:
        return "behavioral"
    else:
        return "general"


def rerank(results, query: str, final_k: int = 10):
    intent = classify_query_intent(query)

    technical = []
    behavioral = []
    others = []

    for r in results:
        test_type = r.get("test_type", "").upper()

        if "K" in test_type:
            technical.append(r)
        elif "P" in test_type or "B" in test_type:
            behavioral.append(r)
        else:
            others.append(r)

    ranked = []

    if intent == "mixed":
        ranked.extend(technical[:5])
        ranked.extend(behavioral[:5])

    elif intent == "technical":
        ranked.extend(technical[:7])
        ranked.extend(behavioral[:3])

    elif intent == "behavioral":
        ranked.extend(behavioral[:7])
        ranked.extend(technical[:3])

    else:
        ranked.extend(results[:final_k])

    # Fallback if not enough results
    if len(ranked) < final_k:
        remaining = [r for r in results if r not in ranked]
        ranked.extend(remaining[:final_k - len(ranked)])

    return ranked[:final_k]
