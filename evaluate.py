"""
evaluate.py — run the fixed evaluation question set through the RAG system.

These are the 5 test questions from planning.md (plus they double as the
Evaluation Report in README.md). Question 5 is intentionally OUT OF CORPUS —
it checks that the grounding guardrail refuses instead of hallucinating.

Run:  python evaluate.py        (loads the index + Groq once, runs all questions)
Needs: a built index (python build_index.py) and GROQ_API_KEY in .env.
"""

from app import answer

QUESTIONS = [
    "Does the 2010 Subaru Outback have head gasket problems, and around what mileage?",
    "Do owners report the Toyota Camry consuming or burning oil, and on which engine or years?",
    "What common problems do owners report on the Honda Civic, and are there model years to avoid?",
    "What do owners say about the reliability of the Subaru CVT transmission?",
    "How does the Honda Civic compare to a Tesla Model 3 for reliability?",  # out of corpus
]


def main():
    for i, q in enumerate(QUESTIONS, 1):
        reply, sources = answer(q)
        print("=" * 80)
        print(f"Q{i}: {q}")
        print("-" * 80)
        print(reply)
        print("\nSources retrieved:")
        print(sources)
        print()


if __name__ == "__main__":
    main()
