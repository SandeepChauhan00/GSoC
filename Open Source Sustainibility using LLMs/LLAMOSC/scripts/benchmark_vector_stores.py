"""
Benchmark script to compare vector store backends.
Tests FAISS vs Chroma with different data sizes.
"""

import time
import random
from typing import List, Dict
from LLAMOSC.simulation.rag_retriever import RAGRetriever


def generate_sample_messages(count: int) -> List[str]:
    """Generate sample conversation messages for testing."""
    templates = [
        "I think we should implement {} for the {} module",
        "The {} feature is causing issues with {}",
        "Can someone review the PR for {}?",
        "I fixed the bug in {} related to {}",
        "We need to discuss the {} architecture",
    ]
    words = [
        "authentication", "database", "API", "frontend", "backend",
        "caching", "logging", "testing", "deployment", "security",
    ]

    messages = []
    for _ in range(count):
        template = random.choice(templates)
        w1, w2 = random.sample(words, 2)
        num_placeholders = template.count("{}")
        if num_placeholders == 2:
            messages.append(template.format(w1, w2))
        else:
            messages.append(template.format(w1))
    return messages


def benchmark_backend(backend: str, messages: List[str], queries: List[str]) -> Dict:
    """Benchmark a single backend."""
    results = {
        "backend": backend,
        "doc_count": len(messages),
        "index_time": 0,
        "query_times": [],
        "avg_query_time": 0,
    }

    retriever = RAGRetriever(backend=backend, k=3)

    start = time.time()
    retriever.index_documents(messages)
    results["index_time"] = time.time() - start

    for query in queries:
        start = time.time()
        retriever.retrieve(query)
        results["query_times"].append(time.time() - start)

    results["avg_query_time"] = sum(results["query_times"]) / len(results["query_times"])

    retriever.clear()
    return results


def run_benchmarks():
    """Run benchmarks for all backends at different scales."""
    print("=" * 60)
    print("Vector Store Benchmark")
    print("=" * 60)

    test_queries = [
        "How do I fix the authentication bug?",
        "What's the status of the API PR?",
        "Who is working on the frontend?",
    ]

    backends = ["faiss", "chroma"]
    data_sizes = [100, 500, 1000]

    for size in data_sizes:
        print(f"\n--- Testing with {size} messages ---\n")
        messages = generate_sample_messages(size)

        for backend in backends:
            try:
                results = benchmark_backend(backend, messages, test_queries)
                print(f"{backend.upper()}:")
                print(f"  Index time: {results['index_time']:.3f}s")
                print(f"  Avg query:  {results['avg_query_time']:.3f}s")
            except Exception as e:
                print(f"{backend.upper()}: Error - {e}")
        print()

    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmarks()