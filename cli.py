"""
cli.py — Unified command-line interface for the Semantic Search system.

Commands:
    index       — Ingest and embed documents from a directory
    search      — Run a semantic query (Phase 3)
    duplicates  — Detect near-duplicate documents (Phase 4)
    cluster     — Cluster documents by topic (Phase 5)

Examples:
    python cli.py index ./data/
    python cli.py search "climate change policy" --topk 5
    python cli.py duplicates --threshold 0.9
    python cli.py cluster --k 5
"""

import argparse
import sys


def cmd_index(args):
    from indexer import index_directory
    index_directory(args.directory)


def cmd_search(args):
    try:
        from search import search
        search(args.query, args.topk)
    except NotImplementedError:
        print("[Search] Not yet implemented. Coming in Phase 3.")


def cmd_duplicates(args):
    try:
        from duplicates import find_duplicates
        find_duplicates(threshold=args.threshold)
    except NotImplementedError:
        print("[Duplicates] Not yet implemented. Coming in Phase 4.")


def cmd_cluster(args):
    try:
        from clustering import cluster_documents
        cluster_documents(n_clusters=args.k)
    except NotImplementedError:
        print("[Cluster] Not yet implemented. Coming in Phase 5.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Semantic Document Search & Clustering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ── index ──────────────────────────────────────────────────
    p_index = subparsers.add_parser(
        "index",
        help="Index all documents in a directory",
        description="Extract text, generate embeddings, and store in SQLite.",
    )
    p_index.add_argument(
        "directory",
        type=str,
        help="Path to folder containing PDF/DOCX/TXT files",
    )
    p_index.set_defaults(func=cmd_index)

    # ── search ─────────────────────────────────────────────────
    p_search = subparsers.add_parser(
        "search",
        help="Semantic search over indexed documents",
        description="Returns top-k documents most similar to the query.",
    )
    p_search.add_argument("query", type=str, help="Natural language search query")
    p_search.add_argument(
        "--topk", type=int, default=5, metavar="K",
        help="Number of results to return (default: 5)",
    )
    p_search.set_defaults(func=cmd_search)

    # ── duplicates ─────────────────────────────────────────────
    p_dupes = subparsers.add_parser(
        "duplicates",
        help="Detect exact and near-duplicate documents",
        description="Uses SHA-256 hashing and embedding similarity.",
    )
    p_dupes.add_argument(
        "--threshold", type=float, default=0.9, metavar="T",
        help="Cosine similarity threshold for near-duplicates (default: 0.9)",
    )
    p_dupes.set_defaults(func=cmd_duplicates)

    # ── cluster ────────────────────────────────────────────────
    p_cluster = subparsers.add_parser(
        "cluster",
        help="Cluster documents by topic",
        description="Runs K-Means and Agglomerative clustering, reports silhouette scores.",
    )
    p_cluster.add_argument(
        "--k", type=int, default=5, metavar="K",
        help="Number of clusters (default: 5)",
    )
    p_cluster.set_defaults(func=cmd_cluster)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        raise


if __name__ == "__main__":
    main()