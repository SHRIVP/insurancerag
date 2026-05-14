"""
MCP server exposing the P&C insurance underwriting RAG pipeline as a Claude Desktop tool.

Setup:
    1. Add to ~/Library/Application Support/Claude/claude_desktop_config.json (see README)
    2. Restart Claude Desktop
    3. Ask Claude: "What are eligibility criteria for frame construction commercial property?"
"""

import os
import sys

# Ensure the project directory is on the path so rag_query can be imported
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))  # chroma_db path is relative

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="Insurance Underwriting RAG",
    instructions=(
        "Use query_underwriting_guidelines to answer any question about P&C insurance "
        "underwriting — Commercial Property, Commercial General Liability (CGL), or "
        "Personal Auto. Always call this tool before answering underwriting questions "
        "so your answer is grounded in the actual guideline documents."
    ),
)


@mcp.tool()
def query_underwriting_guidelines(question: str, lob: str = "") -> str:
    """
    Query P&C insurance underwriting guidelines and NAICS industry classification
    using a RAG pipeline over publicly available policy forms, state insurance
    department documents, and official NAICS manuals (USA 2022 + Canada 2017).

    Returns a grounded answer with source citations (filename + page number).

    Args:
        question: Natural-language underwriting or classification question.
        lob:      Optional filter to scope retrieval. Must be one of:
                    "Commercial Property"
                    "Commercial General Liability"
                    "Personal Auto"
                    "NAICS Classification"
                  Leave blank to search across all LOBs.
    """
    from rag_query import query, format_response

    lob_filter = lob.strip() or None
    result = query(question, lob_filter=lob_filter)
    return format_response(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
