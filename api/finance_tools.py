"""
Finance-specific tools.

Provides helpers for:
- Transaction categorization
- Spending analysis and summaries
- Budget report generation
"""

from typing import List, Dict
from loguru import logger

from .models import Transaction, TransactionCategory


# Keyword-based transaction categorizer
CATEGORY_KEYWORDS: Dict[TransactionCategory, List[str]] = {
    TransactionCategory.housing: [
        "rent", "mortgage", "property", "hoa", "lease",
    ],
    TransactionCategory.food: [
        "restaurant", "cafe", "coffee", "food", "grocery", "supermarket",
        "bakery", "pizza", "burger", "sushi", "dining", "doordash", "ubereats",
    ],
    TransactionCategory.transport: [
        "uber", "lyft", "taxi", "gas", "fuel", "parking", "transit",
        "metro", "bus", "train", "flight", "airline",
    ],
    TransactionCategory.utilities: [
        "electric", "water", "internet", "phone", "utility", "bill",
        "netflix", "spotify", "subscription", "streaming",
    ],
    TransactionCategory.entertainment: [
        "movie", "theater", "concert", "sport", "game", "gym", "fitness",
    ],
    TransactionCategory.healthcare: [
        "pharmacy", "doctor", "hospital", "dental", "medical", "health",
        "insurance",
    ],
    TransactionCategory.income: [
        "salary", "payroll", "deposit", "income", "payment received",
        "transfer in", "direct deposit",
    ],
    TransactionCategory.savings: [
        "savings", "investment", "brokerage", "401k", "ira", "etf",
        "transfer to savings",
    ],
    TransactionCategory.shopping: [
        "amazon", "walmart", "target", "shopping", "clothing", "shoes",
        "electronics", "online purchase",
    ],
}


def categorize_transaction(description: str) -> TransactionCategory:
    """Categorize a transaction based on its description."""
    desc_lower = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return category
    return TransactionCategory.other


def categorize_transactions(transactions: List[Transaction]) -> List[Transaction]:
    """Auto-categorize a list of transactions."""
    for txn in transactions:
        if txn.category is None:
            txn.category = categorize_transaction(txn.description)
    return transactions


def summarize_transactions(transactions: List[Transaction]) -> Dict:
    """Generate a spending summary from a list of transactions."""
    categorized = categorize_transactions(transactions)

    total_income = sum(
        t.amount for t in categorized
        if t.category == TransactionCategory.income or t.amount > 0
    )
    total_expenses = sum(
        t.amount for t in categorized
        if t.category != TransactionCategory.income and t.amount < 0
    )

    # Group by category
    by_category: Dict[str, float] = {}
    for txn in categorized:
        cat = txn.category.value if txn.category else "Other"
        by_category[cat] = by_category.get(cat, 0) + abs(txn.amount)

    # Sort by spend
    top_categories = sorted(
        [{"category": k, "total": round(v, 2)} for k, v in by_category.items()],
        key=lambda x: x["total"],
        reverse=True,
    )

    return {
        "total_income": round(total_income, 2),
        "total_expenses": round(abs(total_expenses), 2),
        "net": round(total_income + total_expenses, 2),
        "top_categories": top_categories,
        "categorized_transactions": categorized,
    }
