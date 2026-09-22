from fastapi.responses import FileResponse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .models import (
    ChatRequest,
    ChatResponse,
    DocumentUploadResponse,
    AnalyzeTransactionsRequest,
    AnalyzeTransactionsResponse,
    SignalResponse,
    SignalTypeResponse,
    MarketTypeResponse,
    NewsItemResponse,
    WatchlistItemResponse,
    AddToWatchlistRequest,
    MarketSummaryResponse,
    AIReportRequest,
    AIReportResponse,
)
from .finance_tools import summarize_transactions
from rag.ingest import ingest_document
from rag.retriever import get_relevant_context

from signal_engine.database import (
    get_signals,
    get_recent_news,
    get_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    get_prices,
)
from signal_engine.models import WatchlistItem, MarketType

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT_PATH = Path("prompts/system_prompt.md")
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8") if SYSTEM_PROMPT_PATH.exists() else ""

app = FastAPI(
    title="Finance AI Assistant & Quantitative Trading API",
    description="Self-hosted finance AI powered by LLaMA 3.1 & 24/7 technical market signals",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=FileResponse)
async def serve_dashboard():
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            resp.raise_for_status()
        return {"status": "ok", "ollama": "connected", "model": LLM_MODEL}
    except Exception as e:
        return {"status": "degraded", "ollama_error": str(e), "model": LLM_MODEL}


@app.get("/models")
async def list_models():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama not reachable: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    sources = []
    if request.use_rag:
        try:
            rag_context, sources = get_relevant_context(request.message)
            if rag_context:
                messages.append({
                    "role": "system",
                    "content": f"Relevant context from user documents:\n\n{rag_context}",
                })
        except Exception as e:
            logger.warning(f"RAG lookup skipped: {e}")

    try:
        recent_signals = get_signals(limit=10)
        if recent_signals:
            sig_lines = [
                f"- {s.symbol} ({s.market.value}): {s.signal_type.value} (Confidence {s.confidence}%, Price ${s.current_price:.2f})"
                for s in recent_signals
            ]
            messages.append({
                "role": "system",
                "content": "Current active market indicator signals (for reference):\n" + "\n".join(sig_lines),
            })
    except Exception as e:
        logger.debug(f"Signal injection skipped: {e}")

    for msg in request.conversation_history:
        messages.append({"role": msg.role.value, "content": msg.content})

    messages.append({"role": "user", "content": request.message})

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": LLM_MODEL, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
            reply = data["message"]["content"]
            return ChatResponse(reply=reply, sources=sources, model=LLM_MODEL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")


@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".csv", ".xlsx", ".xls", ".docx", ".txt"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {file_ext}")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks_count = ingest_document(str(save_path))
        return DocumentUploadResponse(
            filename=file.filename,
            chunks_indexed=chunks_count,
            status="indexed",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document indexing failed: {e}")


@app.post("/analyze-transactions", response_model=AnalyzeTransactionsResponse)
async def analyze_transactions_endpoint(request: AnalyzeTransactionsRequest):
    summary_data = summarize_transactions(request.transactions)

    summary_prompt = (
        f"Analyze these transactions:\n"
        f"- Income: ${summary_data['total_income']}\n"
        f"- Expenses: ${summary_data['total_expenses']}\n"
        f"- Net: ${summary_data['net']}\n"
        f"- Top categories: {summary_data['top_categories'][:5]}\n"
        f"Provide a 3-sentence practical financial observation."
    )

    ai_summary = "Transactions processed successfully."
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": LLM_MODEL, "prompt": summary_prompt, "stream": False},
            )
            if resp.is_success:
                ai_summary = resp.json().get("response", ai_summary)
    except Exception as e:
        logger.warning(f"AI narrative generation skipped: {e}")

    return AnalyzeTransactionsResponse(
        categorized=summary_data["categorized_transactions"],
        summary=ai_summary,
        total_income=summary_data["total_income"],
        total_expenses=summary_data["total_expenses"],
        top_categories=summary_data["top_categories"],
    )


@app.get("/market/summary", response_model=MarketSummaryResponse)
async def market_summary():
    try:
        signals = get_signals(limit=100, latest_only=True)
        news = get_recent_news(hours=12, limit=50)

        counts = {"STRONG_BUY": 0, "BUY": 0, "HOLD": 0, "SELL": 0, "STRONG_SELL": 0}
        for s in signals:
            counts[s.signal_type.value] = counts.get(s.signal_type.value, 0) + 1

        sentiment_avg = 0.0
        if news:
            sentiment_avg = sum(n.sentiment_score for n in news) / len(news)

        watchlist_items = get_watchlist()
        name_map = {item.symbol: item.display_name for item in watchlist_items}

        top_sigs = sorted(signals, key=lambda x: x.confidence, reverse=True)[:5]
        top_resps = [
            SignalResponse(
                id=s.id,
                symbol=s.symbol,
                display_name=name_map.get(s.symbol, s.symbol),
                market=MarketTypeResponse(s.market.value),
                timestamp=s.timestamp,
                signal_type=SignalTypeResponse(s.signal_type.value),
                confidence=s.confidence,
                reasons=s.reasons,
                current_price=s.current_price,
                rsi=s.rsi,
                macd=s.macd,
                win_rate=s.win_rate,
            )
            for s in top_sigs
        ]

        return MarketSummaryResponse(
            total_tracked=len(watchlist_items),
            strong_buys=counts["STRONG_BUY"],
            buys=counts["BUY"],
            holds=counts["HOLD"],
            sells=counts["SELL"],
            strong_sells=counts["STRONG_SELL"],
            top_signals=top_resps,
            overall_sentiment=round(sentiment_avg, 3),
            last_updated=datetime.now(timezone.utc),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compile market summary: {e}")


@app.get("/market/signals", response_model=List[SignalResponse])
async def list_signals(
    symbol: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    latest_only: bool = Query(default=True),
):
    try:
        watchlist_items = get_watchlist()
        name_map = {item.symbol: item.display_name for item in watchlist_items}
        signals = get_signals(symbol=symbol, limit=limit, latest_only=latest_only)
        return [
            SignalResponse(
                id=s.id,
                symbol=s.symbol,
                display_name=name_map.get(s.symbol, s.symbol),
                market=MarketTypeResponse(s.market.value),
                timestamp=s.timestamp,
                signal_type=SignalTypeResponse(s.signal_type.value),
                confidence=s.confidence,
                reasons=s.reasons,
                current_price=s.current_price,
                rsi=s.rsi,
                macd=s.macd,
                win_rate=s.win_rate,
            )
            for s in signals
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/market/news", response_model=List[NewsItemResponse])
async def list_news(
    symbol: Optional[str] = None,
    hours: int = Query(default=12, le=72),
    limit: int = Query(default=30, le=100),
):
    try:
        news = get_recent_news(symbol=symbol, hours=hours, limit=limit)
        results = []
        for n in news:
            if n.sentiment_score > 0.05:
                label = "Bullish"
            elif n.sentiment_score < -0.05:
                label = "Bearish"
            else:
                label = "Neutral"

            results.append(
                NewsItemResponse(
                    id=n.id,
                    headline=n.headline,
                    url=n.url,
                    source=n.source,
                    timestamp=n.timestamp,
                    sentiment_score=n.sentiment_score,
                    sentiment_label=label,
                    symbols=n.symbols,
                )
            )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/market/watchlist", response_model=List[WatchlistItemResponse])
async def list_watchlist():
    try:
        items = get_watchlist()
        return [
            WatchlistItemResponse(
                symbol=i.symbol,
                market=MarketTypeResponse(i.market.value),
                display_name=i.display_name,
                enabled=i.enabled,
            )
            for i in items
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.post("/market/watchlist", response_model=WatchlistItemResponse)
async def add_watchlist_item(req: AddToWatchlistRequest):
    try:
        item = WatchlistItem(
            symbol=req.symbol,
            market=MarketType(req.market.value),
            display_name=req.display_name,
            enabled=True,
        )
        add_to_watchlist(item)
        return WatchlistItemResponse(
            symbol=item.symbol,
            market=MarketTypeResponse(item.market.value),
            display_name=item.display_name,
            enabled=item.enabled,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add symbol: {e}")


@app.delete("/market/watchlist/{symbol}")
async def remove_watchlist_item(symbol: str):
    try:
        remove_from_watchlist(symbol)
        return {"status": "removed", "symbol": symbol}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove: {e}")


@app.get("/market/prices/{symbol}")
async def get_price_history(symbol: str, hours: int = Query(default=24, le=168)):
    try:
        df = get_prices(symbol, hours=hours)
        if df.empty:
            return []
        df_reset = df.reset_index()
        records = []
        for _, row in df_reset.iterrows():
            records.append({
                "timestamp": row["timestamp"].isoformat(),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            })
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.post("/market/report", response_model=AIReportResponse)
async def generate_ai_report(req: AIReportRequest):
    try:
        signals = get_signals(limit=25, latest_only=True)
        news = get_recent_news(hours=12, limit=15)

        sig_summary = "\n".join([
            f"- {s.symbol} ({s.market.value}): {s.signal_type.value} @ ${s.current_price:.2f} (Confidence: {s.confidence}%, RSI: {s.rsi or 'N/A'}, Reasons: {', '.join(s.reasons)})"
            for s in signals[:15]
        ])

        news_summary = "\n".join([
            f"- [{n.source}] {n.headline} (Sentiment: {n.sentiment_score:+.2f})"
            for n in news[:10]
        ])

        report_prompt = (
            f"You are an expert quantitative market analyst. Review the live market indicators and news feeds:\n\n"
            f"=== LATEST TECHNICAL SIGNALS ===\n{sig_summary}\n\n"
            f"=== RECENT NEWS HEADLINES & SENTIMENT ===\n{news_summary}\n\n"
            f"Generate a clear, structured market advisory report with:\n"
            f"1. Executive Market Assessment (Risk sentiment & directional bias)\n"
            f"2. Top Opportunities (High conviction BUY/STRONG_BUY setups with reasoning)\n"
            f"3. Key Risks & Overbought Assets (SELL/STRONG_SELL warnings)\n"
            f"4. Sector / Crypto / Macro Highlights\n"
            f"5. Actionable Next Steps & Risk Management rules\n\n"
            f"Always include an explicit disclaimer stating this is technical analysis for informational purposes and not licensed financial advice."
        )

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": LLM_MODEL, "prompt": report_prompt, "stream": False},
            )
            resp.raise_for_status()
            report_text = resp.json().get("response", "Could not generate report.")

        symbols = list(set([s.symbol for s in signals]))
        return AIReportResponse(
            report=report_text,
            generated_at=datetime.now(timezone.utc),
            model=LLM_MODEL,
            symbols_covered=symbols,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {e}")
