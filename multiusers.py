"""멀티유저/멀티세션 RAG 챗봇 — user 테이블 로그인 + Supabase 세션·벡터 저장."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import bcrypt
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

load_dotenv(dotenv_path=ENV_PATH)

EMBEDDING_DIM = 1536
EMBED_BATCH_SIZE = 10
RAG_MATCH_COUNT = 10
USER_TABLE = "user"


def _get_config(key: str) -> str:
  """Prefer Streamlit secrets, then environment variables."""
  try:
    if hasattr(st, "secrets") and key in st.secrets:
      value = st.secrets[key]
      if value:
        return str(value).strip()
  except Exception:
    pass
  return os.getenv(key, "").strip()


def _sync_config_to_env() -> None:
  for key in ("OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY"):
    val = _get_config(key)
    if val:
      os.environ[key] = val


_sync_config_to_env()


# ---------------------------------------------------------------------------
# Logging (Streamlit Cloud는 앱 디렉터리 쓰기 불가 → 임시 폴더 또는 콘솔만)
# ---------------------------------------------------------------------------
def _writable_log_dir() -> Path | None:
  candidates = [
    Path(__file__).resolve().parent / "logs",
    LOG_DIR,
    Path(tempfile.gettempdir()) / "multiusers_logs",
  ]
  for directory in candidates:
    try:
      directory.mkdir(parents=True, exist_ok=True)
      probe = directory / ".write_probe"
      probe.write_text("", encoding="utf-8")
      probe.unlink(missing_ok=True)
      return directory
    except (OSError, PermissionError):
      continue
  return None


def _setup_logging() -> logging.Logger:
  root = logging.getLogger()
  root.handlers.clear()
  root.setLevel(logging.WARNING)

  fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

  log_dir = _writable_log_dir()
  if log_dir is not None:
    log_path = log_dir / f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(fmt)
    root.addHandler(fh)

  ch = logging.StreamHandler()
  ch.setLevel(logging.WARNING)
  ch.setFormatter(fmt)
  root.addHandler(ch)

  for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
    logging.getLogger(name).setLevel(logging.WARNING)

  return logging.getLogger("multiusers")


logger = _setup_logging()

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def remove_separators(text: str) -> str:
  out = re.sub(r"~~([^~]*)~~", r"\1", text)
  out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
  out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
  out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
  out = re.sub(r"\n{3,}", "\n\n", out)
  return out.strip()


def _env_status() -> dict[str, bool]:
  return {
    "OPENAI_API_KEY": bool(_get_config("OPENAI_API_KEY")),
    "SUPABASE_URL": bool(_get_config("SUPABASE_URL")),
    "SUPABASE_ANON_KEY": bool(_get_config("SUPABASE_ANON_KEY")),
  }


def _missing_keys_message() -> str | None:
  missing = [k for k, ok in _env_status().items() if not ok]
  if not missing:
    return None
  hint = (
    "Streamlit Cloud에서는 **Secrets**에, 로컬에서는 `.env`에 키를 설정해 주세요.\n\n"
    f"로컬 `.env` 경로: `{ENV_PATH}`\n\n"
    "Supabase SQL Editor에서 `multi-user-ref.sql`을 먼저 실행했는지 확인해 주세요."
  )
  return f"# 환경 변수 안내\n\n다음 키가 없습니다: **{', '.join(missing)}**\n\n{hint}"


def get_supabase() -> Client | None:
  url = _get_config("SUPABASE_URL")
  key = _get_config("SUPABASE_ANON_KEY")
  if not url or not key:
    return None
  return create_client(url, key)


def get_llm() -> ChatOpenAI | None:
  api_key = _get_config("OPENAI_API_KEY")
  if not api_key:
    return None
  return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)


def _hash_password(password: str) -> str:
  return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
  try:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
  except ValueError:
    return False


def register_user(sb: Client, login_id: str, password: str) -> tuple[bool, str]:
  login_id = login_id.strip()
  if not login_id or not password:
    return False, "아이디와 비밀번호를 입력해 주세요."
  if len(password) < 4:
    return False, "비밀번호는 4자 이상이어야 합니다."
  try:
    existing = (
      sb.table(USER_TABLE).select("id").eq("login_id", login_id).limit(1).execute()
    )
    if existing.data:
      return False, "이미 사용 중인 아이디입니다."
    row = sb.table(USER_TABLE).insert(
      {"login_id": login_id, "password_hash": _hash_password(password)}
    ).execute()
    if not row.data:
      return False, "회원가입에 실패했습니다."
    return True, "회원가입이 완료되었습니다. 로그인해 주세요."
  except Exception as exc:  # noqa: BLE001
    logger.warning("Register failed: %s", exc)
    return False, f"회원가입 중 오류가 발생했습니다: {exc}"


def login_user(sb: Client, login_id: str, password: str) -> tuple[bool, str, str | None]:
  login_id = login_id.strip()
  if not login_id or not password:
    return False, "아이디와 비밀번호를 입력해 주세요.", None
  try:
    resp = (
      sb.table(USER_TABLE)
      .select("id, password_hash")
      .eq("login_id", login_id)
      .limit(1)
      .execute()
    )
    rows = resp.data or []
    if not rows:
      return False, "아이디 또는 비밀번호가 올바르지 않습니다.", None
    row = rows[0]
    if not _verify_password(password, row["password_hash"]):
      return False, "아이디 또는 비밀번호가 올바르지 않습니다.", None
    return True, f"{login_id}님, 환영합니다.", row["id"]
  except Exception as exc:  # noqa: BLE001
    logger.warning("Login failed: %s", exc)
    return False, f"로그인 중 오류가 발생했습니다: {exc}", None


def _current_user_id() -> str | None:
  return st.session_state.get("user_id")


def _session_owned(sb: Client, session_id: str, user_id: str) -> bool:
  try:
    resp = (
      sb.table("chat_sessions")
      .select("id")
      .eq("id", session_id)
      .eq("user_id", user_id)
      .limit(1)
      .execute()
    )
    return bool(resp.data)
  except Exception as exc:  # noqa: BLE001
    logger.warning("Session ownership check failed: %s", exc)
    return False


def _format_memory_block(messages: list[dict[str, str]], max_items: int = 50) -> str:
  tail = messages[-max_items:] if len(messages) > max_items else messages
  lines: list[str] = []
  for m in tail:
    role = m.get("role", "")
    content = (m.get("content") or "").strip()
    if not content:
      continue
    prefix = "사용자" if role == "user" else "어시스턴트"
    lines.append(f"{prefix}: {content}")
  return "\n".join(lines)


def _build_rag_messages(
  question: str,
  context: str,
  memory_text: str,
) -> list[SystemMessage | HumanMessage]:
  sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
  return [SystemMessage(content=sys), HumanMessage(content=question)]


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
  trimmed = answer[:8000]
  prompt = (
    "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
    "형식:\n1. ...\n2. ...\n3. ...\n"
    "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
    f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
  )
  try:
    out = llm.invoke([HumanMessage(content=prompt)])
    raw = remove_separators(str(getattr(out, "content", out) or ""))
  except Exception as exc:  # noqa: BLE001
    logger.warning("Follow-up generation failed: %s", exc)
    return ""
  if not raw.strip():
    return ""
  return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def _generate_session_title(llm: ChatOpenAI, first_q: str, first_a: str) -> str:
  prompt = (
    "다음 첫 질문과 답변을 바탕으로 대화 세션 제목을 한국어 15자 이내로 한 줄만 작성하세요.\n"
    "따옴표, 설명, 번호 없이 제목만 출력하세요.\n\n"
    f"[질문]\n{first_q[:500]}\n\n[답변]\n{first_a[:800]}"
  )
  try:
    out = llm.invoke([HumanMessage(content=prompt)])
    title = str(getattr(out, "content", out) or "").strip().strip("\"'")
    return title[:80] if title else "새 대화"
  except Exception as exc:  # noqa: BLE001
    logger.warning("Title generation failed: %s", exc)
    return (first_q[:30] + "…") if len(first_q) > 30 else (first_q or "새 대화")


def fetch_session_list(sb: Client, user_id: str) -> list[dict[str, Any]]:
  try:
    resp = (
      sb.table("chat_sessions")
      .select("id, title, updated_at")
      .eq("user_id", user_id)
      .order("updated_at", desc=True)
      .execute()
    )
    return resp.data or []
  except Exception as exc:  # noqa: BLE001
    logger.warning("Session list fetch failed: %s", exc)
    return []


def _ensure_session_row(sb: Client, session_id: str, title: str, user_id: str) -> bool:
  try:
    existing = (
      sb.table("chat_sessions")
      .select("id")
      .eq("id", session_id)
      .eq("user_id", user_id)
      .limit(1)
      .execute()
    )
    if existing.data:
      sb.table("chat_sessions").update({"title": title}).eq("id", session_id).eq(
        "user_id", user_id
      ).execute()
    else:
      sb.table("chat_sessions").insert(
        {"id": session_id, "title": title, "user_id": user_id}
      ).execute()
    return True
  except Exception as exc:  # noqa: BLE001
    logger.warning("Ensure session row failed: %s", exc)
    return False


def save_messages(
  sb: Client, session_id: str, messages: list[dict[str, str]], user_id: str
) -> bool:
  if not _session_owned(sb, session_id, user_id):
    return False
  try:
    sb.table("chat_messages").delete().eq("session_id", session_id).eq(
      "user_id", user_id
    ).execute()
    rows = [
      {
        "session_id": session_id,
        "user_id": user_id,
        "role": m["role"],
        "content": m["content"],
        "message_order": idx,
      }
      for idx, m in enumerate(messages)
    ]
    if rows:
      sb.table("chat_messages").insert(rows).execute()
    sb.table("chat_sessions").update({"updated_at": datetime.utcnow().isoformat()}).eq(
      "id", session_id
    ).eq("user_id", user_id).execute()
    return True
  except Exception as exc:  # noqa: BLE001
    logger.warning("Save messages failed: %s", exc)
    return False


def load_session_from_db(
  sb: Client, session_id: str, user_id: str
) -> tuple[list[dict[str, str]], str]:
  title = "불러온 세션"
  if not _session_owned(sb, session_id, user_id):
    return [], title
  try:
    sresp = (
      sb.table("chat_sessions")
      .select("title")
      .eq("id", session_id)
      .eq("user_id", user_id)
      .limit(1)
      .execute()
    )
    if sresp.data:
      title = sresp.data[0].get("title") or title

    mresp = (
      sb.table("chat_messages")
      .select("role, content, message_order")
      .eq("session_id", session_id)
      .eq("user_id", user_id)
      .order("message_order")
      .execute()
    )
    messages = [{"role": r["role"], "content": r["content"]} for r in (mresp.data or [])]
    return messages, title
  except Exception as exc:  # noqa: BLE001
    logger.warning("Load session failed: %s", exc)
    return [], title


def delete_session_from_db(sb: Client, session_id: str, user_id: str) -> bool:
  if not _session_owned(sb, session_id, user_id):
    return False
  try:
    sb.table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()
    return True
  except Exception as exc:  # noqa: BLE001
    logger.warning("Delete session failed: %s", exc)
    return False


def fetch_vector_file_names(sb: Client, session_id: str, user_id: str) -> list[str]:
  if not _session_owned(sb, session_id, user_id):
    return []
  try:
    resp = (
      sb.table("vector_documents")
      .select("file_name")
      .eq("session_id", session_id)
      .execute()
    )
    return sorted({r["file_name"] for r in (resp.data or []) if r.get("file_name")})
  except Exception as exc:  # noqa: BLE001
    logger.warning("Vector file list failed: %s", exc)
    return []


def _embed_texts(embeddings: OpenAIEmbeddings, texts: list[str]) -> list[list[float]]:
  return embeddings.embed_documents(texts)


def store_vectors_for_session(
  sb: Client,
  session_id: str,
  file_name: str,
  chunks: list[str],
  metadatas: list[dict[str, Any]],
  embeddings: OpenAIEmbeddings,
  user_id: str,
) -> int:
  if not chunks or not _session_owned(sb, session_id, user_id):
    return 0

  stored = 0
  for i in range(0, len(chunks), EMBED_BATCH_SIZE):
    batch_texts = chunks[i : i + EMBED_BATCH_SIZE]
    batch_meta = metadatas[i : i + EMBED_BATCH_SIZE]
    vectors = _embed_texts(embeddings, batch_texts)
    rows = []
    for text, meta, vec in zip(batch_texts, batch_meta, vectors):
      if len(vec) != EMBEDDING_DIM:
        logger.warning("Unexpected embedding dim %s for %s", len(vec), file_name)
      rows.append(
        {
          "session_id": session_id,
          "file_name": file_name,
          "content": text,
          "embedding": vec,
          "metadata": meta,
        }
      )
    sb.table("vector_documents").insert(rows).execute()
    stored += len(rows)
  return stored


def _process_pdfs_to_supabase(
  sb: Client,
  session_id: str,
  uploaded_files: list[Any],
  openai_key: str,
  user_id: str,
) -> tuple[list[str], int]:
  embeddings = OpenAIEmbeddings(api_key=openai_key)
  splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  processed_names: list[str] = []
  total_chunks = 0

  for uf in uploaded_files:
    fname = uf.name or "unknown.pdf"
    suffix = Path(fname).suffix.lower() or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
      tmp.write(uf.getvalue())
      tmp_path = tmp.name
    try:
      loader = PyPDFLoader(tmp_path)
      docs = loader.load()
    finally:
      try:
        os.unlink(tmp_path)
      except OSError:
        pass

    if not docs:
      continue

    for doc in docs:
      doc.metadata = dict(doc.metadata or {})
      doc.metadata["file_name"] = fname

    splits = splitter.split_documents(docs)
    chunks: list[str] = []
    metas: list[dict[str, Any]] = []
    for doc in splits:
      chunks.append(doc.page_content)
      meta = dict(doc.metadata or {})
      meta["file_name"] = fname
      metas.append(meta)

    count = store_vectors_for_session(
      sb, session_id, fname, chunks, metas, embeddings, user_id
    )
    total_chunks += count
    processed_names.append(fname)

  return processed_names, total_chunks


def match_documents_rpc(
  sb: Client,
  session_id: str,
  query: str,
  openai_key: str,
  user_id: str,
  k: int = RAG_MATCH_COUNT,
) -> list[Document]:
  if not _session_owned(sb, session_id, user_id):
    return []

  embeddings = OpenAIEmbeddings(api_key=openai_key)
  query_vec = embeddings.embed_query(query)

  try:
    resp = sb.rpc(
      "match_vector_documents",
      {
        "query_embedding": query_vec,
        "match_count": k,
        "filter_session_id": session_id,
      },
    ).execute()
    rows = resp.data or []
    return [
      Document(
        page_content=r.get("content", ""),
        metadata={
          "file_name": r.get("file_name", ""),
          **(r.get("metadata") or {}),
        },
      )
      for r in rows
      if r.get("content")
    ]
  except Exception as exc:  # noqa: BLE001
    logger.warning("RPC match_vector_documents failed: %s", exc)
    return _match_documents_fallback(sb, session_id, query, query_vec, k, user_id)


def _match_documents_fallback(
  sb: Client,
  session_id: str,
  query: str,
  query_vec: list[float],
  k: int,
  user_id: str,
) -> list[Document]:
  if not _session_owned(sb, session_id, user_id):
    return []
  try:
    resp = (
      sb.table("vector_documents")
      .select("content, file_name, metadata, embedding")
      .eq("session_id", session_id)
      .execute()
    )
    rows = resp.data or []
  except Exception as exc:  # noqa: BLE001
    logger.warning("Fallback vector fetch failed: %s", exc)
    return []

  def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
      return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
      return -1.0
    return dot / (na * nb)

  scored: list[tuple[float, dict[str, Any]]] = []
  for row in rows:
    emb = row.get("embedding")
    if isinstance(emb, str):
      try:
        emb = json.loads(emb)
      except json.JSONDecodeError:
        continue
    if not isinstance(emb, list):
      continue
    scored.append((cosine(query_vec, emb), row))

  scored.sort(key=lambda x: x[0], reverse=True)
  docs: list[Document] = []
  for _, row in scored[:k]:
    docs.append(
      Document(
        page_content=row.get("content", ""),
        metadata={
          "file_name": row.get("file_name", ""),
          **(row.get("metadata") or {}),
        },
      )
    )
  return docs


def persist_session(sb: Client, llm: ChatOpenAI, user_id: str) -> tuple[bool, str]:
  messages = st.session_state.chat_history
  sid = st.session_state.session_id
  has_vectors = bool(fetch_vector_file_names(sb, sid, user_id))
  if not messages and not has_vectors:
    return False, "저장할 대화 또는 PDF 벡터가 없습니다."

  first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
  first_asst = next((m["content"] for m in messages if m["role"] == "assistant"), "")

  if st.session_state.session_persisted and st.session_state.session_title:
    title = st.session_state.session_title
  elif first_asst:
    title = _generate_session_title(llm, first_user, first_asst)
  elif has_vectors:
    title = st.session_state.session_title or "PDF 문서 세션"
  else:
    title = st.session_state.session_title or "새 대화"

  if not _ensure_session_row(sb, sid, title, user_id):
    return False, "세션 헤더 저장에 실패했습니다."

  if not save_messages(sb, sid, messages, user_id):
    return False, "메시지 저장에 실패했습니다."

  st.session_state.session_title = title
  st.session_state.session_persisted = True
  st.session_state.session_options = fetch_session_list(sb, user_id)
  return True, f"세션이 저장되었습니다: {title}"


def apply_loaded_session(session_id: str, messages: list[dict[str, str]], title: str) -> None:
  st.session_state.session_id = session_id
  st.session_state.session_title = title
  st.session_state.chat_history = messages
  st.session_state.conversation_memory = messages[-50:]
  st.session_state.session_persisted = True
  st.session_state.processed_names = []


def reset_ui_session(*, new_id: bool = True) -> None:
  if new_id:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_persisted = False
    st.session_state.session_title = "새 대화"
  st.session_state.chat_history = []
  st.session_state.conversation_memory = []
  st.session_state.processed_names = []


def logout_user() -> None:
  for key in (
    "user_id",
    "login_id",
    "chat_history",
    "conversation_memory",
    "processed_names",
    "session_id",
    "session_title",
    "session_persisted",
    "session_options",
    "selected_session_id",
    "sidebar_action",
    "session_picker",
  ):
    if key in st.session_state:
      del st.session_state[key]


def _init_session() -> None:
  defaults: dict[str, Any] = {
    "chat_history": [],
    "conversation_memory": [],
    "processed_names": [],
    "session_id": str(uuid.uuid4()),
    "session_title": "새 대화",
    "session_persisted": False,
    "session_options": [],
    "selected_session_id": None,
    "sidebar_action": None,
  }
  for k, v in defaults.items():
    if k not in st.session_state:
      st.session_state[k] = v


def _render_header() -> None:
  st.markdown(
    """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""",
    unsafe_allow_html=True,
  )

  c1, c2, c3 = st.columns([1, 4, 1])
  with c1:
    if LOGO_PATH.is_file():
      st.image(str(LOGO_PATH), width=180)
    else:
      st.markdown("### 📚")
  with c2:
    st.markdown(
      """
<h1 style="text-align:center; margin:0; font-size:4rem !important;">
  <span style="color:#1f77b4;">재정경제부</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
""",
      unsafe_allow_html=True,
    )
  with c3:
    st.empty()


def _render_auth_panel(sb: Client) -> bool:
  """Return True when user is logged in."""
  if _current_user_id():
    return True

  st.info("로그인 후 챗봇과 세션 저장 기능을 사용할 수 있습니다.")
  tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

  with tab_login:
    login_id = st.text_input("아이디", key="auth_login_id")
    password = st.text_input("비밀번호", type="password", key="auth_login_pw")
    if st.button("로그인", key="btn_login"):
      ok, msg, uid = login_user(sb, login_id, password)
      if ok and uid:
        st.session_state.user_id = uid
        st.session_state.login_id = login_id.strip()
        reset_ui_session(new_id=True)
        st.session_state.session_options = fetch_session_list(sb, uid)
        st.success(msg)
        st.rerun()
      else:
        st.error(msg)

  with tab_signup:
    new_id = st.text_input("새 아이디", key="auth_signup_id")
    new_pw = st.text_input("새 비밀번호", type="password", key="auth_signup_pw")
    new_pw2 = st.text_input("비밀번호 확인", type="password", key="auth_signup_pw2")
    if st.button("회원가입", key="btn_signup"):
      if new_pw != new_pw2:
        st.error("비밀번호 확인이 일치하지 않습니다.")
      else:
        ok, msg = register_user(sb, new_id, new_pw)
        st.success(msg) if ok else st.error(msg)

  return False


def _render_sidebar(sb: Client, llm: ChatOpenAI, user_id: str) -> None:
  login_id = st.session_state.get("login_id", "")

  def on_session_select() -> None:
    sid = st.session_state.get("session_picker")
    if sid:
      messages, title = load_session_from_db(sb, sid, user_id)
      apply_loaded_session(sid, messages, title)
      st.session_state.processed_names = fetch_vector_file_names(sb, sid, user_id)
      st.session_state.selected_session_id = sid

  with st.sidebar:
    st.markdown(f"**로그인:** {login_id}")
    if st.button("로그아웃"):
      logout_user()
      st.rerun()

    st.markdown("**세션 관리**")
    options = st.session_state.session_options
    id_to_title = {r["id"]: r.get("title", "제목 없음") for r in options}
    picker_ids = [""] + list(id_to_title.keys()) if id_to_title else [""]

    current_pick = st.session_state.get("selected_session_id") or ""
    if current_pick and current_pick not in picker_ids:
      picker_ids.append(current_pick)

    st.selectbox(
      "세션 선택",
      options=picker_ids,
      format_func=lambda x: "— 세션 선택 —"
      if not x
      else id_to_title.get(x, st.session_state.session_title),
      key="session_picker",
      index=picker_ids.index(current_pick) if current_pick in picker_ids else 0,
      on_change=on_session_select,
    )

    col1, col2 = st.columns(2)
    with col1:
      if st.button("세션저장"):
        st.session_state.sidebar_action = "save_insert"
    with col2:
      if st.button("세션로드"):
        st.session_state.sidebar_action = "load"

    col3, col4 = st.columns(2)
    with col3:
      if st.button("세션삭제"):
        st.session_state.sidebar_action = "delete"
    with col4:
      if st.button("화면초기화"):
        st.session_state.sidebar_action = "clear"

    if st.button("vectordb"):
      st.session_state.sidebar_action = "vectordb"

    action = st.session_state.pop("sidebar_action", None)
    if action == "save_insert":
      ok, msg = persist_session(sb, llm, user_id)
      st.success(msg) if ok else st.error(msg)
    elif action == "load":
      sid = st.session_state.get("session_picker") or st.session_state.session_id
      if not sid:
        st.warning("로드할 세션을 선택해 주세요.")
      else:
        if not _session_owned(sb, sid, user_id):
          st.error("해당 세션을 불러올 수 없습니다.")
        else:
          messages, title = load_session_from_db(sb, sid, user_id)
          apply_loaded_session(sid, messages, title)
          st.session_state.processed_names = fetch_vector_file_names(sb, sid, user_id)
          st.session_state.selected_session_id = sid
          st.success(f"세션을 불러왔습니다: {title}")
    elif action == "delete":
      sid = st.session_state.get("session_picker") or st.session_state.session_id
      if not sid:
        st.warning("삭제할 세션을 선택해 주세요.")
      elif delete_session_from_db(sb, sid, user_id):
        st.session_state.session_options = fetch_session_list(sb, user_id)
        reset_ui_session(new_id=True)
        st.session_state.selected_session_id = None
        st.session_state.session_picker = ""
        st.success("세션이 삭제되었습니다.")
      else:
        st.error("세션 삭제에 실패했습니다.")
    elif action == "clear":
      reset_ui_session(new_id=True)
      st.session_state.selected_session_id = None
      st.session_state.session_picker = ""
      st.success("화면이 초기화되었습니다. (DB의 다른 세션은 유지됩니다)")
    elif action == "vectordb":
      names = fetch_vector_file_names(sb, st.session_state.session_id, user_id)
      if names:
        st.markdown("**현재 세션 vectordb 파일**")
        for n in names:
          st.text(f"- {n}")
      else:
        st.text("vectordb에 저장된 파일이 없습니다.")

    st.markdown("**RAG (PDF)**")
    uploads = st.file_uploader(
      "PDF 파일 업로드",
      type=["pdf"],
      accept_multiple_files=True,
    )
    if st.button("파일 처리하기"):
      if not uploads:
        st.warning("업로드된 PDF가 없습니다.")
      else:
        try:
          if not st.session_state.session_persisted:
            _ensure_session_row(
              sb,
              st.session_state.session_id,
              st.session_state.session_title,
              user_id,
            )
          names, chunk_count = _process_pdfs_to_supabase(
            sb,
            st.session_state.session_id,
            list(uploads),
            _get_config("OPENAI_API_KEY"),
            user_id,
          )
          st.session_state.processed_names = list(
            dict.fromkeys(st.session_state.processed_names + names)
          )
          st.session_state.session_persisted = True
          st.session_state.session_options = fetch_session_list(sb, user_id)
          ok, msg = persist_session(sb, llm, user_id)
          st.success(
            f"PDF {len(names)}개, 청크 {chunk_count}개 저장. "
            f"{msg if ok else '자동 세션 저장 실패'}"
          )
        except Exception as exc:  # noqa: BLE001
          logger.warning("PDF 처리 실패: %s", exc)
          st.error(f"PDF 처리 중 오류: {exc}")

    if st.session_state.processed_names:
      st.markdown("**처리된 파일**")
      for name in st.session_state.processed_names:
        st.text(f"- {name}")

    mem_count = len(st.session_state.conversation_memory)
    file_count = len(st.session_state.processed_names)
    settings_text = (
      f"모델: gpt-4o-mini\n"
      f"사용자: {login_id}\n"
      f"현재 세션 ID: {st.session_state.session_id[:8]}…\n"
      f"세션 제목: {st.session_state.session_title}\n"
      f"DB 저장 여부: {'예' if st.session_state.session_persisted else '아니오'}\n"
      f"처리된 PDF 수: {file_count}\n"
      f"대화 메시지 수: {mem_count}"
    )
    st.text(settings_text)


def main() -> None:
  st.set_page_config(
    page_title="재정경제부 RAG 챗봇",
    page_icon="📚",
    layout="wide",
  )
  _init_session()
  _render_header()

  missing_msg = _missing_keys_message()
  if missing_msg:
    st.markdown(missing_msg)
    return

  sb = get_supabase()
  if sb is None:
    st.markdown("# 환경 변수 안내\n\nSupabase 클라이언트를 초기화할 수 없습니다.")
    return

  if not _render_auth_panel(sb):
    return

  llm = get_llm()
  if llm is None:
    st.markdown("# 환경 변수 안내\n\nOpenAI API 키가 없어 LLM을 사용할 수 없습니다.")
    return

  user_id = _current_user_id()
  if not user_id:
    return

  if not st.session_state.session_options:
    st.session_state.session_options = fetch_session_list(sb, user_id)

  _render_sidebar(sb, llm, user_id)

  for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
      st.markdown(remove_separators(msg["content"]))

  user_input = st.chat_input("질문을 입력하세요")
  if not user_input:
    return

  st.session_state.chat_history.append({"role": "user", "content": user_input})
  st.session_state.conversation_memory.append({"role": "user", "content": user_input})
  if len(st.session_state.conversation_memory) > 50:
    st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

  with st.chat_message("user"):
    st.markdown(remove_separators(user_input))

  with st.chat_message("assistant"):
    placeholder = st.empty()
    full_answer = ""

    try:
      vector_names = fetch_vector_file_names(
        sb, st.session_state.session_id, user_id
      )
      use_rag = bool(vector_names)

      if use_rag:
        mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
        docs = match_documents_rpc(
          sb,
          st.session_state.session_id,
          user_input,
          _get_config("OPENAI_API_KEY"),
          user_id,
        )
        if not docs:
          full_answer = (
            "# 안내\n\n"
            "관련 문서를 찾지 못했습니다. PDF를 업로드한 뒤 **파일 처리하기**를 실행해 주세요."
          )
          placeholder.markdown(remove_separators(full_answer))
        else:
          context = "\n\n".join(d.page_content for d in docs)
          messages = _build_rag_messages(user_input, context, mem_txt)
          acc = ""
          for chunk in llm.stream(messages):
            piece = getattr(chunk, "content", "") or ""
            if piece:
              acc += piece
              placeholder.markdown(remove_separators(acc) + "▌")
          full_answer = remove_separators(acc)
          placeholder.markdown(full_answer)
      else:
        mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
        sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
        msgs = [SystemMessage(content=sys), HumanMessage(content=user_input)]
        acc = ""
        for chunk in llm.stream(msgs):
          piece = getattr(chunk, "content", "") or ""
          if piece:
            acc += piece
            placeholder.markdown(remove_separators(acc) + "▌")
        full_answer = remove_separators(acc)
        placeholder.markdown(full_answer)

      if full_answer and not full_answer.lstrip().startswith("# 오류"):
        follow = _generate_followup_section(llm, user_input, full_answer)
        if follow:
          full_answer += follow
          placeholder.markdown(remove_separators(full_answer))

    except Exception as exc:  # noqa: BLE001
      logger.warning("답변 생성 실패: %s", exc)
      full_answer = f"# 오류\n\n요청 처리 중 문제가 발생했습니다.\n\n`{exc}`"
      placeholder.markdown(remove_separators(full_answer))

    st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
    st.session_state.conversation_memory.append(
      {"role": "assistant", "content": full_answer}
    )
    if len(st.session_state.conversation_memory) > 50:
      st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

    ok, _ = persist_session(sb, llm, user_id)
    if ok:
      st.session_state.session_options = fetch_session_list(sb, user_id)


if __name__ == "__main__":
  main()
