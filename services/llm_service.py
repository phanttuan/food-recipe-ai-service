import json
import logging
import re
import time
import unicodedata
from difflib import get_close_matches
import requests
from config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger("ai-service.llm")


def build_recipe_context(transcript: str, description: str, max_chars: int = 3200) -> str:
    """Keep the most useful cooking instructions within a fixed prompt budget.

    YouTube transcripts often start with greetings and ingredient introductions;
    blindly taking the first characters drops the actual cooking method.  This
    function is deterministic and makes no extra model call.
    """
    transcript = re.sub(r"\s+", " ", transcript or "").strip()
    description = re.sub(r"\s+", " ", description or "").strip()

    description_budget = min(550, max_chars // 4) if description else 0
    transcript_budget = max_chars - description_budget
    parts = []
    if description:
        parts.append(f"MÔ TẢ VIDEO: {description[:description_budget]}")

    if not transcript:
        return "\n".join(parts)

    chunks = [item.strip() for item in re.split(r"(?<=[.!?…])\s+", transcript) if item.strip()]
    if len(chunks) < 3:
        chunks = [transcript[index:index + 220] for index in range(0, len(transcript), 220)]
    else:
        chunks = [
            chunk[index:index + 260]
            for chunk in chunks
            for index in range(0, len(chunk), 260)
        ]

    action_words = (
        "rửa", "cắt", "thái", "băm", "ướp", "trộn", "pha", "nêm", "cho", "đun",
        "nấu", "luộc", "hấp", "chiên", "xào", "kho", "rim", "quay", "vớt", "tắt bếp",
        "lửa", "phút", "sôi", "chín", "vàng", "mềm", "giòn", "nước mắm",
    )
    ranked = sorted(
        range(len(chunks)),
        key=lambda index: (
            sum(word in chunks[index].lower() for word in action_words),
            len(chunks[index]),
        ),
        reverse=True,
    )

    # Preserve a little opening context, then choose action-heavy portions plus
    # their preceding neighbor (quantities/objects are often stated just before
    # the action verb), restoring original order before sending to the model.
    selected = {0}
    used = len(chunks[0])
    for index in ranked:
        if index in selected:
            continue
        neighbor = index - 1
        extra = len(chunks[index]) + 1
        include_neighbor = neighbor >= 0 and neighbor not in selected
        if include_neighbor:
            extra += len(chunks[neighbor]) + 1
        if used + extra > transcript_budget:
            continue
        if include_neighbor:
            selected.add(neighbor)
            used += len(chunks[neighbor]) + 1
        selected.add(index)
        used += len(chunks[index]) + 1

    compact_transcript = " ".join(chunks[index] for index in sorted(selected))
    parts.append(f"TRANSCRIPT: {compact_transcript}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Grounding / normalization helpers (no extra LLM calls)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def _appears_in_source(item: str, source_norm: str) -> bool:
    """Check that a model-claimed ingredient/word is actually backed by the
    source text, allowing a 1-word slack for morphological variants."""
    item_norm = _normalize(item)
    words = [w for w in item_norm.split() if len(w) > 1]
    if not words:
        return True
    hits = sum(1 for w in words if w in source_norm)
    return hits >= max(1, len(words) - 1)


def _instruction_grounding_ratio(instruction: str, source_norm: str) -> float:
    """Return the fraction of content words in *instruction* that appear in *source_norm*.

    Short common Vietnamese words (≤2 chars) are skipped since they are
    function words (cho, và, với, rồi …) that don't indicate hallucination.
    """
    words = [w for w in _normalize(instruction).split() if len(w) > 2]
    if not words:
        return 1.0
    hits = sum(1 for w in words if w in source_norm)
    return hits / len(words)


def verify_step_grounding(step: dict, source_text: str) -> dict:
    """Drop main/with items the model invented but the source never mentions.
    Also remove instruction if it is mostly ungrounded so render_step will
    compose a safe template-based fallback from structured fields."""
    if not isinstance(step, dict):
        return step
    source_norm = _normalize(source_text)
    for key in ("main", "with"):
        items = step.get(key) or []
        if isinstance(items, list):
            kept = [it for it in items if _appears_in_source(str(it), source_norm)]
            dropped = [it for it in items if it not in kept]
            if dropped:
                logger.info(f"Dropped ungrounded {key} items: {dropped}")
            step[key] = kept

    # Check instruction grounding — drop if <40% of content words are in source
    instruction = step.get("instruction") or ""
    if instruction:
        ratio = _instruction_grounding_ratio(instruction, source_norm)
        if ratio < 0.4:
            logger.warning(
                f"Dropped hallucinated instruction (grounding {ratio:.0%}): "
                f"{instruction[:80]}..."
            )
            del step["instruction"]

    return step


def reconcile_names(recipe: dict) -> dict:
    """Snap step ingredient mentions to the closest name already declared in
    `ingredients`, so the UI can link step text back to the ingredient list."""
    if not isinstance(recipe, dict):
        return recipe
    ing_names = [ing.get("name", "") for ing in recipe.get("ingredients", []) if isinstance(ing, dict)]
    if not ing_names:
        return recipe
    for step in recipe.get("steps", []):
        if not isinstance(step, dict):
            continue
        for key in ("main", "with"):
            items = step.get(key) or []
            if not isinstance(items, list):
                continue
            step[key] = [
                (get_close_matches(item, ing_names, n=1, cutoff=0.6) or [item])[0]
                for item in items
            ]
    return recipe


def has_cook_phase(steps: list) -> bool:
    """Sanity flag: if a recipe only has prep steps, extraction likely missed
    the actual cooking part of the video (worth flagging for review)."""
    return any(isinstance(s, dict) and s.get("phase") == "cook" for s in steps)


# ---------------------------------------------------------------------------
# Step rendering: LLM emits short fields, code composes the display sentence
# ---------------------------------------------------------------------------

def _clean_parts(value) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip(" ,.") for item in values if str(item).strip(" ,.")]


def _join_parts(parts: list[str]) -> str:
    return ", ".join(parts)


def render_step(step: dict, index: int) -> dict:
    """Render step into instruction field, prioritizing LLM's detailed instruction.
    Never auto-concatenate ingredient lists into fake generic template sentences."""
    if isinstance(step, str):
        return {"stepNumber": index, "instruction": step.strip()}

    if not isinstance(step, dict):
        return {"stepNumber": index, "instruction": f"Bước {index}."}

    instruction = str(step.get("instruction") or "").strip()
    if instruction:
        return {
            "stepNumber": step.get("stepNumber") or step.get("no") or index,
            "phase": step.get("phase", ""),
            "instruction": instruction,
        }

    # Rich fallback if model omitted the instruction field
    action = str(step.get("action") or "").strip()
    main = _join_parts(_clean_parts(step.get("main")))
    with_items = _join_parts(_clean_parts(step.get("with")))
    time_val = str(step.get("time") or "").strip()
    signal = str(step.get("signal") or "").strip()

    parts = []
    if action:
        parts.append(action.capitalize())
    if main:
        parts.append(main)
    if with_items:
        parts.append(f"cùng {with_items}")
    if time_val:
        parts.append(f"trong {time_val}")
    if signal:
        parts.append(f"cho đến khi {signal}")

    if parts:
        fallback_text = " ".join(parts) + "."
        if len(parts) == 1 and action:
            fallback_text = f"{action.capitalize()} các nguyên liệu theo hướng dẫn trong video."
    else:
        fallback_text = f"Thực hiện bước {index} theo hướng dẫn trong video."

    return {
        "stepNumber": step.get("stepNumber") or step.get("no") or index,
        "phase": step.get("phase", ""),
        "instruction": fallback_text,
    }


def render_recipe_steps(recipe: dict) -> dict:
    if not isinstance(recipe, dict):
        return recipe
    raw_steps = recipe.get("steps")
    if isinstance(raw_steps, list):
        recipe["steps"] = [render_step(step, index + 1) for index, step in enumerate(raw_steps)]
    return recipe


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

groq_client = None
if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq AI Client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Groq Client: {e}")

genai_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    try:
        from google import genai
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.debug(f"google-genai SDK init note: {e}")


def call_llm(prompt: str) -> str:
    """
    Main LLM entrypoint:
    1. Try Groq (Llama 3.1 8B Instant - 14.4K RPD / 500K TPD Free).
    2. Fallback to Gemini API if Groq fails or key is missing.
    """
    if groq_client:
        groq_models = [GROQ_MODEL, "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        groq_models = list(dict.fromkeys(groq_models))

        for model in groq_models:
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "Bạn là một đầu bếp chuyên nghiệp Việt Nam. Nhiệm vụ duy nhất của bạn là trích xuất công thức nấu ăn từ nội dung được cung cấp. CHỈ sử dụng thông tin có trong nội dung gốc. TUYỆT ĐỐI KHÔNG bịa đặt hay thêm bất kỳ thông tin nào không có trong nội dung. Luôn trả về JSON hợp lệ."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=model,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                content = chat_completion.choices[0].message.content
                if content:
                    logger.info(f"Groq ({model}) OK")
                    return content.strip()
            except Exception as e:
                logger.warning(f"Groq error '{model}': {e}")

    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        models_to_try = [GEMINI_MODEL, "gemini-2.0-flash-lite", "gemini-1.5-flash-8b"]
        models_to_try = list(dict.fromkeys(models_to_try))

        for model in models_to_try:
            if "pro" in model.lower():
                continue
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"}
            }
            try:
                res = requests.post(url, json=payload, timeout=25)
                if res.status_code == 200:
                    res_data = res.json()
                    candidates = res_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                elif res.status_code == 429:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Gemini error '{model}': {e}")

    raise Exception("No LLM provider available. Set GROQ_API_KEY in .env")


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

FEWSHOT_EXAMPLE = """Ví dụ (1 video mẫu → JSON đúng):
VIDEO: "...băm nhuyễn tỏi, ướp thịt bò với tỏi băm, nước mắm, tiêu trong 15 phút cho thấm rồi bắc chảo phi thơm..."
→ steps: [
  {"no":1,"phase":"prep","action":"băm","main":["tỏi"],"with":[],"time":"","signal":"","instruction":"Băm nhuyễn tỏi."},
  {"no":2,"phase":"prep","action":"ướp","main":["thịt bò"],"with":["tỏi băm","nước mắm","tiêu"],"time":"15 phút","signal":"thấm","instruction":"Ướp thịt bò với tỏi băm, nước mắm và tiêu trong 15 phút cho thấm."},
  {"no":3,"phase":"cook","action":"xào","main":["tỏi"],"with":[],"time":"","signal":"thơm","instruction":"Bắc chảo lên, phi tỏi cho đến khi thơm."}
]
Lưu ý quan trọng:
- Bước 3 KHÔNG lặp lại "thịt bò" dù cùng món, vì video chỉ nói "phi thơm" với tỏi ở thời điểm đó.
- instruction chỉ diễn giải lại nội dung video, KHÔNG thêm chi tiết nào video không nói (ví dụ video không nói "dao bản rộng", "dầu ăn nóng già", "tiêu xay" thì KHÔNG được viết vào)."""


def batch_extract_recipes(videos: list, total_context_budget: int = 6400) -> list:
    """
    Extract recipes from multiple videos in a SINGLE LLM call.
    Each video has: title, description, transcript.
    Returns a list of structured recipe JSON objects.
    """
    if not videos:
        return []

    per_video_budget = min(3200, max(1200, total_context_budget // len(videos)))
    video_blocks = []
    video_sources = []  # keep raw content per video for post-hoc grounding check
    for i, v in enumerate(videos):
        transcript = (v.get("transcript") or "").strip()
        description = (v.get("description") or "").strip()

        content = build_recipe_context(transcript, description, max_chars=per_video_budget)
        if not content:
            content = v.get("title", "Video")
        video_sources.append(content)

        video_blocks.append(
            f"=== VIDEO {i+1}: {v.get('title', '')} ===\n{content}\n=== KẾT THÚC VIDEO {i+1} ==="
        )

    all_videos_text = "\n\n".join(video_blocks)
    num_videos = len(videos)

    prompt = f"""{FEWSHOT_EXAMPLE}

Trích xuất công thức từ {num_videos} video dưới đây. Trả về đúng {num_videos} object theo đúng thứ tự VIDEO.

Quy tắc CHUNG:
- Chỉ dùng dữ kiện có trong VIDEO. Thiếu định lượng, thời gian hoặc dấu hiệu thì không bịa ra con số.
- Tạo 5-10 bước theo đúng trình tự hướng dẫn thực tế trong VIDEO. Không trả lời ngoài JSON.

Quy tắc BẮT BUỘC viết "instruction" cho MỖI BƯỚC:
- MỖI BƯỚC trong "steps" BẮT BUỘC phải có trường "instruction" (câu tiếng Việt chi tiết, rõ ràng). TUYỆT ĐỐI KHÔNG để trống hay chỉ ghi 1-2 từ chung chung.
- Viết câu hướng dẫn CHI TIẾT, ĐẦY ĐỦ, tự nhiên theo từng bước nấu ăn của video (gồm cách làm cụ thể, thứ tự, lửa lớn/nhỏ, thời gian đun, dấu hiệu nhận biết khi nào hoàn thành).
- NGUYÊN TẮC: Chỉ viết những gì video thực sự hướng dẫn, KHÔNG bịa thêm nguyên liệu không có trong video.
- Giữ trọn vẹn sự chi tiết mà người hướng dẫn nói trong video (ví dụ: "Băm nhỏ 2 tép tỏi, phi thơm trên chảo dầu nóng khoảng 1 phút đến khi ngả vàng", "Ướp thịt bò với 1 thìa nước mắm và tiêu trong 15 phút").

{all_videos_text}

Trả về CHÍNH XÁC JSON:
```json
{{
  "recipes": [
    {{
      "title": "Tên món ăn",
      "description": "Tóm tắt ngắn hấp dẫn (2-3 câu giới thiệu món, đặc trưng hương vị, phù hợp dịp nào)",
      "ingredients": [
        {{ "name": "Tên nguyên liệu", "amount": 500, "unit": "g", "category": "thịt/rau/gia vị/khác" }}
      ],
      "steps": [
        {{
          "no": 1,
          "phase": "prep",
          "action": "ướp",
          "main": ["thịt bò"],
          "with": ["tỏi băm", "nước mắm"],
          "time": "10 phút",
          "signal": "thịt thấm gia vị",
          "instruction": "(Chỉ viết những gì video nói — ví dụ: Ướp thịt bò với tỏi băm và nước mắm trong 10 phút cho thấm.)"
        }}
      ],
      "tags": ["tag1"]
    }}
  ]
}}
```
"""

    try:
        raw = call_llm(prompt)
        data = json.loads(raw)
        recipes = data.get("recipes", [])
        if not isinstance(recipes, list):
            recipes = [data] if "title" in data else []

        processed = []
        for i, recipe in enumerate(recipes):
            if not isinstance(recipe, dict):
                processed.append(recipe)
                continue
            source_text = video_sources[i] if i < len(video_sources) else ""
            raw_steps = recipe.get("steps")
            if isinstance(raw_steps, list):
                recipe["steps"] = [verify_step_grounding(s, source_text) for s in raw_steps]
                if not has_cook_phase(recipe["steps"]):
                    logger.warning(f"Recipe '{recipe.get('title', '')}' has no cook-phase step — check source video")
            recipe = reconcile_names(recipe)
            recipe = render_recipe_steps(recipe)
            processed.append(recipe)

        return processed

    except Exception as e:
        logger.error(f"batch_extract_recipes error: {e}")
        return []


def extract_recipe(text_content: str, schema: dict = None) -> dict:
    """Single video extract (backward compatible). Delegates to batch."""
    results = batch_extract_recipes([{"title": "", "description": "", "transcript": text_content}])
    if results and len(results) > 0:
        return {"data": results[0]}
    return {"data": {"title": "", "description": "", "ingredients": [], "steps": [], "tags": []}}