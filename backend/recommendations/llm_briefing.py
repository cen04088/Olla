import os

_LLM_CACHE_SECONDS = 3600  # 동일 조건 브리핑은 1시간 캐시


def _cache_key(course, safety, weather, daylight_margin):
    rainfall_bucket = int((weather.get("rainfall_mm") or 0) // 5) * 5
    wind_bucket = int((weather.get("wind_speed_ms") or 0) // 3) * 3
    daylight_bucket = (
        None if daylight_margin is None else
        "negative" if daylight_margin < 0 else
        "short" if daylight_margin < 60 else
        "ok"
    )
    key_parts = (
        # 브리핑 문장에 코스 이름이 들어가므로 코스별로 캐시해야 다른 코스 이름이 섞여 나오지 않는다
        course.get("id") or course.get("name", ""),
        safety.get("safety_decision", ""),
        course.get("difficulty", ""),
        rainfall_bucket,
        wind_bucket,
        daylight_bucket,
        ",".join(sorted((safety.get("risk_factors") or [])[:2])),
    )
    return "llm_briefing:" + ":".join(str(p) for p in key_parts)


def generate_briefing(course, safety, weather, daylight_margin=None) -> str | None:
    """Claude API로 브리핑 생성. API 키 없거나 오류 시 None 반환 (호출부가 템플릿 폴백 사용)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None

    from django.core.cache import cache
    cache_key = _cache_key(course, safety, weather, daylight_margin)
    cached = cache.get(cache_key)
    if cached:
        return cached

    try:
        import anthropic

        risk_text = ", ".join((safety.get("risk_factors") or [])[:2]) or "특이 위험 없음"
        daylight_text = (
            f"일몰 전 {daylight_margin}분 여유" if (daylight_margin or 0) >= 0
            else "일몰 전 하산 불가"
        ) if daylight_margin is not None else "일몰 정보 없음"

        prompt = (
            f"산행 안전 진단 결과를 2문장으로 요약하세요. 구체적이고 행동 중심으로 작성하세요.\n\n"
            f"코스: {course.get('name', '')} ({course.get('mountain', '')})\n"
            f"안전 등급: {safety.get('safety_label', '')}\n"
            f"위험 요인: {risk_text}\n"
            f"날씨: 강수 {weather.get('rainfall_mm', 0)}mm, 풍속 {weather.get('wind_speed_ms', 0)}m/s\n"
            f"난이도: {course.get('difficulty', '')}, {daylight_text}\n\n"
            "규칙: 2문장 이하, 어린이·노약자 동반자 관점, 마지막 문장은 행동 권고로 끝내세요."
        )

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=160,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        cache.set(cache_key, text, _LLM_CACHE_SECONDS)
        return text
    except Exception:
        return None
