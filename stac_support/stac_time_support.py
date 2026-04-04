from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

def parse_datetime_utc(value: str) -> datetime:
    """Parse RFC3339 datetime ve timezone UTC."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)

def normalize_datetime_range(datetime_range: Optional[str]) -> Optional[str]:
    """Chuan hoa datetime range ve dang RFC3339."""
    if not datetime_range:
        return None
    if "/" not in datetime_range:
        return datetime_range

    parts = datetime_range.split("/")
    formatted_parts: List[str] = []
    for idx, part in enumerate(parts):
        p = part.strip()
        if p == ".." or "T" in p:
            formatted_parts.append(p)
        else:
            suffix = "T00:00:00Z" if idx == 0 else "T23:59:59Z"
            formatted_parts.append(f"{p}{suffix}")
    return "/".join(formatted_parts)

def midpoint_datetime(dt1: datetime, dt2: datetime) -> datetime:
    """Diem giua giua hai moc thoi gian UTC."""
    return dt1 + (dt2 - dt1) / 2

def parse_finite_datetime_range(datetime_range: Optional[str]) -> Tuple[datetime, datetime]:
    """Parse finite RFC3339 range and return UTC bounds."""
    normalized = normalize_datetime_range(datetime_range)
    if not normalized or "/" not in normalized:
        raise ValueError("Representative calendar selection requires a finite datetime range `start/end`.")
    start_raw, end_raw = [part.strip() for part in normalized.split("/", 1)]
    if start_raw == ".." or end_raw == "..":
        raise ValueError("Representative calendar selection does not support open datetime ranges (`..`).")
    start_dt = parse_datetime_utc(start_raw)
    end_dt = parse_datetime_utc(end_raw)
    if start_dt == datetime.min.replace(tzinfo=timezone.utc) or end_dt == datetime.min.replace(tzinfo=timezone.utc):
        raise ValueError(f"Invalid datetime range: {datetime_range}")
    if start_dt >= end_dt:
        raise ValueError(f"Datetime range must satisfy start < end, got: {datetime_range}")
    return start_dt, end_dt

def floor_month_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

def add_month_utc(dt: datetime) -> datetime:
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1)
    return dt.replace(month=dt.month + 1)

def expand_month_periods(datetime_range: Optional[str], allow_partial_periods: bool = False) -> List[Dict[str, Any]]:
    """Expand a finite datetime range into calendar-month periods."""
    start_dt, end_dt = parse_finite_datetime_range(datetime_range)
    inclusive_full_end = end_dt
    if end_dt.hour == 23 and end_dt.minute == 59 and end_dt.second == 59 and end_dt.microsecond == 0:
        inclusive_full_end = end_dt + timedelta(seconds=1)
    periods: List[Dict[str, Any]] = []
    cursor = floor_month_utc(start_dt)
    while cursor < inclusive_full_end:
        next_month = add_month_utc(cursor)
        if allow_partial_periods:
            period_start = max(cursor, start_dt)
            period_end = min(next_month, inclusive_full_end)
            is_full = period_start == cursor and period_end == next_month
            if period_start < period_end:
                period_anchor = midpoint_datetime(period_start, period_end)
                periods.append(
                    {
                        "period_id": period_start.strftime("%Y-%m"),
                        "period_mode": "month",
                        "period_start": period_start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "period_end": period_end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "period_anchor_datetime": period_anchor.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "is_full_period": is_full,
                    }
                )
        else:
            if start_dt <= cursor and next_month <= inclusive_full_end:
                period_anchor = midpoint_datetime(cursor, next_month)
                periods.append(
                    {
                        "period_id": cursor.strftime("%Y-%m"),
                        "period_mode": "month",
                        "period_start": cursor.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "period_end": next_month.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "period_anchor_datetime": period_anchor.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "is_full_period": True,
                    }
                )
        cursor = next_month
    return periods
