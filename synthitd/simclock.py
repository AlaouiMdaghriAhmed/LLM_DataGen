"""Work-calendar helpers (named ``simclock`` to avoid shadowing stdlib ``calendar``)."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass


@dataclass
class WorkCalendar:
    """Maps simulation days to timestamps and classifies work vs off hours."""

    start_date: str
    workday_start_hour: float = 8.0
    workday_end_hour: float = 18.0
    lunch_hours: tuple[float, float] = (12.0, 13.0)

    def __post_init__(self) -> None:
        self._start = _dt.datetime.fromisoformat(self.start_date).replace(
            tzinfo=_dt.timezone.utc
        )

    def day_start_ts(self, day_index: int) -> float:
        return (self._start + _dt.timedelta(days=day_index)).timestamp()

    def ts(self, day_index: int, hour: float) -> float:
        return self.day_start_ts(day_index) + hour * 3600.0

    def is_weekend(self, day_index: int) -> bool:
        d = self._start + _dt.timedelta(days=day_index)
        return d.weekday() >= 5  # 5=Sat, 6=Sun

    def weekday(self, day_index: int) -> int:
        return (self._start + _dt.timedelta(days=day_index)).weekday()

    def date(self, day_index: int) -> _dt.date:
        return (self._start + _dt.timedelta(days=day_index)).date()

    def is_work_hour(self, hour: float) -> bool:
        return self.workday_start_hour <= hour < self.workday_end_hour

    def is_after_hours(self, hour: float) -> bool:
        return not self.is_work_hour(hour)

    @staticmethod
    def hour_of_ts(ts: float) -> float:
        d = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc)
        return d.hour + d.minute / 60.0 + d.second / 3600.0

    @staticmethod
    def day_key(ts: float) -> str:
        return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%Y-%m-%d")
