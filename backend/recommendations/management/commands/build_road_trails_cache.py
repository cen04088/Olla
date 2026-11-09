import time

from django.core.management.base import BaseCommand, CommandError

from recommendations.local_road_api import (
    ROAD_CACHE_PATH,
    ROAD_DBF_PATH,
    ROAD_SHP_PATH,
    build_road_cache_rows,
    write_road_cache,
)


class Command(BaseCommand):
    help = (
        "road/WG_MT_WAY 등산로 SHP를 data/road_trails.json.gz 경량 캐시로 변환합니다. "
        "SHP가 바뀌었을 때만 다시 실행하면 됩니다(배포 서버에서는 SHP를 읽지 않음)."
    )

    def handle(self, *args, **options):
        if not ROAD_SHP_PATH.exists() or not ROAD_DBF_PATH.exists():
            raise CommandError(f"SHP 파일이 없습니다: {ROAD_SHP_PATH}")

        started = time.time()
        rows = build_road_cache_rows()
        if not rows:
            raise CommandError("SHP에서 읽은 등산로가 없습니다.")

        path = write_road_cache(rows)
        size_mb = path.stat().st_size / 1_000_000
        self.stdout.write(self.style.SUCCESS(
            f"{len(rows):,}개 등산로 → {path} ({size_mb:.1f}MB, {time.time() - started:.0f}초)"
        ))
