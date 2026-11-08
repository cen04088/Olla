import os
from importlib.util import find_spec
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# .env 파일 자동 로드 (있으면)
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass
FRONTEND_DIST_DIR = BASE_DIR.parent / "frontend" / "dist"


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name, default=None):
    value = os.environ.get(name, "")
    if not value:
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


_secret_key = os.environ.get("DJANGO_SECRET_KEY")
if not _secret_key:
    _is_production = bool(os.environ.get("DATABASE_URL") or os.environ.get("RAILWAY_PUBLIC_DOMAIN"))
    if _is_production:
        raise RuntimeError(
            "DJANGO_SECRET_KEY 환경변수가 설정되지 않았습니다. 프로덕션에서는 반드시 설정하세요."
        )
    _secret_key = "dev-only-forestrx-secret-key"
SECRET_KEY = _secret_key
DEBUG = env_bool("DJANGO_DEBUG", default=False)

ALLOWED_HOSTS = env_list("ALLOWED_HOSTS", ["localhost", "127.0.0.1", ".railway.app", ".up.railway.app"])
if railway_domain := os.environ.get("RAILWAY_PUBLIC_DOMAIN"):
    ALLOWED_HOSTS.append(railway_domain)

CSRF_TRUSTED_ORIGINS = env_list("CSRF_TRUSTED_ORIGINS")
if railway_domain:
    CSRF_TRUSTED_ORIGINS.append(f"https://{railway_domain}")

CORS_ALLOWED_ORIGINS = env_list("CORS_ALLOWED_ORIGINS", ["http://127.0.0.1:5173", "http://localhost:5173"])
HAS_WHITENOISE = find_spec("whitenoise") is not None

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "recommendations",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "recommendations.middleware.DevCorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
if HAS_WHITENOISE:
    MIDDLEWARE.insert(1, "whitenoise.middleware.WhiteNoiseMiddleware")

ROOT_URLCONF = "forestrx.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [FRONTEND_DIST_DIR],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "forestrx.wsgi.application"

_DATABASE_URL = os.environ.get("DATABASE_URL")
if _DATABASE_URL:
    import dj_database_url
    DATABASES = {
        "default": dj_database_url.config(
            default=_DATABASE_URL,
            conn_max_age=600,
            conn_health_checks=True,
        )
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

# 캐시: 기본 LocMemCache는 gunicorn 워커마다 따로라 날씨·LLM 브리핑을 워커 수만큼 중복 호출한다.
# 배포 환경에서는 같은 컨테이너의 워커들이 공유하는 파일 캐시를 쓴다. (DJANGO_CACHE_DIR로 경로 지정 가능)
_CACHE_DIR = os.environ.get("DJANGO_CACHE_DIR", "").strip()
if not _CACHE_DIR and (_DATABASE_URL or os.environ.get("RAILWAY_PUBLIC_DOMAIN")):
    import tempfile
    _CACHE_DIR = os.path.join(tempfile.gettempdir(), "forestrx-cache")
if _CACHE_DIR:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.filebased.FileBasedCache",
            "LOCATION": _CACHE_DIR,
            "OPTIONS": {"MAX_ENTRIES": 5000},
        }
    }

LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

# 정적 파일 설정
# STATIC_URL: Django admin 등 내부 정적 파일 경로
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# frontend/dist/assets 만 Django staticfiles에 등록
# (WHITENOISE_ROOT가 /assets/*.js를 직접 처리하므로 중복 불필요)
_FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"
STATICFILES_DIRS = [_FRONTEND_ASSETS_DIR] if _FRONTEND_ASSETS_DIR.exists() else []

if HAS_WHITENOISE:
    STORAGES = {
        "staticfiles": {
            # CompressedStaticFilesStorage: 파일명 유지 + gzip/brotli 압축만
            # (ManifestStaticFilesStorage는 파일명에 해시를 추가해 index.html과 불일치 발생)
            "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage",
        },
    }
    # whitenoise가 frontend/dist/ 전체를 루트로 직접 서빙
    # → /assets/GuideTab-xxx.js, /assets/index-xxx.css 등 모든 Vite 빌드 파일 처리
    WHITENOISE_ROOT = str(FRONTEND_DIST_DIR)
    WHITENOISE_INDEX_FILE = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    "loggers": {
        "recommendations": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
