"""Keep model download credentials separate from dataset upload credentials."""

import httpx

READ_TOKEN_KEY = "HUGGINGFACE_READ_API"
UPLOAD_TOKEN_KEY = "HUGGINGFACE_API"
TOKEN_SETTINGS_URL = "https://huggingface.co/settings/tokens"
WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
READ_PERMISSIONS = {"repo.read", "repo.content.read"}


class DownloadTokenError(ValueError):
    """A safe-to-display error that never contains token or account details."""


def read_only_identity(identity):
    """Accept Read tokens or explicitly recognized read-only fine-grained scopes."""
    try:
        access = identity["auth"]["accessToken"]
        role = access["role"]
        if role == "read":
            return not access.get("fineGrained")
        if role != "fineGrained":
            return False
        scope = access["fineGrained"]
        if not isinstance(scope, dict) or set(scope) - {"scoped", "global", "canReadGatedRepos"}:
            return False
        if scope.get("global", []) != []:
            return False  # Global discussion/post grants allow writes.
        if type(scope.get("canReadGatedRepos", False)) is not bool:
            return False
        if not isinstance(scope["scoped"], list):
            return False
        for item in scope["scoped"]:
            permissions = item["permissions"]
            if not isinstance(permissions, list) or any(
                not isinstance(permission, str) or permission not in READ_PERMISSIONS
                for permission in permissions
            ):
                return False
        return True
    except (KeyError, TypeError, AttributeError):
        return False


def validate_download_token(token, *, upload_token="", transport=None):
    if not isinstance(token, str) or not token or any(
        char.isspace() or ord(char) < 32 or ord(char) == 127 for char in token
    ):
        raise DownloadTokenError("The model-download token must be a single nonempty token.")
    if token == upload_token:
        raise DownloadTokenError(
            "Use a separate read-only model-download token. The dataset-upload token "
            "cannot be reused for model downloads."
        )
    try:
        with httpx.Client(timeout=15, follow_redirects=False, transport=transport) as client:
            response = client.get(WHOAMI_URL, headers={"Authorization": "Bearer " + token})
            response.raise_for_status()
            identity = response.json()
    except (httpx.HTTPError, ValueError, OSError):
        raise DownloadTokenError(
            "Could not verify the Hugging Face model-download token. Check the token "
            "and network connection, then retry through /setup. No download token was sent to Lambda."
        ) from None
    if not read_only_identity(identity):
        raise DownloadTokenError(
            "The model-download token must be read-only. Use a Read token or a "
            "fine-grained token with only repository read permissions. Write-capable "
            "or unrecognized permissions are not accepted."
        )


def model_download_token(settings):
    """False explicitly disables Hub cached/environment-token fallback."""
    token = settings.secrets.get(READ_TOKEN_KEY, "")
    if not token:
        return False
    validate_download_token(token, upload_token=settings.secrets.get(UPLOAD_TOKEN_KEY, ""))
    return token
