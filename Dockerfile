FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# The :ro mount that used to be the primary control is gone - this container
# writes now. Path containment in safe_resolve() is the control; running as a
# non-root uid is what is left of defence in depth. uid 1000 also matches the
# vault's file ownership, so written notes keep the ownership Samba expects.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 8080

CMD ["python", "-m", "src.server"]
