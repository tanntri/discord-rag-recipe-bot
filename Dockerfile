FROM langchain/langgraph-api:3.12

WORKDIR /app

COPY . /app

RUN PYTHONDONTWRITEBYTECODE=1 uv sync

# --- LangGraph API Configuration ---
# Explicitly set the checkpoint store to SQLite to override the default.
ENV LANGGRAPH_CHECKPOINT_STORE="sqlite"

# Set the database URI for the SQLite checkpoint store.
ENV DATABASE_URI="sqlite:///app/checkpoints.db"

# Clear out any potential lingering Redis URI settings from the base image.
# This ensures it doesn't try to connect to Redis as well.
ENV REDIS_URI=

# Ensure the LangServe graph is correctly configured
ENV LANGSERVE_GRAPHS='{"chef": "/app/src/graphs/graphs.py:app"}'

# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing build deps from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm /usr/bin/uv /usr/bin/uvx
