FROM python:3.10.5-bullseye as venv-image

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_FOLDER=/app
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade --quiet pip setuptools && \
        pip install -r ./requirements.txt

WORKDIR $APP_FOLDER

COPY . .

RUN pip install .


FROM venv-image as test-image

ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH

COPY requirements-test.txt ./
RUN pip install -r requirements-test.txt


FROM test-image as tool-image

COPY requirements-tool.txt ./
RUN pip install -r requirements-tool.txt

COPY .pre-commit-config.yaml ./
RUN git init && pre-commit install-hooks


FROM python:3.10.5-bullseye as run-image

ARG VERSION
ARG LAST_COMMIT

ENV USER=app
ENV UID=42000
ENV GID=42001
ENV HOME_DIRECTORY=/app
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Add Python bin to PATH
ENV PATH=/opt/venv/bin:$PATH

COPY --from=venv-image /opt/venv/. /opt/venv/

# Create home directory
RUN mkdir -p ${HOME_DIRECTORY}

# Create group and user
RUN groupadd --force --gid $GID $USER && \
        useradd -M --home $HOME_DIRECTORY --base-dir $HOME_DIRECTORY \
        --uid $UID --gid $GID --shell "/bin/bash" $USER

# Set HOME_DIRECTORY as default
WORKDIR $HOME_DIRECTORY

# Copy files into HOME_DIRECTORY
COPY --chown=${USER}:${USER} . .

# Default user
USER $USER

# Set different env variables
ENV VERSION=${VERSION}
ENV LAST_COMMIT=${LAST_COMMIT}

# Append the current directory to your python path
ENV PYTHONPATH=$HOME_DIRECTORY:$PYTHONPATH
