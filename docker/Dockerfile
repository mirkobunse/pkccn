FROM python:3.10.4-bullseye

# build-time variables to be set by the Makefile
ARG group
ARG gid
ARG user
ARG uid
RUN groupadd --gid $gid $group && \
    useradd --create-home --shell /bin/bash --gid $gid --uid $uid $user

# install Julia and LaTeX for plotting
RUN pip install jill && \
    jill install 1.6.5 --confirm && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        texlive \
        texlive-latex-extra \
        texlive-luatex \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER $user

CMD ["/bin/bash"]
