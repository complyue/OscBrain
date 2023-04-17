FROM gitpod/workspace-python:latest

USER gitpod

# or git will keep prompting
RUN git config --global pull.ff only

USER root
