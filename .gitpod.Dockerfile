FROM condaforge/mambaforge:latest

# or git will keep prompting
RUN git config --global pull.ff only
