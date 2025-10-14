FROM nvcr.io/nvidia/pytorch:21.11-py3

ARG USERNAME=maui63
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Persist bash history: https://code.visualstudio.com/remote/advancedcontainers/persist-bash-history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory \
    && chown $USERNAME /commandhistory/.bash_history \
    && echo "$SNIPPET" >> "/home/$USERNAME/.bashrc"

RUN apt-get update -y && apt-get upgrade -y

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

RUN conda install -y conda
RUN conda install -y seaborn shapely pybboxes


USER $USERNAME

# Stuff for yolov9
RUN pip install --upgrade pip
# RUN pip install seaborn shapely pybboxes

# Fix opencv import
RUN pip uninstall opencv-python && pip install opencv-python-headless

# imports
RUN pip install loguru exif arrow
RUN pip install --upgrade pymavlink

# Fix pillow version for yolov9
RUN pip install Pillow==9.5.0

WORKDIR /home/$USERNAME