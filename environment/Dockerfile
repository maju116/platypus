FROM rocker/rstudio:4.0.2

# Basic dependencies
RUN apt-get update && apt-get install -y \
      libcurl4-gnutls-dev \
      libssl-dev \
      libpng-dev \
      apt-utils \
      vim \
      nano \
      libxml2 \
      libxml2-dev \
      curl \
      gnupg2 \
      build-essential libssl-dev \
      libpq-dev \
      ssh \
      libtiff5-dev \
      libfontconfig1-dev \
      libcairo2-dev \
      libssl-dev \
      libsasl2-dev \
      libfftw3-dev

# Make sure bash is the standard shell
SHELL ["/bin/bash", "-c"]

# Set the CRAN mirror
RUN echo "options(repos = c(CRAN = 'https://cran.rstudio.com/'))" >> $(R RHOME)/etc/Rprofile.site

COPY packages.R /tmp/packages.R
RUN R -f /tmp/packages.R

RUN cp /usr/share/zoneinfo/Europe/Warsaw /etc/localtime

COPY user-settings /home/rstudio/.rstudio/monitored/user-settings/
RUN chmod -R 757 /home/rstudio/.rstudio
COPY .Rprofile /home/rstudio/.Rprofile

RUN echo "alias ll='ls -la'" >> /root/.bashrc && \
    echo "alias ls='ls --color=auto'" >> /root/.bashrc

# Then install python3 and pip3
RUN apt-get update && apt-get install -y python3 \
    python3-pip \
    build-essential

RUN pip3 install numpy pandas pydicom pytest scikit-image sklearn tensorflow keras

RUN apt-get install -y libpython3-dev

CMD /init
