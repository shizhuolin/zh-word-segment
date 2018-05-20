#!/bin/bash

which wget >/dev/null 2>&1
WGET=$?
which curl >/dev/null 2>&1
CURL=$?
if [ "$WGET" -eq 0 ]; then
    DL_CMD="wget -c"
elif [ "$CURL" -eq 0 ]; then
    DL_CMD="curl -C - -O"
else
    echo "You need wget or curl installed to download"
    exit 1
fi

if test ! -e "icwb2-data.zip"
then
  $DL_CMD "http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip"
fi

unzip -u icwb2-data.zip