#!/usr/bin/env bash
FILE=1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz
FILENAME=CASIA-WebFace.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILE" -O FILENAME && rm -rf /tmp/cookies.txt

#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz" -O CASIA-WebFace.zip && rm -rf /tmp/cookies.txt

unzip FILENAME

rm -rf FILENAME