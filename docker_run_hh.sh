docker rm -f hh_app
nvidia-docker run --rm --name hh_app -ti -p 8000:8000 -v /cloud/data/hh_credit_card_app:/srv/hh_credit_card_app tinmankinetics/tensorflow:latest /bin/bash 