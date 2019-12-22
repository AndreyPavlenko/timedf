#!/bin/bash

declare -A vars

OMNISCI_TMP=$(mktemp -d)

vars["OMNISCI_PATH"]=$OMNISCI_PATH
vars["OMNISCI_STORAGE"]=$OMNISCI_STORAGE
vars["OMNISCI_USER"]=$OMNISCI_USER
vars["OMNISCI_GROUP"]=$OMNISCI_GROUP
vars["OMNISCI_PORT"]=$1
vars["OMNISCI_HTTP_PORT"]=$2
vars["OMNISCI_CALCITE_PORT"]=$3

for v in OMNISCI_PATH OMNISCI_STORAGE OMNISCI_USER OMNISCI_GROUP OMNISCI_PORT OMNISCI_HTTP_PORT OMNISCI_CALCITE_PORT; do
  echo -e "$v:\t${vars[$v]}"
done

vars["OMNISCI_DATA"]=${OMNISCI_DATA:="${vars['OMNISCI_STORAGE']}/data"}
sudo mkdir -p "${vars['OMNISCI_DATA']}"
sudo mkdir -p "${vars['OMNISCI_STORAGE']}"

if [ ! -d "${vars['OMNISCI_DATA']}/mapd_catalogs" ]; then
  sudo ${vars["OMNISCI_PATH"]}/bin/initdb ${vars['OMNISCI_DATA']}
fi

sudo chown -R ${vars['OMNISCI_USER']}:${vars['OMNISCI_GROUP']} "${vars['OMNISCI_DATA']}"
sudo chown -R ${vars['OMNISCI_USER']}:${vars['OMNISCI_GROUP']} "${vars['OMNISCI_STORAGE']}"

for f in omnisci_server omnisci_server@ omnisci_sd_server omnisci_sd_server@ omnisci_web_server omnisci_web_server@ ; do
  if [ -f $f.service.in ]; then
    sed -e "s#@OMNISCI_PATH@#${vars['OMNISCI_PATH']}#g" \
        -e "s#@OMNISCI_STORAGE@#${vars['OMNISCI_STORAGE']}#g" \
        -e "s#@OMNISCI_DATA@#${vars['OMNISCI_DATA']}#g" \
        -e "s#@OMNISCI_USER@#${vars['OMNISCI_USER']}#g" \
        -e "s#@OMNISCI_GROUP@#${vars['OMNISCI_GROUP']}#g" \
        $f.service.in > $OMNISCI_TMP/$f.service
    sudo cp $OMNISCI_TMP/$f.service /lib/systemd/system/
  fi
done

sed -e "s#@OMNISCI_PATH@#${vars['OMNISCI_PATH']}#g" \
    -e "s#@OMNISCI_STORAGE@#${vars['OMNISCI_STORAGE']}#g" \
    -e "s#@OMNISCI_DATA@#${vars['OMNISCI_DATA']}#g" \
    -e "s#@OMNISCI_USER@#${vars['OMNISCI_USER']}#g" \
    -e "s#@OMNISCI_GROUP@#${vars['OMNISCI_GROUP']}#g" \
    -e "s#@OMNISCI_PORT@#${vars['OMNISCI_PORT']}#g" \
    -e "s#@OMNISCI_HTTP_PORT@#${vars['OMNISCI_HTTP_PORT']}#g" \
    -e "s#@OMNISCI_CALCITE_PORT@#${vars['OMNISCI_CALCITE_PORT']}#g" \
    omnisci.conf.in > $OMNISCI_TMP/omnisci.conf

sudo cp $OMNISCI_TMP/omnisci.conf ${vars['OMNISCI_STORAGE']}
sudo chown ${vars['OMNISCI_USER']}:${vars['OMNISCI_GROUP']} "${vars['OMNISCI_STORAGE']}/omnisci.conf"

sudo systemctl daemon-reload
