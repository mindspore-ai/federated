#!/bin/bash

BASEPATH=$(
  cd "$(dirname "$0")"
  pwd
)

rm -rf ${BASEPATH}/fl_ssl_cert && mkdir ${BASEPATH}/fl_ssl_cert
cd ${BASEPATH}/fl_ssl_cert

echo "basicConstraints = CA:TRUE
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer" > ca_ext.cnf

# generate ca's cert and private key for signing server and client cert
openssl genrsa -out ca.key 3072
openssl req -new -key ca.key -out ca.csr -subj  "/C=CN/ST=Some-State/O=FL/CN=CA"
openssl x509 -req -days 3650 -in ca.csr -signkey ca.key -out ca.crt -extfile ca_ext.cnf

# generate server's cert

IP=127.0.0.1
DNS=MindSporeFederated
CN=federated.mindspore.cn

echo "subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names
[alt_names]
IP.1 = $IP
DNS.1 = $DNS
" > san_ext.cnf

openssl genrsa -out serverkey.pem 3072
openssl req -new -key serverkey.pem -out serverreq.pem -subj "/C=CN/ST=Some-State/O=FL/CN=server"
openssl x509 -req -days 3650 -sha256 -in serverreq.pem -out server.crt -CA ca.crt -CAkey ca.key -CAcreateserial -extfile san_ext.cnf
openssl pkcs12 -export -in server.crt -inkey serverkey.pem -out server.p12 -passout pass:"server_password_12345"

openssl genrsa -out clientkey.pem 3072
openssl req -new -key clientkey.pem -out clientreq.pem -subj "/C=CN/ST=Some-State/O=FL/CN=client"
openssl x509 -req -days 3650 -sha256 -in clientreq.pem -out client.crt -CA ca.crt -CAkey ca.key -CAcreateserial -extfile san_ext.cnf
openssl pkcs12 -export -in client.crt -inkey clientkey.pem -out client.p12 -passout pass:"client_password_12345"

# print cert infos
# openssl x509 -in server.crt -text -noout
# openssl x509 -in client.crt -text -noout
# openssl x509 -in ca.crt -text -noout
