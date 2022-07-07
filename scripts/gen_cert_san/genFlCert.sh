#!/usr/bin/bash
# 生成CA私钥(ca.key)
openssl genrsa -out ca.key 3072 
# 生成CA证书签名请求(ca.csr)
openssl req -new -key ca.key -out ca.csr -subj "/C=CN/ST=Some-State/O=Internet Widgits Pty Ltd/CN=CA"
# 生成自签名CA证书(cacert.pem)
openssl x509 -req -days 3650 -in ca.csr -signkey ca.key -out cacert.pem   -extfile  v3_ca.ext 

#创建客户端证书
# 生成client私钥(clientkey.pem )
openssl genrsa -out clientkey.pem 3072 
# 生成client证书签名请求(clientreq.pem)
openssl req -new -key  clientkey.pem -out clientreq.pem  -subj "/C=CN/ST=Some-State/O=Internet Widgits Pty Ltd/CN=client"
# 生成client签名证书( client.crt )
openssl x509 -req  -days 3650 -sha256  -in clientreq.pem -out client.crt -extfile v3_san.ext -CA cacert.pem -CAkey ca.key -CAcreateserial
# 制作p12格式的证书( client.p12 )
openssl pkcs12 -export -out client.p12 -in client.crt -inkey clientkey.pem -password  pass:123456


#创建server端证书
# 生成server私钥(serverkey.pem )
openssl genrsa -out serverkey.pem 3072 
# 生成server证书签名请求(serverreq.pem)
openssl req -new -key  serverkey.pem -out serverreq.pem  -subj "/C=CN/ST=Some-State/O=Internet Widgits Pty Ltd/CN=server"
# 生成server签名证书( server.crt )
openssl x509 -req  -days 3650 -sha256  -in serverreq.pem -out server.crt -extfile v3_san.ext -CA cacert.pem -CAkey ca.key -CAcreateserial
# 制作p12格式的证书( server.p12 )
openssl pkcs12 -export -out server.p12 -in server.crt -inkey serverkey.pem -password  pass:123456


#检查证书内容：
openssl x509 -in server.crt -text -noout
openssl x509 -in client.crt -text -noout
openssl x509 -in cacert.pem  -text -noout





	


