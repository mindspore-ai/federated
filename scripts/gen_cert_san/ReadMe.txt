1. 打开文件v3_san.ext， 在"[alt_names]"章节填写server的ip或域名
2.执行脚本 bash genFlCert.sh 生成证书
3.修改联邦学习server端的config.json，配置如下：
     "server_cert_path": "cert_san/server.p12",     #生成证书的绝对路径
     "crl_path": "",
     "client_cert_path": "cert_san/client.p12",     #生成证书的绝对路径
     "ca_cert_path": "cert_san/cacert.pem",          #生成证书的绝对路径
