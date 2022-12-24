# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# pylint: disable=broad-except
"""Generate mysql random data."""
import argparse
import random

import pymysql


def get_parser():
    """Get args."""
    parser = argparse.ArgumentParser(description="Run generate_random_data.py case")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--database", type=str, default="mydb")
    parser.add_argument("--charset", type=str, default="utf8")
    parser.add_argument("--user", type=str, default="root")
    parser.add_argument("--password", type=str, default="123456")
    parser.add_argument("--leader_table", type=str, default="leader_table")
    parser.add_argument("--follower_table", type=str, default="follower_table")
    return parser


args, _ = get_parser().parse_known_args()
host = args.host
port = args.port
database = args.database
charset = args.charset
user = args.user
password = args.password
leader_table = args.leader_table
follower_table = args.follower_table


class Mysqldb():
    """Mysql database source"""

    def __init__(self):
        self.conn = self.get_conn()
        self.cursor = self.get_cursor()

    def get_conn(self):
        conn = pymysql.connect(host=host,
                               port=port,
                               database=database,
                               charset=charset,
                               user=user,
                               password=password)
        return conn

    def get_cursor(self):
        cursor = self.conn.cursor()
        return cursor

    def select_all(self, sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def select_one(self, sql):
        self.cursor.execute(sql)
        return self.cursor.fetchone()

    def select_many(self, sql, num):
        self.cursor.execute(sql)
        return self.cursor.fetchmany(num)

    def commit_data(self, sql):
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            print("commit success")
        except Exception as e:
            print(e)
            self.conn.rollback()

    def __del__(self):
        self.cursor.close()
        self.conn.close()


if __name__ == '__main__':

    db = Mysqldb()
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'

    db.commit_data(
        "DROP TABLE if exists {}".format(follower_table))

    db.commit_data(
        "DROP TABLE if exists {}".format(leader_table))

    db.commit_data(
        "CREATE TABLE {} (id INT AUTO_INCREMENT PRIMARY KEY, oaid VARCHAR(255), feature0 FLOAT, "
        "feature1 FLOAT, feature2 FLOAT, feature3 FLOAT, feature4 FLOAT, feature5 FLOAT, feature6 FLOAT,"
        "feature7 FLOAT, feature8 FLOAT, feature9 FLOAT)".format(follower_table))

    db.commit_data(
        "CREATE TABLE {} (id INT AUTO_INCREMENT PRIMARY KEY, oaid VARCHAR(255), "
        "feature10 FLOAT, feature11 FLOAT, feature12 FLOAT, feature13 FLOAT, \
        feature14 FLOAT, feature15 FLOAT, feature16 FLOAT, feature17 FLOAT, feature18 FLOAT,"
        "feature19 FLOAT, feature20 FLOAT, feature21 FLOAT)".format(leader_table))

    for i in range(1000):
        oaid = ""
        for j in range(32):
            oaid += random.choice(base_str)
        print(oaid)
        rans = []
        for j in range(30):
            ran_float = random.uniform(1.0, 1000.0)
            rans.append(ran_float)
        fsql = "insert into {}(oaid,feature0,feature1,feature2,feature3,feature4,feature5," \
               "feature6,feature7,feature8,feature9) values('{}',{},{},{},{},{},{},{},{},{},{})".format(follower_table,
                                                                                                        oaid, rans[0],
                                                                                                        rans[1],
                                                                                                        rans[2],
                                                                                                        rans[3],
                                                                                                        rans[4],
                                                                                                        rans[5],
                                                                                                        rans[6],
                                                                                                        rans[7],
                                                                                                        rans[8],
                                                                                                        rans[9])
        db.commit_data(sql=fsql)
        lsql = "insert into {}(oaid,feature10,feature11,feature12,feature13,feature14,feature15," \
               "feature16,feature17,feature18,feature19,feature20,feature21) " \
               "values('{}',{},{},{},{},{},{},{},{},{},{},{},{})".format(leader_table,
                                                                         oaid, rans[10], rans[11], rans[12], rans[13],
                                                                         rans[14], rans[15], rans[16], rans[17],
                                                                         rans[18], rans[19], rans[20], rans[21])
        db.commit_data(sql=lsql)
        res = db.select_all("select oaid,feature0,feature1,feature2,feature3,feature4,feature5," \
                            "feature6,feature7,feature8,feature9 from {}".format(follower_table))
        for item in res:
            print(item)
        res = db.select_all("select oaid,feature10,feature11,feature12,feature13,feature14,feature15," \
                            "feature16,feature17,feature18,feature19 from {}".format(leader_table))
        for item in res:
            print(item)
