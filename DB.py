import pymysql.cursors
import json
from tqdm import tqdm
# 数据库连接参数
class DB:
    def __init__(self, host, user, password, database):
        self.db_config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }

    def execute_query(self,query, params=None):
        connection = pymysql.connect(**self.db_config)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                connection.commit()
                return cursor.fetchall()
        finally:
            connection.close()
    #检查数据是否存在：
    def check_data_exist(self,id,table):
        # 检查记录是否存在
        check_query = f"""
                SELECT count(id)
                FROM {table}
                WHERE id = %s
                """
        check_params = (id)
        result = self.execute_query(check_query, check_params)
        return result[0]["count(id)"]

    def get_data(self,id,table):
        select_query = f"""
                SELECT case_detail 
                FROM {table}
                WHERE id = %s
                """
        check_params = (id)
        result = self.execute_query(select_query, check_params)
        return result[0]["case_detail"]
    # 插入数据
    def insert_record(self, fact, accusation, plaintiff, defendant, extraction_content,classified_info,buli,youli,zhongli,table):

        query = f"""
        INSERT INTO {table} (fact, accusation, plaintiff, defendant,  extraction_content,classified_info,buli,youli,zhongli)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s,%s)
        """
        params = (fact, accusation, plaintiff, defendant, extraction_content,classified_info,buli,youli,zhongli)
        self.execute_query(query, params)

    def update_all(self, id, fact,extract_content,classified_info,buli,youli,zhongli, table):
        result = self.check_data_exist(id, table)
        print(table)
        if result > 0:
            print("已更新")
            update_query = f"""
                            UPDATE {table}
                            SET Case_Content = %s,Key_Information = %s ,Classified_Information= %s,Unfavorable_Information= %s,Favorable_Information= %s,Neutral_Information = %s
                            WHERE id = %s
                            """
            update_params = (fact,extract_content,classified_info,buli,youli,zhongli, id)
            self.execute_query(update_query, update_params)

    def update_plaintiff(self, id, plaintiff, table):
        # result = self.check_data_exist(id, table)
        # print(table)
        # if result > 0:
        update_query = f"""
                        UPDATE {table}
                        SET plaintiff = %s
                        WHERE id = %s
                        """
        update_params = (plaintiff, id)
        print("已更新")
        self.execute_query(update_query, update_params)
    def insert_or_update_record_to_direct(self, id, history, table):

        # 检查数据是否存在
        result = self.check_data_exist(id, table)
        print(table)
        try:
            if result > 0:
                # 如果记录存在，则更新记录
                update_query = f"""
                UPDATE {table}
                SET chat_history = %s
                WHERE id = %s
                """
                update_params = (history, id)
                self.execute_query(update_query, update_params)
                print("已更新")
            else:
                # 如果记录不存在，则插入新记录
                insert_query = f"""
                INSERT INTO {table} (id, chat_history)
                VALUES (%s, %s)
                """
                insert_params = (id, history)
                self.execute_query(insert_query, insert_params)
        except pymysql.MySQLError as e:
            print(e)

    # 删除数据
    def delete_record(self,record_id,table):
        query = f"DELETE FROM {table} WHERE id = %s"
        params = (record_id,)
        self.execute_query(query, params)

    # 查询数据
    def fetch_records(self,column_name,table):
        query = f"SELECT {column_name} FROM {table}"
        return self.execute_query(query)
        # 增加字段并赋值

    # 根据id查询数据
    def fetch_record_by_id(self, column_name, table, record_id):
        query = f"SELECT {column_name} FROM {table} WHERE id = %s"
        return self.execute_query(query, (record_id,))

    # 检查字段是否存在
    def check_column_existence(self, table, column_name):
        query = f"""
        SELECT COUNT(*) AS column_count
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = '{self.db_config['database']}'
          AND TABLE_NAME = '{table}'
          AND COLUMN_NAME = '{column_name}'
        """
        result = self.execute_query(query)
        column_count = result[0]['column_count']
        return column_count > 0

    def add_column_and_update(self, table, new_column_name, values,type,id):
        # 增加新字段
        if not self.check_column_existence(table, new_column_name):
            add_column_sql = f"""
               ALTER TABLE {table}
               ADD COLUMN {new_column_name} {type};
               """
            self.execute_query(add_column_sql)

        update_column_sql = f"""
           UPDATE {table}
           SET {new_column_name} = %s
           WHERE id = %s;
           """
        self.execute_query(update_column_sql, (values, id))


if __name__ == '__main__':
    db = DB(host='localhost', user='root', password='020201', database='deception_detection')
    messages = db.fetch_records("*","law_data")
    print(messages)
    # 查找信息
    # 处理分类之后的数据
    for item in tqdm(messages):
        fact = item["fact"]
        masked_info = item["masked_info"]
        classify_message = json.loads(item["classification_message"])
        criminal = item["criminals"]
        id = item["id"]
        # 获取信息分类：
        key=classify_message.items()

        buli_info = classify_message["不利信息"]
        youli_info = classify_message["有利信息"]
        zhongli_info = classify_message["中立信息"]
        all_info=[]
        all_info.append(buli_info)
        all_info.append(youli_info)
        all_info.append(zhongli_info)
        all_key=["buli","youli","zhongli"]
        for index,item in enumerate(all_info):
            process_info=""
            for i, info in enumerate(item):
                process_info+="信息"+str(i + 1) + ":" + info + "\n"
            db.add_column_and_update('law_data', all_key[index], process_info,"TEXT", id)  # print(all_key[index])
                # str_harm += "信息" + str(i + 1) + ":" + info + "\n"



