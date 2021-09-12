import pymysql
import config


db = pymysql.connect(host=config.data_config['host'],
                     port=config.data_config['port'],
                     user=config.data_config['user'],
                     passwd=config.data_config['passwd'],
                     db=config.data_config['db'],
                     charset=config.data_config['charset'])


def query_select(query):
    cur = db.cursor(pymysql.cursors.DictCursor)
    cur.execute(query)
    res = cur.fetchall()
    cur.close()
    return res


def get_log():
    query = "SELECT user_id user, apart_id apt FROM click_log"
    return query_select(query)
