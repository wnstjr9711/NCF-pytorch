import pymysql
import config


def query_select(query):
    db = pymysql.connect(host=config.data_config['host'],
                         port=config.data_config['port'],
                         user=config.data_config['user'],
                         passwd=config.data_config['passwd'],
                         db=config.data_config['db'],
                         charset=config.data_config['charset'])
    cur = db.cursor(pymysql.cursors.DictCursor)
    cur.execute(query)
    res = cur.fetchall()
    cur.close()
    return res


def get_log():
    query = "SELECT user_id user, apart_id apt, 1 as 'searched' FROM click_log"
    return query_select(query)


def get_log2():
    query = "SELECT user_id, apart_id, 1 as 'searched' FROM click_log"
    return query_select(query)


def get_user():
    query = "SELECT id user_id, age, gender FROM user"
    return query_select(query)


def get_apt():
    query = "SELECT id apart_id, price, built_at, gu, dong FROM apart"
    return query_select(query)


def get_user_average(uid):
    query = "SELECT price FROM click_log, apart WHERE click_log.apart_id = apart.id and user_id = {}".format(uid)
    return query_select(query)


def get_apt_price():
    query = "SELECT id, price FROM apart"
    return query_select(query)


def get_user_selected(uid):
    query = "SELECT price FROM click_log, apart WHERE click_log.apart_id = apart.id and user_id = {}".format(uid)
    return query_select(query)
