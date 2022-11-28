
user = 'mirzakhani_1937'
password = 'p4h6GGRgeslnWcjIFZoViIY3b92EaC18'
host = 'data.codeup.com'
#host = '157.230.209.171'

def get_db_url(db):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

