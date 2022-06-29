import mysql.connector

mydb = mysql.connector.connect(
    host = "localhost",
    user = "chau",
    password = "chau123"
)

print(mydb)