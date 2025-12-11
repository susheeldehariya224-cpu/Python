import sqlite3
import random

DB = "student.db"

names = [
    "Aarav", "Priya", "Rohan", "Meera", "Arjun", "Simran", "Dev", "Isha",
    "Kabir", "Ananya", "Aditya", "Sneha", "Rahul", "Tanya", "Vikram", "Nisha",
    "Manish", "Pooja", "Harsh", "Riya", "Aryan", "Sakshi", "Karan", "Alok",
    "Neha", "Varun", "Shreya", "Laksh", "Komal", "Raj", "Diya", "Yash",
    "Anjali", "Mohit", "Kavya", "Suresh", "Payal", "Vivek", "Radhika", "Om",
    # extra names for randomness
    "Kiran", "Sunil", "Nidhi", "Bipin", "Leena", "Ramesh", "Preeti", "Tuhin",
    "Gargi", "Zara", "Faisal", "Reena", "Vikrant", "Kajal", "Irfan", "Sana"
]

classes = ["AI", "Machine Learning", "Web Dev", "Cyber Security", "Cloud Computing", "Data Analytics", "Data Science"]
sections = ["A", "B", "C", "D", None]  # None will be inserted as NULL occasionally

random.seed(42)  # reproducible

conn = sqlite3.connect(DB)
cur = conn.cursor()

# make sure table exists
cur.execute("""CREATE TABLE IF NOT EXISTS STUDENT(NAME VARCHAR(50), CLASS VARCHAR(50), SECTION VARCHAR(10), MARKS INT)""")

def rand_name():
    # choose random and sometimes add suffix to create duplicates/variants
    n = random.choice(names)
    if random.random() < 0.15:
        n = n + "-" + str(random.randint(1, 99))
    return n

def rand_marks():
    # generate variety: skewed distribution plus extremes
    r = random.random()
    if r < 0.05:
        return 100
    if r < 0.1:
        return 0
    if r < 0.2:
        return random.randint(1, 30)
    if r < 0.8:
        return random.randint(31, 90)
    return random.randint(91, 99)

count = 200  # change to how many rows you want
for _ in range(count):
    n = rand_name()
    c = random.choice(classes)
    s = random.choice(sections)
    m = rand_marks()
    # occasionally insert NULL marks
    if random.random() < 0.02:
        m = None
    cur.execute("INSERT INTO STUDENT (NAME, CLASS, SECTION, MARKS) VALUES (?, ?, ?, ?)", (n, c, s, m))

conn.commit()
conn.close()
print(f"Inserted {count} random rows into {DB}")