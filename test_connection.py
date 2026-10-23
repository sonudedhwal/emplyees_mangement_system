import datetime

import mysql.connector

try:
    conn = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="",
        database="employee_management",
        connection_timeout=5,
    )
    cursor = conn.cursor()

    # 1. Tables and how many rows each has
    cursor.execute("SHOW TABLES")
    tables = [row[0] for row in cursor.fetchall()]
    print("Tables:", tables)

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
        print(f"  {table}: {cursor.fetchone()[0]} rows")

    # 2. Today's day-to-day work snapshot
    today = datetime.date.today()

    if "daily_attendance" in tables:
        cursor.execute(
            "SELECT COUNT(*) FROM daily_attendance "
            "WHERE work_date = %s AND status IN ('Present','Work From Home','Half Day')",
            (today,),
        )
        print(f"Present today ({today}):", cursor.fetchone()[0])

    if "daily_tasks" in tables:
        cursor.execute(
            "SELECT COUNT(*) FROM daily_tasks "
            "WHERE task_date = %s AND status <> 'Completed'",
            (today,),
        )
        print(f"Pending tasks today ({today}):", cursor.fetchone()[0])

    cursor.close()
    conn.close()
    print("Connection OK")

except mysql.connector.Error as e:
    print("Failed:", e)
    print("Check that MySQL is running in XAMPP and that the app has been "
          "started once (it creates the employee_management database).")
