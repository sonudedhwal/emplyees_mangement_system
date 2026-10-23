import mysql.connector
import pandas as pd

# ==========================================
# DATABASE CONFIGURATION (XAMPP defaults)
# =======================================

DB_CONFIG = {
    "host": "127.0.0.1",  # Changed from "localhost" to force IPv4 loopback
    "user": "root",
    "password": "",
    "database": "employee_management",
}

def _server_connection():
    """Connection without selecting a database (used to create it)."""
    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
    )


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    """Create the database and all tables if they do not exist."""
    conn = _server_connection()
    cur = conn.cursor()
    cur.execute("CREATE DATABASE IF NOT EXISTS employee_management")
    conn.commit()
    cur.close()
    conn.close()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS employees (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100),
            phone VARCHAR(20),
            department VARCHAR(50),
            designation VARCHAR(50),
            gender VARCHAR(10),
            age INT,
            join_date DATE,
            salary DECIMAL(12,2),
            status VARCHAR(20) DEFAULT 'Active',
            attendance DECIMAL(5,2) DEFAULT 0,
            avg_work_hours DECIMAL(5,2) DEFAULT 0,
            tasks_completion DECIMAL(5,2) DEFAULT 0,
            last_rating DECIMAL(5,2) DEFAULT 0,
            final_score DECIMAL(5,2) NULL,
            predicted_score DECIMAL(5,2) NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            employee_id INT NOT NULL,
            work_date DATE NOT NULL,
            check_in TIME,
            check_out TIME,
            status VARCHAR(20) DEFAULT 'Present',
            UNIQUE KEY uniq_emp_day (employee_id, work_date),
            FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_tasks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            employee_id INT NOT NULL,
            task_date DATE NOT NULL,
            title VARCHAR(200) NOT NULL,
            description TEXT,
            priority VARCHAR(10) DEFAULT 'Medium',
            status VARCHAR(20) DEFAULT 'Pending',
            hours_spent DECIMAL(5,2) DEFAULT 0,
            FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS leave_requests (
            id INT AUTO_INCREMENT PRIMARY KEY,
            employee_id INT NOT NULL,
            leave_type VARCHAR(30),
            start_date DATE,
            end_date DATE,
            reason TEXT,
            status VARCHAR(20) DEFAULT 'Pending',
            FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
        )
        """
    )

    conn.commit()
    cur.close()
    conn.close()


def _query_df(sql, params=None):
    conn = get_connection()
    try:
        return pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()


def _execute(sql, params=None):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        conn.commit()
        cur.close()
    finally:
        conn.close()


# ==========================================
# EMPLOYEES
# ==========================================

def add_employee(name, email, phone, department, designation, gender, age,
                 join_date, salary):
    _execute(
        """
        INSERT INTO employees
        (name, email, phone, department, designation, gender, age, join_date, salary)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (name, email, phone, department, designation, gender, age,
         join_date, salary),
    )


def get_employees():
    return _query_df("SELECT * FROM employees ORDER BY id")


def search_employees(name):
    df = _query_df(
        "SELECT * FROM employees WHERE name LIKE %s OR department LIKE %s",
        (f"%{name}%", f"%{name}%"),
    )
    return df.to_dict("records")


def delete_employee(employee_id):
    _execute("DELETE FROM employees WHERE id = %s", (int(employee_id),))


def update_status(employee_id, status):
    _execute("UPDATE employees SET status = %s WHERE id = %s",
             (status, int(employee_id)))


def update_prediction(employee_id, prediction):
    _execute("UPDATE employees SET predicted_score = %s WHERE id = %s",
             (float(prediction), int(employee_id)))


def update_final_score(employee_id, score):
    _execute("UPDATE employees SET final_score = %s WHERE id = %s",
             (float(score), int(employee_id)))


# ==========================================
# DAILY ATTENDANCE
# ==========================================

def mark_attendance(employee_id, work_date, check_in, check_out, status):
    _execute(
        """
        INSERT INTO daily_attendance (employee_id, work_date, check_in, check_out, status)
        VALUES (%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            check_in = VALUES(check_in),
            check_out = VALUES(check_out),
            status = VALUES(status)
        """,
        (int(employee_id), work_date, check_in, check_out, status),
    )
    refresh_employee_metrics(employee_id)


def get_attendance(work_date=None):
    if work_date:
        return _query_df(
            """
            SELECT a.id, e.name, e.department, a.work_date,
                   a.check_in, a.check_out, a.status
            FROM daily_attendance a
            JOIN employees e ON e.id = a.employee_id
            WHERE a.work_date = %s
            ORDER BY e.name
            """,
            (work_date,),
        )
    return _query_df(
        """
        SELECT a.id, e.name, e.department, a.work_date,
               a.check_in, a.check_out, a.status
        FROM daily_attendance a
        JOIN employees e ON e.id = a.employee_id
        ORDER BY a.work_date DESC, e.name
        """
    )


# ==========================================
# DAILY TASKS
# ==========================================

def add_task(employee_id, task_date, title, description, priority, hours_spent):
    _execute(
        """
        INSERT INTO daily_tasks
        (employee_id, task_date, title, description, priority, hours_spent)
        VALUES (%s,%s,%s,%s,%s,%s)
        """,
        (int(employee_id), task_date, title, description, priority,
         float(hours_spent)),
    )
    refresh_employee_metrics(employee_id)


def get_tasks(employee_id=None, task_date=None):
    sql = """
        SELECT t.id, e.name, t.task_date, t.title, t.description,
               t.priority, t.status, t.hours_spent
        FROM daily_tasks t
        JOIN employees e ON e.id = t.employee_id
        WHERE 1=1
    """
    params = []
    if employee_id:
        sql += " AND t.employee_id = %s"
        params.append(int(employee_id))
    if task_date:
        sql += " AND t.task_date = %s"
        params.append(task_date)
    sql += " ORDER BY t.task_date DESC, t.id DESC"
    return _query_df(sql, tuple(params))


def update_task_status(task_id, status):
    _execute("UPDATE daily_tasks SET status = %s WHERE id = %s",
             (status, int(task_id)))
    df = _query_df("SELECT employee_id FROM daily_tasks WHERE id = %s",
                   (int(task_id),))
    if len(df) > 0:
        refresh_employee_metrics(int(df.iloc[0]["employee_id"]))


# ==========================================
# LEAVE REQUESTS
# ==========================================

def add_leave(employee_id, leave_type, start_date, end_date, reason):
    _execute(
        """
        INSERT INTO leave_requests
        (employee_id, leave_type, start_date, end_date, reason)
        VALUES (%s,%s,%s,%s,%s)
        """,
        (int(employee_id), leave_type, start_date, end_date, reason),
    )


def get_leaves():
    return _query_df(
        """
        SELECT l.id, e.name, l.leave_type, l.start_date, l.end_date,
               l.reason, l.status
        FROM leave_requests l
        JOIN employees e ON e.id = l.employee_id
        ORDER BY l.id DESC
        """
    )


def update_leave_status(leave_id, status):
    _execute("UPDATE leave_requests SET status = %s WHERE id = %s",
             (status, int(leave_id)))


# ==========================================
# AUTO-CALCULATED METRICS (used by ML)
# ==========================================

def refresh_employee_metrics(employee_id):
    """Recalculate attendance %, avg work hours and task completion %."""
    employee_id = int(employee_id)

    att = _query_df(
        """
        SELECT
            COUNT(*) AS total_days,
            SUM(CASE WHEN status IN ('Present','Work From Home','Half Day') THEN 1 ELSE 0 END) AS present_days
        FROM daily_attendance WHERE employee_id = %s
        """,
        (employee_id,),
    )
    total_days = int(att.iloc[0]["total_days"] or 0)
    present_days = int(att.iloc[0]["present_days"] or 0)
    attendance_pct = (present_days / total_days * 100) if total_days else 0

    hrs = _query_df(
        """
        SELECT AVG(TIME_TO_SEC(TIMEDIFF(check_out, check_in)) / 3600) AS avg_hours
        FROM daily_attendance
        WHERE employee_id = %s AND check_in IS NOT NULL AND check_out IS NOT NULL
        """,
        (employee_id,),
    )
    avg_hours = float(hrs.iloc[0]["avg_hours"] or 0)

    tasks = _query_df(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) AS done
        FROM daily_tasks WHERE employee_id = %s
        """,
        (employee_id,),
    )
    total_tasks = int(tasks.iloc[0]["total"] or 0)
    done_tasks = int(tasks.iloc[0]["done"] or 0)
    completion_pct = (done_tasks / total_tasks * 100) if total_tasks else 0

    _execute(
        """
        UPDATE employees
        SET attendance = %s, avg_work_hours = %s, tasks_completion = %s
        WHERE id = %s
        """,
        (round(attendance_pct, 2), round(max(avg_hours, 0), 2),
         round(completion_pct, 2), employee_id),
    )
