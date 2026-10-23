import datetime

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

from database import (
    init_db,
    add_employee,
    get_employees,
    search_employees,
    delete_employee,
    update_status,
    update_prediction,
    update_final_score,
    mark_attendance,
    get_attendance,
    add_task,
    get_tasks,
    update_task_status,
    add_leave,
    get_leaves,
    update_leave_status,
    refresh_employee_metrics,
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Employee Management System",
    page_icon="🏢",
    layout="wide",
)

st.title("🏢 Employee Management System")
st.write("Python + Streamlit + MySQL + XAMPP + AI/ML")

# Create database and tables automatically on first run
try:
    init_db()
except Exception as e:
    st.error(f"Could not connect to MySQL. Is XAMPP running? Error: {e}")
    st.stop()

FEATURES = [
    "attendance",
    "avg_work_hours",
    "tasks_completion",
    "last_rating",
]

# ==========================================
# SIDEBAR
# ==========================================

menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Dashboard",
        "Add Employee",
        "View Employees",
        "Search Employee",
        "Daily Attendance",
        "Daily Tasks",
        "Leave Management",
        "Update Status / Delete",
        "AI/ML Performance Prediction",
    ],
)


def employee_picker(label="Select Employee", key=None):
    """Dropdown of employees, returns (id, name) or (None, None)."""
    df = get_employees()
    if len(df) == 0:
        st.info("No employees yet. Add an employee first.")
        return None, None
    options = {f"{row['id']} - {row['name']}": row["id"] for _, row in df.iterrows()}
    choice = st.selectbox(label, list(options.keys()), key=key)
    return int(options[choice]), choice


# ==========================================
# DASHBOARD
# ==========================================

if menu == "Dashboard":

    st.header("📊 Employee Dashboard")

    try:
        df = get_employees()
        today = datetime.date.today()

        att_today = get_attendance(today)
        present_today = (
            len(att_today[att_today["status"].isin(["Present", "Work From Home", "Half Day"])])
            if len(att_today) > 0 else 0
        )

        tasks_today = get_tasks(task_date=today)
        pending_tasks = (
            len(tasks_today[tasks_today["status"] != "Completed"])
            if len(tasks_today) > 0 else 0
        )

        leaves = get_leaves()
        pending_leaves = (
            len(leaves[leaves["status"] == "Pending"]) if len(leaves) > 0 else 0
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Employees", len(df))
        c2.metric("Present Today", present_today)
        c3.metric("Pending Tasks Today", pending_tasks)
        c4.metric("Pending Leaves", pending_leaves)
        c5.metric(
            "Avg Task Completion",
            f"{df['tasks_completion'].astype(float).mean():.1f}%" if len(df) > 0 else "0%",
        )

        st.subheader("Today's Attendance")
        if len(att_today) > 0:
            st.dataframe(att_today, use_container_width=True)
        else:
            st.info("No attendance marked today.")

        st.subheader("Today's Tasks")
        if len(tasks_today) > 0:
            st.dataframe(tasks_today, use_container_width=True)
        else:
            st.info("No tasks added today.")

        if len(df) > 0:
            st.subheader("Employees by Department")
            st.bar_chart(df["department"].value_counts())

    except Exception as e:
        st.error(f"Database Error: {e}")

# ==========================================
# ADD EMPLOYEE
# ==========================================

elif menu == "Add Employee":

    st.header("➕ Add New Employee")

    name = st.text_input("Employee Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    department = st.selectbox(
        "Department",
        ["HR", "IT", "Finance", "Sales", "Marketing", "Operations", "Support"],
    )
    designation = st.text_input("Designation")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=18, max_value=70, value=25)
    join_date = st.date_input("Join Date", value=datetime.date.today())
    salary = st.number_input("Monthly Salary", min_value=0.0, value=20000.0, step=500.0)

    if st.button("Save Employee"):
        if name.strip() == "":
            st.warning("Please enter employee name.")
        else:
            try:
                add_employee(
                    name, email, phone, department, designation,
                    gender, age, join_date, salary,
                )
                st.success("Employee added successfully!")
            except Exception as e:
                st.error(f"Database Error: {e}")

# ==========================================
# VIEW EMPLOYEES
# ==========================================

elif menu == "View Employees":

    st.header("👥 All Employees")

    try:
        df = get_employees()
        if len(df) == 0:
            st.info("No employees found.")
        else:
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download Employee Data",
                df.to_csv(index=False),
                "employees.csv",
                "text/csv",
            )
    except Exception as e:
        st.error(f"Database Error: {e}")

# ==========================================
# SEARCH EMPLOYEE
# ==========================================

elif menu == "Search Employee":

    st.header("🔎 Search Employee")

    keyword = st.text_input("Enter Employee Name or Department")

    if st.button("Search"):
        try:
            results = search_employees(keyword)
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else:
                st.warning("Employee not found.")
        except Exception as e:
            st.error(f"Database Error: {e}")

# ==========================================
# DAILY ATTENDANCE
# ==========================================

elif menu == "Daily Attendance":

    st.header("🕘 Daily Attendance")

    tab1, tab2 = st.tabs(["Mark Attendance", "Attendance Records"])

    with tab1:
        emp_id, _ = employee_picker(key="att_emp")
        if emp_id:
            work_date = st.date_input("Date", value=datetime.date.today(), key="att_date")
            status = st.selectbox(
                "Status",
                ["Present", "Absent", "Work From Home", "Half Day", "On Leave"],
            )
            check_in = st.time_input("Check-in Time", value=datetime.time(9, 0))
            check_out = st.time_input("Check-out Time", value=datetime.time(18, 0))

            if st.button("Save Attendance"):
                try:
                    if status in ("Absent", "On Leave"):
                        check_in, check_out = None, None
                    mark_attendance(emp_id, work_date, check_in, check_out, status)
                    st.success("Attendance saved.")
                except Exception as e:
                    st.error(f"Database Error: {e}")

    with tab2:
        filter_date = st.date_input("Filter by date", value=datetime.date.today(), key="att_filter")
        show_all = st.checkbox("Show all dates")
        try:
            records = get_attendance(None if show_all else filter_date)
            if len(records) == 0:
                st.info("No attendance records.")
            else:
                st.dataframe(records, use_container_width=True)
                st.download_button(
                    "Download Attendance",
                    records.to_csv(index=False),
                    "attendance.csv",
                    "text/csv",
                )
        except Exception as e:
            st.error(f"Database Error: {e}")

# ==========================================
# DAILY TASKS
# ==========================================

elif menu == "Daily Tasks":

    st.header("📝 Daily Tasks / Work Log")

    tab1, tab2, tab3 = st.tabs(["Assign / Log Task", "View Tasks", "Update Task Status"])

    with tab1:
        emp_id, _ = employee_picker(key="task_emp")
        if emp_id:
            task_date = st.date_input("Task Date", value=datetime.date.today(), key="task_date")
            title = st.text_input("Task Title")
            description = st.text_area("Description")
            priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1)
            hours_spent = st.number_input("Hours Spent", 0.0, 24.0, 1.0, step=0.5)

            if st.button("Save Task"):
                if title.strip() == "":
                    st.warning("Please enter a task title.")
                else:
                    try:
                        add_task(emp_id, task_date, title, description, priority, hours_spent)
                        st.success("Task saved.")
                    except Exception as e:
                        st.error(f"Database Error: {e}")

    with tab2:
        try:
            tasks = get_tasks()
            if len(tasks) == 0:
                st.info("No tasks found.")
            else:
                st.dataframe(tasks, use_container_width=True)
                st.download_button(
                    "Download Tasks",
                    tasks.to_csv(index=False),
                    "tasks.csv",
                    "text/csv",
                )
        except Exception as e:
            st.error(f"Database Error: {e}")

    with tab3:
        task_id = st.number_input("Task ID", min_value=1, step=1)
        new_status = st.selectbox("New Status", ["Pending", "In Progress", "Completed"])
        if st.button("Update Task"):
            try:
                update_task_status(task_id, new_status)
                st.success("Task status updated.")
            except Exception as e:
                st.error(f"Database Error: {e}")

# ==========================================
# LEAVE MANAGEMENT
# ==========================================

elif menu == "Leave Management":

    st.header("🌴 Leave Management")

    tab1, tab2 = st.tabs(["Apply Leave", "Approve / Reject"])

    with tab1:
        emp_id, _ = employee_picker(key="leave_emp")
        if emp_id:
            leave_type = st.selectbox("Leave Type", ["Casual", "Sick", "Paid", "Unpaid"])
            start_date = st.date_input("Start Date", value=datetime.date.today(), key="l_start")
            end_date = st.date_input("End Date", value=datetime.date.today(), key="l_end")
            reason = st.text_area("Reason")

            if st.button("Submit Leave"):
                if end_date < start_date:
                    st.warning("End date cannot be before start date.")
                else:
                    try:
                        add_leave(emp_id, leave_type, start_date, end_date, reason)
                        st.success("Leave request submitted.")
                    except Exception as e:
                        st.error(f"Database Error: {e}")

    with tab2:
        try:
            leaves = get_leaves()
            if len(leaves) == 0:
                st.info("No leave requests.")
            else:
                st.dataframe(leaves, use_container_width=True)
                leave_id = st.number_input("Leave ID", min_value=1, step=1)
                decision = st.selectbox("Decision", ["Approved", "Rejected", "Pending"])
                if st.button("Update Leave"):
                    update_leave_status(leave_id, decision)
                    st.success("Leave status updated.")
        except Exception as e:
            st.error(f"Database Error: {e}")

# ==========================================
# UPDATE STATUS / DELETE
# ==========================================

elif menu == "Update Status / Delete":

    st.header("⚙️ Update Status / Delete Employee")

    emp_id, _ = employee_picker(key="manage_emp")
    if emp_id:
        new_status = st.selectbox("Employee Status", ["Active", "On Leave", "Resigned", "Terminated"])
        if st.button("Update Status"):
            try:
                update_status(emp_id, new_status)
                st.success("Status updated.")
            except Exception as e:
                st.error(f"Database Error: {e}")

        st.divider()
        st.warning("Deleting an employee also deletes their attendance, tasks and leaves.")
        confirm = st.checkbox("I confirm I want to delete this employee")
        if st.button("Delete Employee"):
            if not confirm:
                st.warning("Please tick the confirmation box first.")
            else:
                try:
                    delete_employee(emp_id)
                    st.success("Employee deleted successfully.")
                except Exception as e:
                    st.error(f"Database Error: {e}")

# ==========================================
# AI / ML PERFORMANCE PREDICTION
# ==========================================

elif menu == "AI/ML Performance Prediction":

    st.header("🤖 AI / ML Employee Performance Prediction")
    st.write(
        "Linear Regression predicts an employee's performance score (0-100) "
        "from attendance, daily work hours, task completion and last rating."
    )

    try:
        df = get_employees()

        for col in FEATURES + ["final_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        with st.expander("Record actual performance score (used for training)"):
            emp_id, _ = employee_picker("Employee", key="score_emp")
            if emp_id:
                actual = st.number_input("Actual Performance Score", 0.0, 100.0, 70.0)
                if st.button("Save Actual Score"):
                    update_final_score(emp_id, actual)
                    st.success("Actual score saved.")
                    st.rerun()

        train_data = df.dropna(subset=FEATURES + ["final_score"])

        if len(train_data) < 3:
            st.info(
                "Enter actual performance scores for at least 3 employees "
                "to train the supervised ML model."
            )
        else:
            model = LinearRegression()
            model.fit(train_data[FEATURES], train_data["final_score"])
            st.success(f"ML model trained on {len(train_data)} employees.")

            st.subheader("Predict Employee Performance")

            emp_id, _ = employee_picker("Employee to predict", key="pred_emp")

            if emp_id:
                if st.button("Auto-fill from daily records"):
                    refresh_employee_metrics(emp_id)
                    st.rerun()

                row = df[df["id"] == emp_id].iloc[0]

                attendance = st.number_input("Attendance (%)", 0.0, 100.0,
                                             float(row["attendance"] or 0))
                avg_hours = st.number_input("Avg Work Hours / Day", 0.0, 24.0,
                                            float(row["avg_work_hours"] or 0))
                completion = st.number_input("Task Completion (%)", 0.0, 100.0,
                                             float(row["tasks_completion"] or 0))
                rating = st.number_input("Last Rating", 0.0, 100.0,
                                         float(row["last_rating"] or 0))

                if st.button("Predict Performance"):
                    input_data = pd.DataFrame(
                        [[attendance, avg_hours, completion, rating]],
                        columns=FEATURES,
                    )
                    prediction = float(model.predict(input_data)[0])
                    prediction = max(0, min(100, prediction))

                    st.success(f"Predicted Performance Score: {prediction:.2f}")

                    if prediction >= 80:
                        st.info("Rating: Excellent")
                    elif prediction >= 60:
                        st.info("Rating: Good")
                    elif prediction >= 40:
                        st.warning("Rating: Needs Improvement")
                    else:
                        st.error("Rating: At Risk")

                    update_prediction(emp_id, prediction)
                    st.info("Prediction saved to MySQL.")

    except Exception as e:
        st.error(f"ML / Database Error: {e}")
