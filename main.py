import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from database import (
    init_db,
    add_employee,
    get_employees,
    delete_employee,
    update_status,
    mark_attendance,
    get_attendance,
    add_task,
    get_tasks,
    update_task_status,
    add_leave,
    get_leaves,
    update_leave_status,
)

app = FastAPI(title="Employee Management API")


@app.on_event("startup")
def startup():
    # Creates the database and tables if they do not exist (XAMPP MySQL must be running)
    init_db()


def to_json(df: pd.DataFrame):
    """DataFrame -> JSON-safe list (handles NaN, Decimal, date, time)."""
    df = df.astype(object).where(df.notna(), None)
    return jsonable_encoder(df.to_dict("records"))


# ==========================================
# REQUEST MODELS
# ==========================================

class EmployeeIn(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    department: str
    designation: Optional[str] = None
    gender: str = "Other"
    age: int = Field(25, ge=18, le=70)
    join_date: datetime.date = Field(default_factory=datetime.date.today)
    salary: float = Field(0, ge=0)


class AttendanceIn(BaseModel):
    employee_id: int
    work_date: datetime.date = Field(default_factory=datetime.date.today)
    check_in: Optional[datetime.time] = None
    check_out: Optional[datetime.time] = None
    status: str = "Present"  # Present, Absent, Work From Home, Half Day, On Leave


class TaskIn(BaseModel):
    employee_id: int
    task_date: datetime.date = Field(default_factory=datetime.date.today)
    title: str
    description: Optional[str] = None
    priority: str = "Medium"  # Low, Medium, High
    hours_spent: float = Field(0, ge=0, le=24)


class LeaveIn(BaseModel):
    employee_id: int
    leave_type: str = "Casual"
    start_date: datetime.date
    end_date: datetime.date
    reason: Optional[str] = None


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():
    return {"message": "Welcome to the Employee Management API"}


# ==========================================
# PERFORMANCE PREDICTION (formula based, like the original)
# ==========================================

@app.get("/predict")
def predict_performance(
    attendance: float = Query(..., ge=0, le=100, description="Attendance %"),
    work_hours: float = Query(..., ge=0, le=24, description="Average work hours per day"),
    task_completion: float = Query(..., ge=0, le=100, description="Task completion %"),
):
    score = (
        attendance * 0.3
        + task_completion * 0.4
        + min(work_hours, 10) * 3
    )
    score = max(0, min(100, score))

    if score >= 80:
        rating = "Excellent"
    elif score >= 60:
        rating = "Good"
    elif score >= 40:
        rating = "Needs Improvement"
    else:
        rating = "At Risk"

    return {"predicted_score": round(score, 2), "rating": rating}


# ==========================================
# EMPLOYEES
# ==========================================

@app.post("/employees")
def create_employee(emp: EmployeeIn):
    if emp.name.strip() == "":
        raise HTTPException(status_code=400, detail="Employee name is required")
    add_employee(
        emp.name, emp.email, emp.phone, emp.department, emp.designation,
        emp.gender, emp.age, emp.join_date, emp.salary,
    )
    return {"message": "Employee added successfully"}


@app.get("/employees")
def list_employees(department: Optional[str] = None):
    df = get_employees()
    if department and len(df) > 0:
        df = df[df["department"] == department]
    return to_json(df)


@app.patch("/employees/{employee_id}/status")
def change_status(employee_id: int, status: str):
    update_status(employee_id, status)
    return {"message": "Status updated"}


@app.delete("/employees/{employee_id}")
def remove_employee(employee_id: int):
    delete_employee(employee_id)
    return {"message": "Employee deleted"}


# ==========================================
# DAILY ATTENDANCE
# ==========================================

@app.post("/attendance")
def save_attendance(a: AttendanceIn):
    if a.status in ("Absent", "On Leave"):
        a.check_in, a.check_out = None, None
    mark_attendance(a.employee_id, a.work_date, a.check_in, a.check_out, a.status)
    return {"message": "Attendance saved"}


@app.get("/attendance")
def list_attendance(work_date: Optional[datetime.date] = None):
    return to_json(get_attendance(work_date))


# ==========================================
# DAILY TASKS
# ==========================================

@app.post("/tasks")
def create_task(t: TaskIn):
    add_task(t.employee_id, t.task_date, t.title, t.description, t.priority, t.hours_spent)
    return {"message": "Task saved"}


@app.get("/tasks")
def list_tasks(
    employee_id: Optional[int] = None,
    task_date: Optional[datetime.date] = None,
):
    return to_json(get_tasks(employee_id, task_date))


@app.patch("/tasks/{task_id}/status")
def change_task_status(task_id: int, status: str):
    if status not in ("Pending", "In Progress", "Completed"):
        raise HTTPException(status_code=400, detail="Invalid status")
    update_task_status(task_id, status)
    return {"message": "Task status updated"}


# ==========================================
# LEAVES
# ==========================================

@app.post("/leaves")
def apply_leave(l: LeaveIn):
    if l.end_date < l.start_date:
        raise HTTPException(status_code=400, detail="End date is before start date")
    add_leave(l.employee_id, l.leave_type, l.start_date, l.end_date, l.reason)
    return {"message": "Leave request submitted"}


@app.get("/leaves")
def list_leaves():
    return to_json(get_leaves())


@app.patch("/leaves/{leave_id}/status")
def decide_leave(leave_id: int, status: str):
    if status not in ("Approved", "Rejected", "Pending"):
        raise HTTPException(status_code=400, detail="Invalid status")
    update_leave_status(leave_id, status)
    return {"message": "Leave status updated"}


# ==========================================
# DAILY SUMMARY FOR ONE EMPLOYEE
# ==========================================

@app.get("/employees/{employee_id}/daily-summary")
def daily_summary(employee_id: int, day: Optional[datetime.date] = None):
    day = day or datetime.date.today()

    emp_df = get_employees()
    emp_df = emp_df[emp_df["id"] == employee_id]
    if len(emp_df) == 0:
        raise HTTPException(status_code=404, detail="Employee not found")

    att = get_attendance(day)
    att = att[att["name"] == emp_df.iloc[0]["name"]] if len(att) > 0 else att
    tasks = get_tasks(employee_id, day)

    done = int((tasks["status"] == "Completed").sum()) if len(tasks) > 0 else 0
    hours = float(tasks["hours_spent"].astype(float).sum()) if len(tasks) > 0 else 0.0

    return {
        "employee": emp_df.iloc[0]["name"],
        "date": day,
        "attendance": to_json(att),
        "tasks_total": len(tasks),
        "tasks_completed": done,
        "hours_logged": round(hours, 2),
        "tasks": to_json(tasks),
    }
