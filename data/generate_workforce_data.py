"""
Synthesize temporal workforce data from cross-sectional patterns.

The IBM HR Analytics dataset is a single point-in-time snapshot.
A Bi-LSTM requires sequential monthly data. This module generates
realistic monthly snapshots with turnover distributions derived
from cross-sectional features (tenure, age, job satisfaction).

DEVIATION NOTE: Using synthetic temporal data instead of real HRIS
time-series. This is a known limitation documented in the blueprint.
Production deployment would require 2-3 years of actual HRIS data.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from industries import get_industry

# ---------------------------------------------------------------------------
# Industry-aware workforce data generator.
#
# Previously hardcoded for Energy (DEPARTMENTS, ROLES, dept_sizes, base_salary
# dicts). v6+ reads these from the active industry profile, so a new vertical
# onboards with zero changes to this file — just `set_industry(PROFILE)` and
# call `save_datasets_for_industry(...)`.
#
# The Energy module-level constants below are kept as fallbacks for scripts
# that call the generator with industry=None *before* the industries package
# has been imported (unlikely in practice but harmless).
# ---------------------------------------------------------------------------

DEPARTMENTS = [
    "Operations", "Engineering", "Maintenance",
    "HSE", "Projects", "Commercial", "IT", "HR"
]

ROLES = {
    "Operations": ["Plant Operator", "Control Room Technician", "Shift Supervisor",
                    "Operations Manager", "Field Operator"],
    "Engineering": ["Process Engineer", "Electrical Engineer", "Mechanical Engineer",
                     "Instrumentation Engineer", "Project Engineer"],
    "Maintenance": ["Maintenance Technician", "Reliability Engineer", "Planner/Scheduler",
                     "Welding Inspector", "NDT Technician"],
    "HSE": ["HSE Coordinator", "Safety Engineer", "Environmental Specialist",
            "Fire Protection Engineer"],
    "Projects": ["Project Manager", "Construction Manager", "Commissioning Engineer",
                  "Cost Engineer", "Scheduler"],
    "Commercial": ["Commercial Analyst", "Contracts Manager", "Procurement Specialist"],
    "IT": ["Systems Engineer", "Data Analyst", "SCADA Specialist", "Cybersecurity Analyst"],
    "HR": ["HR Business Partner", "Talent Acquisition Specialist", "Compensation Analyst"],
}

# Energy-specific fallback base salaries (kept for --industry=None invocation).
_LEGACY_ENERGY_BASE_SALARY = {
    "Operations": 75000, "Engineering": 95000, "Maintenance": 70000,
    "HSE": 85000, "Projects": 100000, "Commercial": 90000,
    "IT": 92000, "HR": 78000,
}


def _generate_employee(emp_id, department, start_month, rng,
                       roles_map=None, base_salary_map=None):
    """Generate a single employee's initial record.

    `roles_map` and `base_salary_map` default to the Energy values so
    call sites that don't pass an industry still work unchanged.
    """
    roles_map = roles_map or ROLES
    base_salary_map = base_salary_map or _LEGACY_ENERGY_BASE_SALARY
    role = rng.choice(roles_map[department])
    age = rng.integers(22, 62)
    tenure_years = min(rng.exponential(5), 35)
    tenure_years = max(0.5, tenure_years)

    base_salary = base_salary_map[department]
    salary = base_salary * (1 + 0.03 * tenure_years + rng.normal(0, 0.1))

    # Comp ratio: employee salary vs market midpoint
    comp_ratio = salary / (base_salary * 1.1) + rng.normal(0, 0.05)

    satisfaction = np.clip(rng.normal(3.5, 0.8), 1, 5)
    performance = np.clip(rng.normal(3.2, 0.7), 1, 5)
    engagement = np.clip(rng.normal(3.8, 0.6), 1, 5)

    return {
        "employee_id": emp_id,
        "department": department,
        "role": role,
        "age": int(age),
        "tenure_years": round(tenure_years, 1),
        "salary": round(salary, 0),
        "comp_ratio": round(comp_ratio, 2),
        "satisfaction": round(satisfaction, 2),
        "performance": round(performance, 2),
        "engagement": round(engagement, 2),
    }


def _compute_attrition_prob(emp, month, rng):
    """
    Compute monthly attrition probability based on realistic factors:
    - Low satisfaction / engagement -> higher risk
    - Low comp ratio -> higher risk
    - Age near retirement (>58) -> higher risk
    - January spike (post-bonus departures)
    - Short tenure (<2 years) -> higher risk (new hire churn)
    """
    base = 0.015  # ~1.5% monthly base rate -> ~18% annual

    # Satisfaction effect
    if emp["satisfaction"] < 2.5:
        base += 0.025
    elif emp["satisfaction"] < 3.0:
        base += 0.01

    # Comp ratio effect
    if emp["comp_ratio"] < 0.85:
        base += 0.02
    elif emp["comp_ratio"] < 0.95:
        base += 0.008

    # Retirement cohort
    if emp["age"] > 58:
        base += 0.03
    elif emp["age"] > 55:
        base += 0.01

    # New hire churn
    if emp["tenure_years"] < 2:
        base += 0.012

    # Engagement
    if emp["engagement"] < 2.5:
        base += 0.015

    # January spike (month 1 = January)
    if month % 12 == 1:
        base += 0.01

    # Add noise
    base += rng.normal(0, 0.003)
    return np.clip(base, 0.001, 0.15)


def generate_temporal_dataset(n_employees=300, n_months=36, seed=42,
                               industry=None):
    """
    Generate monthly workforce snapshots for n_months, calibrated to the
    active industry profile.

    Parameters
    ----------
    n_employees : unused kept for backward-compat signature. When `industry`
                  is provided, the total headcount is sum(industry.dept_sizes).
    n_months    : 36 months of synthetic history by default.
    seed        : deterministic — same seed + same industry → same CSVs.
    industry    : an `IndustryBase` instance. When None, uses `get_industry()`
                  (default Energy). The profile's `dept_economics.base_salary`,
                  `dept_sizes`, and `roles_by_dept` become the source of truth.

    Returns
    -------
    employee_df, monthly_df, individual_df
    """
    if industry is None:
        industry = get_industry()

    departments = list(industry.dept_economics.keys())
    roles_map = industry.roles_by_dept or ROLES
    base_salary_map = {
        d: industry.dept_economics[d].base_salary for d in departments
    }
    # dept_sizes: use profile values; fall back to 40/dept if missing.
    dept_sizes = industry.dept_sizes if industry.dept_sizes else {
        d: 40 for d in departments
    }

    rng = np.random.default_rng(seed)

    # Initialize employees across departments
    employees = []
    emp_id = 1
    for dept, size in dept_sizes.items():
        for _ in range(size):
            employees.append(
                _generate_employee(emp_id, dept, 0, rng,
                                   roles_map=roles_map,
                                   base_salary_map=base_salary_map)
            )
            emp_id += 1

    employee_df = pd.DataFrame(employees)
    active = {e["employee_id"]: dict(e) for e in employees}

    # External labor-market index. Column name stays `bls_energy_index` so
    # the trained Bi-LSTM (whose feature column list is fixed) still works.
    # For Finance, this represents BLS Financial Activities (CES 55).
    bls_index = 100 + np.cumsum(rng.normal(0.1, 0.5, n_months))

    monthly_records = []
    individual_records = []

    for month in range(n_months):
        departures = []

        for eid, emp in list(active.items()):
            prob = _compute_attrition_prob(emp, month, rng)
            left = rng.random() < prob

            individual_records.append({
                "employee_id": eid,
                "month": month,
                "department": emp["department"],
                "age": emp["age"],
                "tenure_years": emp["tenure_years"],
                "comp_ratio": emp["comp_ratio"],
                "satisfaction": emp["satisfaction"],
                "performance": emp["performance"],
                "engagement": emp["engagement"],
                "bls_energy_index": round(bls_index[month], 2),
                "attrition": int(left),
            })

            if left:
                departures.append(eid)

        # Remove departed employees
        for eid in departures:
            del active[eid]

        # Hire replacements (partial backfill with 1-2 month lag)
        n_hires = int(len(departures) * rng.uniform(0.5, 0.9))
        for _ in range(n_hires):
            dept = rng.choice(departments)
            new_emp = _generate_employee(
                emp_id, dept, month, rng,
                roles_map=roles_map, base_salary_map=base_salary_map,
            )
            new_emp["tenure_years"] = 0.0
            active[emp_id] = new_emp
            emp_id += 1

        # Age and tenure progression
        for eid in active:
            active[eid]["tenure_years"] = round(active[eid]["tenure_years"] + 1/12, 1)
            if month % 12 == 0:
                active[eid]["age"] += 1
            # Slight satisfaction drift
            active[eid]["satisfaction"] = round(np.clip(
                active[eid]["satisfaction"] + rng.normal(0, 0.05), 1, 5), 2)

        # Aggregate monthly department metrics
        for dept in departments:
            dept_emps = [e for e in active.values() if e["department"] == dept]
            dept_departures = sum(1 for d in departures if d in
                                  {e["employee_id"] for e in employees
                                   if e["department"] == dept})
            monthly_records.append({
                "month": month,
                "department": dept,
                "headcount": len(dept_emps),
                "departures": dept_departures,
                "avg_tenure": round(np.mean([e["tenure_years"] for e in dept_emps]), 1) if dept_emps else 0,
                "avg_satisfaction": round(np.mean([e["satisfaction"] for e in dept_emps]), 2) if dept_emps else 0,
                "avg_comp_ratio": round(np.mean([e["comp_ratio"] for e in dept_emps]), 2) if dept_emps else 0,
                "avg_engagement": round(np.mean([e["engagement"] for e in dept_emps]), 2) if dept_emps else 0,
                "bls_energy_index": round(bls_index[month], 2),
            })

    monthly_df = pd.DataFrame(monthly_records)
    individual_df = pd.DataFrame(individual_records)

    return employee_df, monthly_df, individual_df


def save_datasets(output_dir=None, industry=None):
    """Generate and save workforce CSVs for the (active or provided) industry.

    For the default Energy vertical this writes to `data/*.csv`, preserving
    the v5/v6 on-disk layout. For other verticals, pass the industry explicitly
    and use `save_datasets_for_industry` (below) which routes to the profile's
    `data_subdir`.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    employee_df, monthly_df, individual_df = generate_temporal_dataset(
        industry=industry,
    )

    employee_df.to_csv(output_dir / "employees.csv", index=False)
    monthly_df.to_csv(output_dir / "monthly_department.csv", index=False)
    individual_df.to_csv(output_dir / "individual_monthly.csv", index=False)

    print(f"Saved datasets to {output_dir}")
    print(f"  Employees: {len(employee_df)} records")
    print(f"  Monthly dept: {len(monthly_df)} records")
    print(f"  Individual monthly: {len(individual_df)} records")
    print(f"  Attrition rate: {individual_df['attrition'].mean():.2%}")

    return employee_df, monthly_df, individual_df


def save_datasets_for_industry(industry_key: str,
                                base_dir: Path = None) -> tuple:
    """Generate + persist workforce CSVs for a named industry in the registry.

    Routes output to `data/<industry.data_subdir>/*.csv` — so the dashboard
    loader can pick the right set based on the active industry. Temporarily
    activates the target industry via set_industry() so that forecast_tools
    / intervention lookups during generation match the profile; restores
    the original active industry at the end.
    """
    from industries import REGISTRY, set_industry, get_industry

    if industry_key not in REGISTRY:
        raise ValueError(
            f"Unknown industry '{industry_key}'. "
            f"Registered: {list(REGISTRY.keys())}"
        )

    profile = REGISTRY[industry_key]
    if base_dir is None:
        base_dir = Path(__file__).parent
    out_dir = Path(base_dir) / profile.data_subdir if profile.data_subdir \
        else Path(base_dir)

    prev = get_industry()
    set_industry(profile)
    try:
        result = save_datasets(output_dir=out_dir, industry=profile)
    finally:
        set_industry(prev)
    return result


if __name__ == "__main__":
    save_datasets()
