"""Organisation model: domains, roles, employees, teams, and assets.

Three data-sensitive domains mirror the Chimera/ChimeraLog setting (tech, finance,
healthcare) so results are comparable, but here the org is a *deterministic* object
owned by the simulator rather than an emergent property of LLM agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import RNG, SimConfig


# --- domain packs -----------------------------------------------------------
# Each role: (abbr, title, headcount_weight, privilege 0..3, sensitive_access)
_TECH = [
    ("eng", "Software Engineer", 30, 1, ["source_code", "ci_secrets"]),
    ("sre", "Site Reliability Engineer", 8, 2, ["prod_db", "ci_secrets", "pii"]),
    ("sec", "Security Engineer", 4, 3, ["prod_db", "ci_secrets", "pii", "ids_logs"]),
    ("pm", "Product Manager", 8, 1, ["roadmap", "customer_list"]),
    ("ds", "Data Scientist", 8, 1, ["pii", "analytics"]),
    ("it", "IT Administrator", 4, 3, ["directory", "endpoints", "mail_admin"]),
    ("sales", "Account Executive", 10, 1, ["customer_list", "pricing"]),
    ("hr", "HR Specialist", 4, 2, ["hr_records", "payroll"]),
    ("fin", "Finance Analyst", 4, 2, ["financials", "payroll"]),
    ("exec", "Executive", 3, 2, ["roadmap", "financials", "mna"]),
]
_FINANCE = [
    ("trdr", "Trader", 16, 2, ["trade_book", "market_data", "client_pii"]),
    ("quant", "Quant Researcher", 8, 1, ["models", "market_data"]),
    ("ops", "Operations Analyst", 12, 1, ["settlements", "client_pii"]),
    ("comp", "Compliance Officer", 6, 3, ["trade_book", "surveillance", "client_pii"]),
    ("risk", "Risk Analyst", 6, 2, ["risk_models", "trade_book"]),
    ("it", "IT Administrator", 5, 3, ["directory", "endpoints", "mail_admin"]),
    ("rm", "Relationship Manager", 10, 1, ["client_pii", "pricing"]),
    ("hr", "HR Specialist", 3, 2, ["hr_records", "payroll"]),
    ("fin", "Finance Controller", 4, 2, ["financials", "payroll"]),
    ("exec", "Managing Director", 3, 2, ["financials", "mna", "trade_book"]),
]
_HEALTH = [
    ("phys", "Physician", 18, 1, ["ehr", "phi"]),
    ("nurse", "Nurse", 24, 1, ["ehr", "phi"]),
    ("lab", "Lab Technician", 8, 1, ["lab_results", "phi"]),
    ("bill", "Billing Specialist", 8, 1, ["billing", "phi", "insurance"]),
    ("rsrch", "Clinical Researcher", 6, 1, ["trial_data", "phi"]),
    ("it", "IT Administrator", 4, 3, ["directory", "endpoints", "ehr_admin"]),
    ("hw", "Health Information Mgr", 4, 2, ["ehr_admin", "phi", "release_of_info"]),
    ("hr", "HR Specialist", 3, 2, ["hr_records", "payroll"]),
    ("fin", "Finance Analyst", 3, 2, ["financials", "payroll"]),
    ("admin", "Administrator", 3, 2, ["ehr_admin", "financials"]),
]

DOMAINS: dict[str, list[tuple[str, str, int, int, list[str]]]] = {
    "tech": _TECH,
    "finance": _FINANCE,
    "healthcare": _HEALTH,
}

# crown-jewel assets per domain (targets that make an episode high-impact)
CROWN_JEWELS: dict[str, list[str]] = {
    "tech": ["source_code", "customer_list", "mna", "pii"],
    "finance": ["trade_book", "client_pii", "mna", "models"],
    "healthcare": ["phi", "ehr", "trial_data", "release_of_info"],
}


@dataclass
class Employee:
    """A person in the org, with latent traits that feed the psychology model."""

    emp_id: str
    name: str
    role_abbr: str
    role_title: str
    team: str
    manager_id: str
    privilege: int
    sensitive_access: list[str]
    host: str
    ip: str
    email: str
    # Big Five (OCEAN), each 0..1
    ocean: dict[str, float] = field(default_factory=dict)
    # latent critical-pathway predisposition score 0..1 (requirement R1)
    predisposition: float = 0.0
    # population membership assigned up front (ground-truth design, not a label):
    #   "normal" | "anomalous_benign" | "insider"
    population: str = "normal"
    hire_day: int = -3650  # negative = hired before sim start
    tenure_years: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "emp_id": self.emp_id,
            "name": self.name,
            "role_abbr": self.role_abbr,
            "role_title": self.role_title,
            "team": self.team,
            "manager_id": self.manager_id,
            "privilege": self.privilege,
            "sensitive_access": self.sensitive_access,
            "host": self.host,
            "ip": self.ip,
            "email": self.email,
            "ocean": self.ocean,
            "predisposition": round(self.predisposition, 4),
            "population": self.population,
            "hire_day": self.hire_day,
            "tenure_years": round(self.tenure_years, 2),
        }


_FIRST = (
    "Alex Sam Jordan Taylor Casey Morgan Riley Jamie Quinn Avery Parker Drew Reese "
    "Harper Rowan Sage Emerson Finley Hayden Kai Lane Micah Noor Oakley Remy Shay "
    "Tatum Blake River Skyler Devin Ellis Frankie Gray Indi Jules Kit Lux Marlowe Nico"
).split()
_LAST = (
    "Lee Patel Kim Garcia Nguyen Smith Chen Lopez Khan Ali Rossi Dubois Mueller Novak "
    "Haddad Okafor Silva Costa Ivanov Yamamoto Tan Wong Ahmed Roy Das Park Cho Reyes "
    "Mendez Fischer Weber Moreau Bianchi Romano Suzuki Sato Kowalski Petrov Horvath Ortega"
).split()


@dataclass
class Organization:
    domain: str
    employees: list[Employee]
    teams: dict[str, list[str]]

    def by_id(self, emp_id: str) -> Employee:
        return self._index[emp_id]

    def __post_init__(self) -> None:
        self._index = {e.emp_id: e for e in self.employees}

    def ids(self) -> list[str]:
        return [e.emp_id for e in self.employees]

    def role_map(self) -> dict[str, str]:
        return {e.emp_id: e.role_title for e in self.employees}


def _ocean(gen: np.random.Generator) -> dict[str, float]:
    # mild correlations are ignored; independent Beta draws are adequate here
    keys = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    return {k: float(np.clip(gen.beta(2.5, 2.5), 0, 1)) for k in keys}


def build_org(cfg: SimConfig, rng: RNG) -> Organization:
    """Construct a deterministic organisation for ``cfg``.

    Population assignment (insider / anomalous-benign / normal) is decided here and
    is *design state*, not a label: labels only ever attach to events the person
    actually generates, and the insider population may still behave benignly until
    (and unless) a pathway activates.
    """
    gen = rng.stream("org")
    roles = DOMAINS[cfg.domain]
    weights = np.array([r[2] for r in roles], dtype=float)
    weights /= weights.sum()

    n = cfg.n_employees
    counts = np.floor(weights * n).astype(int)
    # distribute the remainder to the largest roles
    while counts.sum() < n:
        counts[int(np.argmax(weights * n - counts))] += 1

    employees: list[Employee] = []
    teams: dict[str, list[str]] = {}
    idx = 0
    # one manager per role group (first person of the role)
    for (abbr, title, _w, priv, access), c in zip(roles, counts):
        team = f"{cfg.domain}-{abbr}"
        teams.setdefault(team, [])
        manager_id = ""
        for j in range(int(c)):
            idx += 1
            emp_id = f"{abbr}-{j + 1:03d}"
            g = rng.stream("emp", emp_id)
            name = f"{_FIRST[g.integers(len(_FIRST))]} {_LAST[g.integers(len(_LAST))]}"
            host = f"HOST-{abbr.upper()}-{j + 1:03d}"
            ip = f"10.{hash(team) % 200}.{(idx // 250) % 250}.{idx % 250}"
            email = f"{emp_id}@corp.example"
            ocean = _ocean(g)
            # predisposition rises with neuroticism, falls with conscientiousness,
            # plus idiosyncratic variance — this is the CPIR "personal predisposition"
            pre = 0.55 * ocean["neuroticism"] + 0.35 * (1 - ocean["conscientiousness"])
            pre = float(np.clip(0.5 * pre + 0.5 * g.beta(1.6, 4.0), 0, 1))
            tenure = float(np.clip(g.exponential(2.5), 0.05, 20))
            emp = Employee(
                emp_id=emp_id,
                name=name,
                role_abbr=abbr,
                role_title=title,
                team=team,
                manager_id=manager_id,
                privilege=priv,
                sensitive_access=list(access),
                host=host,
                ip=ip,
                email=email,
                ocean=ocean,
                predisposition=pre,
                hire_day=-int(tenure * 365),
                tenure_years=tenure,
            )
            if manager_id == "":
                manager_id = emp_id  # first of role manages the rest
                emp.manager_id = emp_id
            else:
                emp.manager_id = manager_id
            employees.append(emp)
            teams[team].append(emp_id)

    _assign_populations(cfg, rng, employees)
    return Organization(domain=cfg.domain, employees=employees, teams=teams)


def _assign_populations(cfg: SimConfig, rng: RNG, employees: list[Employee]) -> None:
    gen = rng.stream("populations")
    n = len(employees)
    n_insider = max(1, round(cfg.insider_prevalence * n)) if cfg.insider_prevalence > 0 else 0
    n_anom = round(cfg.anomalous_benign_rate * n)

    # insiders are sampled with probability proportional to predisposition — the
    # pathway is *more likely* for the predisposed, but not guaranteed, and plenty
    # of predisposed people remain benign (so predisposition alone cannot be the
    # shortcut label).
    pre = np.array([e.predisposition for e in employees])
    # super-linear in predisposition so insiders skew predisposed, with a floor so
    # anyone *can* become one (and many predisposed people never do).
    p = pre ** 1.5 + 0.08
    p = p / p.sum()
    insider_idx = set(
        gen.choice(n, size=min(n_insider, n), replace=False, p=p).tolist()
    )
    remaining = [i for i in range(n) if i not in insider_idx]
    anom_idx = set(
        gen.choice(remaining, size=min(n_anom, len(remaining)), replace=False).tolist()
    )
    for i, e in enumerate(employees):
        if i in insider_idx:
            e.population = "insider"
        elif i in anom_idx:
            e.population = "anomalous_benign"
        else:
            e.population = "normal"
