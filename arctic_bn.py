from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"`pgmpy\.estimators\.StructureScore` is deprecated",
)

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# Risk matrix: Low / Medium / High
RISK_MATRIX = np.array(
    [
        [0, 0, 1],
        [0, 1, 2],
        [1, 2, 2],
    ],
    dtype=int,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax."""
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def build_ship_speed_cpt() -> np.ndarray:
    """ShipSpeed CPT."""
    base = np.array(
        [
            [0.07, 0.14, 0.30],
            [0.14, 0.17, 0.45],
            [0.69, 0.59, 0.20],
            [0.10, 0.10, 0.05],
        ],
        dtype=float,
    )

    def morph_slower(col: np.ndarray, thickness_idx: int) -> np.ndarray:
        """Shift mass to slower speeds."""
        if thickness_idx == 0:
            return col / col.sum()
        strength = 0.28 * thickness_idx
        p = col.copy()
        frac2 = min(0.70, 0.60 * strength)
        frac3 = min(0.80, 0.85 * strength)
        move2 = p[2] * frac2
        move3 = p[3] * frac3
        total_move = move2 + move3
        p[0] += 0.58 * total_move
        p[1] += 0.42 * total_move
        p[2] -= move2
        p[3] -= move3
        p = np.clip(p, 1e-10, None)
        return p / p.sum()

    table = np.zeros((4, 9))
    for ic in range(3):
        for it in range(3):
            table[:, ic * 3 + it] = morph_slower(base[:, ic], it)
    return table


def _stuck_logits(ic: int, it: int, sp: int, vc: int) -> np.ndarray:
    """Logits for GettingStuck."""
    ice = ic + it
    hull = 2 - vc
    fast = sp / 3.0
    slow = 1.0 - fast
    return np.array(
        [
            0.00 - 0.50 * ice + 2.50 * fast + 0.40 * hull,
            0.00 + 0.20 * ice + 1.20 * slow - 0.20 * hull,
            -3.00 + 0.90 * ice + 2.80 * slow - 0.50 * hull,
        ]
    )


def _coll_logits(ic: int, it: int, sp: int, vc: int) -> np.ndarray:
    """Logits for ShipIceCollision."""
    ice = ic + it
    hull = 2 - vc
    fast = sp / 3.0
    return np.array(
        [
            2.50 - 0.55 * ice - 1.00 * fast + 0.40 * hull,
            0.00 + 0.15 * ice + 0.90 * fast - 0.20 * hull,
            -3.50 + 0.95 * ice + 2.00 * fast - 0.50 * hull,
        ]
    )


def build_occurrence_cpt(*, stuck: bool) -> np.ndarray:
    """GettingStuck or ShipIceCollision CPT."""
    out = np.zeros((3, 108))
    col = 0
    for ic in range(3):
        for it in range(3):
            for sp in range(4):
                for vc in range(3):
                    logits = _stuck_logits(ic, it, sp, vc) if stuck else _coll_logits(ic, it, sp, vc)
                    out[:, col] = _softmax(logits)
                    col += 1
    return out


def build_accident_consequences_cpt() -> np.ndarray:
    """AccidentConsequences CPT."""
    table = np.zeros((3, 36))
    col = 0
    for it in range(3):
        for sp in range(4):
            for vc in range(3):
                fast = sp / 3.0
                hull = 2 - vc
                entrapment = (1.0 - fast) * it * 0.70
                collision = fast * (it * 1.00)
                damage = max(entrapment, collision)
                protection = 0.65 * hull
                eff = float(np.clip(damage - protection, -1.5, 3.5))
                logits = np.array(
                    [
                        1.0 - 0.90 * eff,
                        0.1 + 0.25 * eff,
                        -3.0 + 1.10 * eff,
                    ]
                )
                table[:, col] = _softmax(logits)
                col += 1
    return table


def build_composite_risk_cpt() -> np.ndarray:
    """CompositeRisk CPT."""
    vals = np.zeros((3, 27))
    col = 0
    for gs in range(3):
        for coll in range(3):
            for cons in range(3):
                r_stuck = int(RISK_MATRIX[gs, cons])
                r_coll = int(RISK_MATRIX[coll, cons])
                composite = max(r_stuck, r_coll)
                vals[composite, col] = 1.0
                col += 1
    return vals


def build_model() -> DiscreteBayesianNetwork:
    """Build the model."""
    edges = [
        ("IceConcentration", "ShipSpeed"),
        ("IceThickness", "ShipSpeed"),
        ("IceConcentration", "GettingStuck"),
        ("IceThickness", "GettingStuck"),
        ("ShipSpeed", "GettingStuck"),
        ("VesselIceClass", "GettingStuck"),
        ("IceConcentration", "ShipIceCollision"),
        ("IceThickness", "ShipIceCollision"),
        ("ShipSpeed", "ShipIceCollision"),
        ("VesselIceClass", "ShipIceCollision"),
        ("IceThickness", "AccidentConsequences"),
        ("ShipSpeed", "AccidentConsequences"),
        ("VesselIceClass", "AccidentConsequences"),
        ("GettingStuck", "CompositeRisk"),
        ("ShipIceCollision", "CompositeRisk"),
        ("AccidentConsequences", "CompositeRisk"),
    ]
    model = DiscreteBayesianNetwork(edges)

    cpd_ice_conc = TabularCPD("IceConcentration", 3, [[0.90], [0.06], [0.04]])
    cpd_ice_thick = TabularCPD("IceThickness", 3, [[0.66], [0.24], [0.10]])
    cpd_vessel = TabularCPD("VesselIceClass", 3, [[0.20], [0.50], [0.30]])

    spd_tbl = build_ship_speed_cpt()
    cpd_speed = TabularCPD(
        "ShipSpeed",
        4,
        values=spd_tbl.tolist(),
        evidence=["IceConcentration", "IceThickness"],
        evidence_card=[3, 3],
    )

    stuck_tbl = build_occurrence_cpt(stuck=True)
    cpd_stuck = TabularCPD(
        "GettingStuck",
        3,
        values=stuck_tbl.tolist(),
        evidence=["IceConcentration", "IceThickness", "ShipSpeed", "VesselIceClass"],
        evidence_card=[3, 3, 4, 3],
    )

    coll_tbl = build_occurrence_cpt(stuck=False)
    cpd_coll = TabularCPD(
        "ShipIceCollision",
        3,
        values=coll_tbl.tolist(),
        evidence=["IceConcentration", "IceThickness", "ShipSpeed", "VesselIceClass"],
        evidence_card=[3, 3, 4, 3],
    )

    cons_tbl = build_accident_consequences_cpt()
    cpd_cons = TabularCPD(
        "AccidentConsequences",
        3,
        values=cons_tbl.tolist(),
        evidence=["IceThickness", "ShipSpeed", "VesselIceClass"],
        evidence_card=[3, 4, 3],
    )

    cr_tbl = build_composite_risk_cpt()
    cpd_risk = TabularCPD(
        "CompositeRisk",
        3,
        values=cr_tbl.tolist(),
        evidence=["GettingStuck", "ShipIceCollision", "AccidentConsequences"],
        evidence_card=[3, 3, 3],
    )

    model.add_cpds(
        cpd_ice_conc,
        cpd_ice_thick,
        cpd_vessel,
        cpd_speed,
        cpd_stuck,
        cpd_coll,
        cpd_cons,
        cpd_risk,
    )
    return model


def risk_index_expected(probs: np.ndarray) -> float:
    """Expected risk index."""
    return float(np.dot(probs, np.array([1.0, 10.0, 100.0])))


def _check_occurrence_monotonicity() -> None:
    """Speed check."""
    ic, it, vc = 2, 2, 1
    stuck_probs = [float(_softmax(_stuck_logits(ic, it, sp, vc))[2]) for sp in range(4)]
    coll_probs = [float(_softmax(_coll_logits(ic, it, sp, vc))[2]) for sp in range(4)]
    stuck_mono = all(stuck_probs[i] >= stuck_probs[i + 1] for i in range(3))
    coll_mono = all(coll_probs[i] <= coll_probs[i + 1] for i in range(3))
    print(f"  GettingStuck.Probable: {[round(p, 3) for p in stuck_probs]} monotone↓={stuck_mono}")
    print(f"  ShipIceCollision.Probable: {[round(p, 3) for p in coll_probs]} monotone↑={coll_mono}")


def main() -> None:
    model = build_model()
    ok = model.check_model()
    print(f"Model valid: {ok}")
    assert ok, "CPT validation failed."

    infer = VariableElimination(model)
    names_cr = ["Low", "Medium", "High"]
    speed_labels = ["<5 kn", "5–8 kn", "8–11 kn", ">11 kn"]

    print("\n=== Scenario 1: Severe ice + PC7 ===")
    print("Evidence: IceConc=2, IceThick=2, VesselIceClass=1")

    ref_table5 = {
        0: np.array([0.08, 0.34, 0.58]),
        1: np.array([0.22, 0.52, 0.26]),
        2: np.array([0.58, 0.35, 0.07]),
        3: np.array([0.65, 0.28, 0.07]),
    }

    expectations: list[float] = []
    high_probs: list[float] = []

    for sp in range(4):
        q = infer.query(
            variables=["CompositeRisk"],
            evidence={
                "IceConcentration": 2,
                "IceThickness": 2,
                "VesselIceClass": 1,
                "ShipSpeed": sp,
            },
        )
        probs = q.values
        expectations.append(risk_index_expected(probs))
        high_probs.append(float(probs[2]))
        ref = ref_table5[sp]
        print(
            f"\n  ShipSpeed={speed_labels[sp]}: "
            + ", ".join(f"{names_cr[i]}={probs[i]:.3f}" for i in range(3))
        )
        print(
            f"    E[risk-index]={risk_index_expected(probs):.2f}"
            f" | Table5 target="
            + ", ".join(f"{names_cr[i]}={ref[i]:.3f}" for i in range(3))
        )

    print("\n--- Occurrence monotonicity ---")
    _check_occurrence_monotonicity()

    best = int(np.argmin(expectations))
    print(f"\n  Argmin E[risk-index]: {speed_labels[best]}")

    table5_argmin = int(np.argmin([risk_index_expected(ref_table5[sp]) for sp in range(4)]))
    print(f"\n  Table 5 argmin: {speed_labels[table5_argmin]}")

    ph_min_idx = int(np.argmin(high_probs))
    print(f"\n  P(High) across speeds: {[round(p, 3) for p in high_probs]}")
    print(f"  Minimum at: {speed_labels[ph_min_idx]}")

    print("\n=== Fig. 2.4 validation ===")
    scenarios = [
        ("Light ice", 0, 0, ">11 kn"),
        ("Medium ice", 1, 1, "8–11 kn"),
        ("Severe ice", 2, 2, "5–8 kn"),
    ]
    for label, ic, it, expected_optimum in scenarios:
        exp_vals = []
        for sp in range(4):
            q = infer.query(
                variables=["CompositeRisk"],
                evidence={
                    "IceConcentration": ic,
                    "IceThickness": it,
                    "VesselIceClass": 1,
                    "ShipSpeed": sp,
                },
            )
            exp_vals.append(risk_index_expected(q.values))
        best_sp = int(np.argmin(exp_vals))
        print(f"  {label}: argmin = {speed_labels[best_sp]}, expected = {expected_optimum}")

    print("\n=== Best-case scenario ===")
    q2 = infer.query(
        variables=["CompositeRisk"],
        evidence={"IceConcentration": 0, "IceThickness": 0, "VesselIceClass": 0},
    )
    p2 = q2.values
    print(f"  P(Low)={p2[0]:.4f}, P(Med)={p2[1]:.4f}, P(High)={p2[2]:.4f}")

    print("\n=== Backward inference ===")
    for vc_label, vc in [("High ice class (PC5+)", 0), ("Low ice class (Unclass.)", 2)]:
        qi = infer.query(variables=["IceConcentration"], evidence={"CompositeRisk": 2, "VesselIceClass": vc})
        qt = infer.query(variables=["IceThickness"], evidence={"CompositeRisk": 2, "VesselIceClass": vc})
        qs = infer.query(variables=["ShipSpeed"], evidence={"CompositeRisk": 2, "VesselIceClass": vc})
        print(f"\n  {vc_label}:")
        print(f"    P(IceConc=>70%)    = {float(qi.values[2]):.3f}")
        print(f"    P(IceThick=>80 cm) = {float(qt.values[2]):.3f}")
        print(f"    P(Speed=<5 kn)     = {float(qs.values[0]):.3f}")
        print(f"    P(Speed=>11 kn)    = {float(qs.values[3]):.3f}")

    print("\n=== No evidence ===")
    q_cr = infer.query(variables=["CompositeRisk"])
    print("  Unconditional CompositeRisk: " + ", ".join(f"{names_cr[i]}={q_cr.values[i]:.4f}" for i in range(3)))
    q_sp = infer.query(variables=["ShipSpeed"])
    sp_labels2 = ["<5", "5-8", "8-11", ">11"]
    print("  Unconditional ShipSpeed: " + ", ".join(f"{sp_labels2[i]}kn={q_sp.values[i]:.4f}" for i in range(4)))


if __name__ == "__main__":
    main()