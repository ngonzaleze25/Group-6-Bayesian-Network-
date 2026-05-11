from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# ==========================================================
# 1. Define Model Structure
# ==========================================================

model = DiscreteBayesianNetwork([
    ('IceConcentration', 'ShipSpeed'),
    ('IceThickness', 'ShipSpeed'),
    ('ShipSpeed', 'GettingStuck'),
    ('ShipSpeed', 'Sinking'),
    ('IceThickness', 'Consequence'),
    ('ShipSpeed', 'Consequence'),
    ('VesselIceClass', 'Consequence'),
    ('GettingStuck', 'CompositeRisk'),
    ('Sinking', 'CompositeRisk'),
])

# ==========================================================
# 2. Root Priors
# ==========================================================

cpd_ice_conc = TabularCPD(
    variable='IceConcentration',
    variable_card=3,
    values=[[0.3], [0.4], [0.3]],
    state_names={'IceConcentration': ['Low', 'Medium', 'High']}
)

cpd_ice_thick = TabularCPD(
    variable='IceThickness',
    variable_card=3,
    values=[[0.4], [0.4], [0.2]],
    state_names={'IceThickness': ['Thin', 'Thick', 'VeryThick']}
)

cpd_vessel = TabularCPD(
    variable='VesselIceClass',
    variable_card=3,
    values=[[0.2], [0.5], [0.3]],
    state_names={'VesselIceClass': ['High', 'Medium', 'Low']}
)

# ==========================================================
# 3. Ship Speed CPT
# ==========================================================

cpd_speed = TabularCPD(
    variable='ShipSpeed',
    variable_card=3,
    evidence=['IceConcentration', 'IceThickness'],
    evidence_card=[3, 3],
    values=[
        [0.1, 0.2, 0.4, 0.2, 0.4, 0.6, 0.5, 0.7, 0.9],  # Slow
        [0.3, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.2, 0.1],  # Medium
        [0.6, 0.4, 0.2, 0.4, 0.2, 0.1, 0.2, 0.1, 0.0],  # Fast
    ],
    state_names={
        'ShipSpeed': ['Slow', 'Medium', 'Fast'],
        'IceConcentration': ['Low', 'Medium', 'High'],
        'IceThickness': ['Thin', 'Thick', 'VeryThick']
    }
)

# ==========================================================
# 4. Getting Stuck
# ==========================================================

cpd_stuck = TabularCPD(
    variable='GettingStuck',
    variable_card=2,
    evidence=['ShipSpeed'],
    evidence_card=[3],
    values=[
        [0.6, 0.3, 0.1],  # Yes
        [0.4, 0.7, 0.9],  # No
    ],
    state_names={
        'GettingStuck': ['Yes', 'No'],
        'ShipSpeed': ['Slow', 'Medium', 'Fast']
    }
)

# ==========================================================
# 5. Sinking
# ==========================================================

cpd_sinking = TabularCPD(
    variable='Sinking',
    variable_card=2,
    evidence=['ShipSpeed'],
    evidence_card=[3],
    values=[
        [0.1, 0.3, 0.6],  # Yes
        [0.9, 0.7, 0.4],  # No
    ],
    state_names={
        'Sinking': ['Yes', 'No'],
        'ShipSpeed': ['Slow', 'Medium', 'Fast']
    }
)

# ==========================================================
# 6. Consequence
# ==========================================================

cpd_consequence = TabularCPD(
    variable='Consequence',
    variable_card=3,
    evidence=['IceThickness', 'ShipSpeed', 'VesselIceClass'],
    evidence_card=[3, 3, 3],
    values=[
        [0.7,0.5,0.3, 0.5,0.3,0.2, 0.3,0.2,0.1,
         0.5,0.3,0.2, 0.3,0.2,0.1, 0.2,0.1,0.05,
         0.3,0.2,0.1, 0.2,0.1,0.05,0.1,0.05,0.02],
        [0.2,0.3,0.4, 0.3,0.4,0.4, 0.4,0.4,0.4,
         0.3,0.4,0.4, 0.4,0.4,0.4, 0.4,0.4,0.4,
         0.4,0.4,0.4, 0.4,0.4,0.4, 0.4,0.4,0.4],
        [0.1,0.2,0.3, 0.2,0.3,0.4, 0.3,0.4,0.5,
         0.2,0.3,0.4, 0.3,0.4,0.5, 0.4,0.5,0.55,
         0.3,0.4,0.5, 0.4,0.5,0.55,0.5,0.55,0.58],
    ],
    state_names={
        'Consequence': ['Minor', 'Major', 'Critical'],
        'IceThickness': ['Thin', 'Thick', 'VeryThick'],
        'ShipSpeed': ['Slow', 'Medium', 'Fast'],
        'VesselIceClass': ['High', 'Medium', 'Low']
    }
)

# ==========================================================
# 7. Composite Risk
# ==========================================================

cpd_risk = TabularCPD(
    variable='CompositeRisk',
    variable_card=3,
    evidence=['GettingStuck', 'Sinking'],
    evidence_card=[2, 2],
    values=[
        [0.05, 0.2, 0.1, 0.7],   # Low
        [0.25, 0.5, 0.4, 0.25],  # Medium
        [0.7, 0.3, 0.5, 0.05],   # High
    ],
    state_names={
        'CompositeRisk': ['Low', 'Medium', 'High'],
        'GettingStuck': ['Yes', 'No'],
        'Sinking': ['Yes', 'No']
    }
)

# ==========================================================
# 8. Add CPDs and Validate
# ==========================================================

model.add_cpds(
    cpd_ice_conc, cpd_ice_thick, cpd_vessel,
    cpd_speed, cpd_stuck, cpd_sinking,
    cpd_consequence, cpd_risk
)

print("Model valid:", model.check_model())

infer = VariableElimination(model)

# ==========================================================
# SCENARIO 1
# Severe ice → vary speed
# ==========================================================

print("\n=== Scenario 1: Risk by Speed under Severe Ice ===")

for speed in ['Slow', 'Medium', 'Fast']:
    result = infer.query(
        variables=['CompositeRisk'],
        evidence={
            'IceConcentration': 'High',
            'IceThickness': 'VeryThick',
            'ShipSpeed': speed
        }
    )
    print(f"\nSpeed = {speed}")
    print(result)

# ==========================================================
# SCENARIO 2
# Diagnostic inference given High Risk
# ==========================================================

print("\n=== Scenario 2: Diagnostic Inference Given High Risk ===")

for var in ['IceConcentration', 'IceThickness', 'ShipSpeed']:
    result = infer.query(
        variables=[var],
        evidence={'CompositeRisk': 'High'}
    )
    print(f"\nPosterior for {var} given High Risk:")
    print(result)