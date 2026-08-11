import os
import itertools
import time
import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

DATASET_FOLDER = "dataset"

FILES = {
    "z2": "74283_modified_z2_nand2_to_and2_ground_truth_0noise.csv",
    "z17": "74283_modified_z17_and5_to_nand5_ground_truth_0noise.csv",
    "z18": "74283_modified_z18_and2_to_nand2_ground_truth_0noise.csv",
    "o1": "74283_modified_o1_nor5_to_or5_ground_truth_0noise.csv"
}

INPUTS = [
    "i1", "i2", "i3", "i4", "i5",
    "i6", "i7", "i8", "i9"
]

OUTPUTS = [
    "o1", "o2", "o3", "o4", "o5"
]

EXPECTED_OUTPUTS = [
    "S0",
    "S1",
    "S2",
    "S3",
    "Cout"
]


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("74283 OPTIMIZED PORT MAPPING ANALYSIS")
print("=" * 70)

frames = []

for anomaly, filename in FILES.items():

    path = os.path.join(
        DATASET_FOLDER,
        filename
    )

    if not os.path.exists(path):

        print("\nERROR: Dataset not found:")
        print(path)

        exit()

    df = pd.read_csv(path)

    # Remove accidental CSV index columns
    df = df.loc[
        :,
        ~df.columns.str.contains("^Unnamed")
    ]

    missing = [
        column
        for column in INPUTS + OUTPUTS
        if column not in df.columns
    ]

    if missing:

        print(
            f"\nERROR: Missing columns in {filename}:"
        )

        print(missing)

        exit()

    df = df[
        INPUTS + OUTPUTS
    ].copy()

    df["dataset"] = anomaly

    frames.append(df)

    print(
        f"Loaded {anomaly:4s}: "
        f"{len(df)} observations"
    )


data = pd.concat(
    frames,
    ignore_index=True
)


# Convert everything to integer

for column in INPUTS + OUTPUTS:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


data = data.dropna(
    subset=INPUTS + OUTPUTS
)


for column in INPUTS + OUTPUTS:

    data[column] = (
        data[column]
        .astype(int)
    )


print()
print(
    f"Total observations: {len(data)}"
)


# ============================================================
# 2. CHECK INPUT COVERAGE
# ============================================================

unique_inputs = (
    data[INPUTS]
    .drop_duplicates()
)


print()
print("=" * 70)
print("INPUT COVERAGE")
print("=" * 70)

print(
    f"Unique i1-i9 combinations : "
    f"{len(unique_inputs)}"
)

print(
    "Possible 9-bit combinations : 512"
)


if len(unique_inputs) == 512:

    print(
        "Coverage                    : COMPLETE"
    )

else:

    print(
        "Coverage                    : INCOMPLETE"
    )


# ============================================================
# 3. NUMPY ARRAYS
# ============================================================

X = data[
    INPUTS
].to_numpy(dtype=np.int8)

Y = data[
    OUTPUTS
].to_numpy(dtype=np.int8)


# ============================================================
# 4. SEARCH SETUP
# ============================================================

print()
print("=" * 70)
print("STARTING OPTIMIZED SEARCH")
print("=" * 70)

print(
    """
We need to infer:

A0-A3
B0-B3
Cin

from:

i1-i9

and infer:

S0-S3
Cout

from:

o1-o5

Because A + B = B + A, swapping the complete
A and B operands represents the same arithmetic
mapping.

The available datasets contain faults, so the
best mapping is determined by maximum agreement
with ideal 74283 addition.
"""
)


start_time = time.time()


# ============================================================
# 5. PRECOMPUTE OUTPUT AGREEMENT
# ============================================================

# Given five ideal output vectors, we need to determine which
# observed o-column best corresponds to each ideal output.
#
# Instead of repeatedly slicing large arrays for every
# permutation, calculate a 5x5 agreement matrix:
#
# ideal signal x observed signal


def find_best_output_mapping(expected):

    agreement = np.zeros(
        (5, 5),
        dtype=float
    )


    for ideal_index in range(5):

        expected_column = expected[
            :,
            ideal_index
        ]


        for observed_index in range(5):

            agreement[
                ideal_index,
                observed_index
            ] = np.mean(
                expected_column
                ==
                Y[:, observed_index]
            )


    best_score = -1

    best_perm = None


    # Only 5! = 120 possibilities.
    #
    # But now each permutation operates on the tiny
    # 5x5 agreement matrix instead of all 1680 rows.

    for permutation in itertools.permutations(
        range(5)
    ):

        score = sum(
            agreement[
                ideal_index,
                permutation[ideal_index]
            ]

            for ideal_index in range(5)
        ) / 5


        if score > best_score:

            best_score = score

            best_perm = permutation


    return (
        best_score,
        best_perm,
        agreement
    )


# ============================================================
# 6. SEARCH INPUT MAPPINGS
# ============================================================

best_candidates = []

tested = 0


# There are 9 possibilities for Cin.
#
# After choosing Cin, eight signals remain.
#
# We divide those into two groups of four.
# A/B symmetry lets us eliminate duplicate group swaps.


for cin_index in range(9):

    print()
    print(
        f"Testing Cin candidate: "
        f"{INPUTS[cin_index]}"
    )


    remaining = [
        index
        for index in range(9)
        if index != cin_index
    ]


    # Fix first remaining signal into operand A.
    # This removes equivalent A/B group swapping.

    fixed = remaining[0]


    combinations = list(
        itertools.combinations(
            remaining[1:],
            3
        )
    )


    for group_number, extra in enumerate(
        combinations,
        start=1
    ):

        group_a = (
            fixed,
        ) + extra


        group_b = tuple(
            index
            for index in remaining
            if index not in group_a
        )


        # Try all possible bit significance orders
        # for the two 4-bit operands.

        for a_order in itertools.permutations(
            group_a
        ):

            A = (
                X[:, a_order[0]]
                +
                2 * X[:, a_order[1]]
                +
                4 * X[:, a_order[2]]
                +
                8 * X[:, a_order[3]]
            )


            for b_order in itertools.permutations(
                group_b
            ):

                B = (
                    X[:, b_order[0]]
                    +
                    2 * X[:, b_order[1]]
                    +
                    4 * X[:, b_order[2]]
                    +
                    8 * X[:, b_order[3]]
                )


                Cin = X[
                    :,
                    cin_index
                ]


                total = (
                    A + B + Cin
                )


                expected = np.column_stack([

                    total & 1,

                    (total >> 1) & 1,

                    (total >> 2) & 1,

                    (total >> 3) & 1,

                    (total >> 4) & 1

                ]).astype(np.int8)


                (
                    score,
                    output_perm,
                    _
                ) = find_best_output_mapping(
                    expected
                )


                tested += 1


                candidate = {

                    "score": score,

                    "cin": cin_index,

                    "a_order": a_order,

                    "b_order": b_order,

                    "output_perm": output_perm
                }


                # Don't store every candidate.
                #
                # Keep only the strongest 20.

                best_candidates.append(
                    candidate
                )


                best_candidates.sort(
                    key=lambda item: item["score"],
                    reverse=True
                )


                if len(best_candidates) > 20:

                    best_candidates = (
                        best_candidates[:20]
                    )


        print(
            f"  Operand grouping "
            f"{group_number:2d}/{len(combinations)} "
            f"complete"
        )


    elapsed = (
        time.time()
        -
        start_time
    )


    print(
        f"Finished Cin={INPUTS[cin_index]} "
        f"| elapsed {elapsed:.1f} seconds"
    )


# ============================================================
# 7. SEARCH COMPLETE
# ============================================================

elapsed = (
    time.time()
    -
    start_time
)


print()
print("=" * 70)
print("SEARCH COMPLETE")
print("=" * 70)

print(
    f"Input mappings evaluated : {tested}"
)

print(
    f"Total search time         : "
    f"{elapsed:.2f} seconds"
)


# ============================================================
# 8. DISPLAY TOP 10
# ============================================================

print()
print("=" * 70)
print("TOP MAPPING CANDIDATES")
print("=" * 70)


for rank, result in enumerate(
    best_candidates[:10],
    start=1
):

    print()
    print("-" * 70)

    print(
        f"CANDIDATE #{rank}"
    )

    print(
        f"Average bit agreement: "
        f"{result['score'] * 100:.2f}%"
    )


    a_order = result[
        "a_order"
    ]

    b_order = result[
        "b_order"
    ]

    cin_index = result[
        "cin"
    ]

    output_perm = result[
        "output_perm"
    ]


    print()
    print("Inputs:")

    print(
        f"A0={INPUTS[a_order[0]]}, "
        f"A1={INPUTS[a_order[1]]}, "
        f"A2={INPUTS[a_order[2]]}, "
        f"A3={INPUTS[a_order[3]]}"
    )

    print(
        f"B0={INPUTS[b_order[0]]}, "
        f"B1={INPUTS[b_order[1]]}, "
        f"B2={INPUTS[b_order[2]]}, "
        f"B3={INPUTS[b_order[3]]}"
    )

    print(
        f"Cin={INPUTS[cin_index]}"
    )


    print()
    print("Outputs:")


    for ideal_index, name in enumerate(
        EXPECTED_OUTPUTS
    ):

        print(
            f"{name:4s} = "
            f"{OUTPUTS[output_perm[ideal_index]]}"
        )


# ============================================================
# 9. ANALYZE BEST CANDIDATE
# ============================================================

best = best_candidates[0]


a_order = best[
    "a_order"
]

b_order = best[
    "b_order"
]

cin_index = best[
    "cin"
]

output_perm = best[
    "output_perm"
]


A = (
    X[:, a_order[0]]
    +
    2 * X[:, a_order[1]]
    +
    4 * X[:, a_order[2]]
    +
    8 * X[:, a_order[3]]
)


B = (
    X[:, b_order[0]]
    +
    2 * X[:, b_order[1]]
    +
    4 * X[:, b_order[2]]
    +
    8 * X[:, b_order[3]]
)


Cin = X[
    :,
    cin_index
]


total = (
    A + B + Cin
)


expected = np.column_stack([

    total & 1,

    (total >> 1) & 1,

    (total >> 2) & 1,

    (total >> 3) & 1,

    (total >> 4) & 1

]).astype(np.int8)


mapped_outputs = Y[
    :,
    output_perm
]


bit_matches = (
    mapped_outputs
    ==
    expected
)


row_matches = np.all(
    bit_matches,
    axis=1
)


# ============================================================
# 10. FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("BEST INFERRED MAPPING")
print("=" * 70)


print()
print("74283 INPUT PORTS")
print("-" * 30)


print(
    f"A0   -> {INPUTS[a_order[0]]}"
)

print(
    f"A1   -> {INPUTS[a_order[1]]}"
)

print(
    f"A2   -> {INPUTS[a_order[2]]}"
)

print(
    f"A3   -> {INPUTS[a_order[3]]}"
)


print()


print(
    f"B0   -> {INPUTS[b_order[0]]}"
)

print(
    f"B1   -> {INPUTS[b_order[1]]}"
)

print(
    f"B2   -> {INPUTS[b_order[2]]}"
)

print(
    f"B3   -> {INPUTS[b_order[3]]}"
)


print()


print(
    f"Cin  -> {INPUTS[cin_index]}"
)


print()
print("74283 OUTPUT PORTS")
print("-" * 30)


for ideal_index, name in enumerate(
    EXPECTED_OUTPUTS
):

    print(
        f"{name:4s} -> "
        f"{OUTPUTS[output_perm[ideal_index]]}"
    )


# ============================================================
# 11. AGREEMENT STATISTICS
# ============================================================

print()
print("=" * 70)
print("AGREEMENT WITH IDEAL 74283")
print("=" * 70)


print(
    f"\nAverage individual-bit agreement: "
    f"{bit_matches.mean() * 100:.2f}%"
)


print(
    f"Rows where ALL five outputs match: "
    f"{row_matches.mean() * 100:.2f}%"
)


print()
print("Per-output agreement:")


for index, name in enumerate(
    EXPECTED_OUTPUTS
):

    agreement = (
        bit_matches[:, index].mean()
        * 100
    )

    print(
        f"{name:4s}: "
        f"{agreement:.2f}%"
    )


# ============================================================
# 12. PER-DATASET ANALYSIS
# ============================================================

print()
print("=" * 70)
print("PER-ANOMALY DATASET AGREEMENT")
print("=" * 70)


dataset_labels = data[
    "dataset"
].to_numpy()


for anomaly in FILES.keys():

    mask = (
        dataset_labels
        ==
        anomaly
    )


    dataset_bit_accuracy = (
        bit_matches[
            mask
        ].mean()
        * 100
    )


    dataset_row_accuracy = (
        row_matches[
            mask
        ].mean()
        * 100
    )


    print()

    print(
        f"{anomaly}:"
    )

    print(
        f"  Bit agreement       : "
        f"{dataset_bit_accuracy:.2f}%"
    )

    print(
        f"  Full-output matches : "
        f"{dataset_row_accuracy:.2f}%"
    )


# ============================================================
# 13. IMPORTANT INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)


print(
    """
This mapping was inferred mathematically from the
available anomalous 74283 datasets.

It is NOT yet being claimed as the official mapping
from the original .sys benchmark file.

If the best candidate has substantially stronger
agreement than alternative mappings and behaves
consistently across the four datasets, it provides
strong evidence for the physical port assignment.

Also remember:

A + B = B + A

Therefore the dataset behaviour alone may not be
able to distinguish the names A and B. A complete
swap of the two operands is mathematically equivalent.
"""
)