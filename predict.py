import os
import joblib
import pandas as pd


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "svm_anomaly_classifier.pkl"
)

TEST_DATA_PATH = os.path.join(
    "outputs",
    "held_out_test_set.csv"
)

FEATURES = [
    "i1", "i2", "i3", "i4", "i5",
    "i6", "i7", "i8", "i9",
    "o1", "o2", "o3", "o4", "o5"
]


ANOMALY_INFO = {

    "z2": {
        "location": "Internal node z2",
        "modification": "NAND2 -> AND2"
    },

    "z17": {
        "location": "Internal node z17",
        "modification": "AND5 -> NAND5"
    },

    "z18": {
        "location": "Internal node z18",
        "modification": "AND2 -> NAND2"
    },

    "o1": {
        "location": "Output node o1",
        "modification": "NOR5 -> OR5"
    }
}


# ============================================================
# 2. INFERRED DATASET SIGNAL MAPPING
# ============================================================

#
# INPUT MAPPING USED:
#
# i1 = A3
# i2 = B3
#
# i3 = A2
# i4 = B2
#
# i5 = A1
# i6 = B1
#
# i7 = Cin
#
# i8 = A0
# i9 = B0
#
#
# OUTPUT MAPPING INFERRED FROM DATASET BEHAVIOUR:
#
# o1 = S3
# o2 = Cout
# o3 = S2
# o4 = S1
# o5 = S0
#
#
# IMPORTANT:
#
# This mapping was inferred mathematically from the
# available anomalous 74283 datasets.
#
# It has NOT been directly verified using the
# original benchmark .sys file.
#


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print(
        "\nTrained model was not found:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nRun train_model.py first."
    )

    exit()


model = joblib.load(
    MODEL_PATH
)


# ============================================================
# 4. LOAD HELD-OUT TEST DATASET
# ============================================================

if not os.path.exists(TEST_DATA_PATH):

    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print(
        "\nHeld-out test dataset was not found:"
    )

    print(
        TEST_DATA_PATH
    )

    print(
        "\nRun the updated train_model.py first."
    )

    exit()


test_data = pd.read_csv(
    TEST_DATA_PATH
)


required_columns = (
    FEATURES
    +
    ["anomaly_class"]
)


missing_columns = [
    column
    for column in required_columns
    if column not in test_data.columns
]


if missing_columns:

    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print(
        "\nThe held-out test dataset is missing:"
    )

    print(
        missing_columns
    )

    exit()


test_data = test_data[
    required_columns
].copy()


# ============================================================
# 5. DISPLAY DATASET OBSERVATION
# ============================================================

def display_observation(row):

    print(
        "\nINPUT SIGNALS"
    )

    print(
        "-" * 50
    )


    for i in range(1, 10):

        print(
            f"i{i}="
            f"{int(row[f'i{i}'])}",
            end="   "
        )


        if i % 3 == 0:

            print()


    print(
        "\nOUTPUT SIGNALS"
    )

    print(
        "-" * 50
    )


    for i in range(1, 6):

        print(
            f"o{i}="
            f"{int(row[f'o{i}'])}",
            end="   "
        )


    print()


# ============================================================
# 6. PERFORM ML PREDICTION
# ============================================================

def predict_observation(row):

    observation = pd.DataFrame(
        [[
            row[feature]
            for feature in FEATURES
        ]],
        columns=FEATURES
    )


    prediction = model.predict(
        observation
    )[0]


    return prediction


# ============================================================
# 7. CONVERT PHYSICAL 74283 SIGNALS TO ML FEATURES
# ============================================================

def build_ml_observation(
    A,
    B,
    Cin,
    observed_sum_bits,
    observed_cout
):

    # --------------------------------------------------------
    # Convert A and B to four-bit binary strings
    # --------------------------------------------------------

    A_bits = f"{A:04b}"
    B_bits = f"{B:04b}"


    # --------------------------------------------------------
    # Extract A bits
    # --------------------------------------------------------

    A3 = int(
        A_bits[0]
    )

    A2 = int(
        A_bits[1]
    )

    A1 = int(
        A_bits[2]
    )

    A0 = int(
        A_bits[3]
    )


    # --------------------------------------------------------
    # Extract B bits
    # --------------------------------------------------------

    B3 = int(
        B_bits[0]
    )

    B2 = int(
        B_bits[1]
    )

    B1 = int(
        B_bits[2]
    )

    B0 = int(
        B_bits[3]
    )


    # --------------------------------------------------------
    # Extract observed Sum bits
    # --------------------------------------------------------

    S3 = int(
        observed_sum_bits[0]
    )

    S2 = int(
        observed_sum_bits[1]
    )

    S1 = int(
        observed_sum_bits[2]
    )

    S0 = int(
        observed_sum_bits[3]
    )


    # --------------------------------------------------------
    # Build 14-feature ML observation
    # --------------------------------------------------------
    #
    # INPUT:
    #
    # i1 = A3
    # i2 = B3
    # i3 = A2
    # i4 = B2
    # i5 = A1
    # i6 = B1
    # i7 = Cin
    # i8 = A0
    # i9 = B0
    #
    #
    # OUTPUT:
    #
    # o1 = S3
    # o2 = Cout
    # o3 = S2
    # o4 = S1
    # o5 = S0
    #

    values = {

        "i1": A3,
        "i2": B3,

        "i3": A2,
        "i4": B2,

        "i5": A1,
        "i6": B1,

        "i7": Cin,

        "i8": A0,
        "i9": B0,

        "o1": S3,
        "o2": observed_cout,
        "o3": S2,
        "o4": S1,
        "o5": S0
    }


    observation = pd.Series(
        values
    )


    return observation


# ============================================================
# 8. DISPLAY PHYSICAL -> DATASET MAPPING
# ============================================================

def display_signal_mapping():

    print()
    print("=" * 60)
    print("74283 DATASET SIGNAL MAPPING")
    print("=" * 60)


    print(
        """
INPUT MAPPING

i1 = A3
i2 = B3

i3 = A2
i4 = B2

i5 = A1
i6 = B1

i7 = Cin

i8 = A0
i9 = B0


OUTPUT MAPPING

o1 = S3
o2 = Cout
o3 = S2
o4 = S1
o5 = S0
"""
    )


    print(
        "NOTE:"
    )

    print(
        "This mapping was inferred from dataset "
        "behaviour and has not been directly "
        "verified using the original .sys file."
    )


# ============================================================
# 9. INTERACTIVE 74283 CIRCUIT DIAGNOSIS
# ============================================================

def interactive_diagnosis():

    print()
    print("=" * 60)
    print("       74283 AI CIRCUIT DIAGNOSTIC SYSTEM")
    print("=" * 60)


    # --------------------------------------------------------
    # INPUT A
    # --------------------------------------------------------

    while True:

        try:

            A = int(
                input(
                    "\nEnter A (0-15): "
                )
            )


            if not 0 <= A <= 15:

                raise ValueError


            break


        except ValueError:

            print(
                "Please enter an integer from 0 to 15."
            )


    # --------------------------------------------------------
    # INPUT B
    # --------------------------------------------------------

    while True:

        try:

            B = int(
                input(
                    "Enter B (0-15): "
                )
            )


            if not 0 <= B <= 15:

                raise ValueError


            break


        except ValueError:

            print(
                "Please enter an integer from 0 to 15."
            )


    # --------------------------------------------------------
    # INPUT CIN
    # --------------------------------------------------------

    while True:

        try:

            Cin = int(
                input(
                    "Enter Cin (0/1): "
                )
            )


            if Cin not in [0, 1]:

                raise ValueError


            break


        except ValueError:

            print(
                "Please enter only 0 or 1."
            )


    # --------------------------------------------------------
    # CALCULATE EXPECTED OUTPUT
    # --------------------------------------------------------

    total = (
        A + B + Cin
    )


    expected_sum = (
        total & 0b1111
    )


    expected_cout = (
        1 if total > 15 else 0
    )


    A_bits = (
        f"{A:04b}"
    )

    B_bits = (
        f"{B:04b}"
    )

    expected_sum_bits = (
        f"{expected_sum:04b}"
    )


    # --------------------------------------------------------
    # DISPLAY APPLIED INPUTS
    # --------------------------------------------------------

    print()
    print("-" * 60)
    print("APPLIED INPUTS")
    print("-" * 60)


    print(
        f"A    = {A_bits} ({A})"
    )

    print(
        f"B    = {B_bits} ({B})"
    )

    print(
        f"Cin  = {Cin}"
    )


    # --------------------------------------------------------
    # DISPLAY EXPECTED OUTPUT
    # --------------------------------------------------------

    print()
    print("EXPECTED 74283 OUTPUT")
    print("-" * 60)


    print(
        f"Sum  = {expected_sum_bits}"
    )

    print(
        f"Cout = {expected_cout}"
    )


    # --------------------------------------------------------
    # ENTER OBSERVED SUM
    # --------------------------------------------------------

    while True:

        observed_sum_bits = input(
            "\nEnter observed Sum (4-bit): "
        ).strip()


        if (
            len(observed_sum_bits) == 4
            and
            all(
                bit in "01"
                for bit in observed_sum_bits
            )
        ):

            break


        print(
            "Please enter exactly four binary bits, "
            "for example 0110."
        )


    # --------------------------------------------------------
    # ENTER OBSERVED COUT
    # --------------------------------------------------------

    while True:

        try:

            observed_cout = int(
                input(
                    "Enter observed Cout (0/1): "
                )
            )


            if observed_cout not in [0, 1]:

                raise ValueError


            break


        except ValueError:

            print(
                "Please enter only 0 or 1."
            )


    # --------------------------------------------------------
    # DISPLAY COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CIRCUIT TEST RESULT")
    print("=" * 60)


    print(
        f"\nExpected : "
        f"Cout={expected_cout}, "
        f"Sum={expected_sum_bits}"
    )


    print(
        f"Observed : "
        f"Cout={observed_cout}, "
        f"Sum={observed_sum_bits}"
    )


    # --------------------------------------------------------
    # NORMAL OUTPUT
    # --------------------------------------------------------

    if (
        observed_sum_bits == expected_sum_bits
        and
        observed_cout == expected_cout
    ):

        print()
        print("-" * 60)
        print("RESULT: NORMAL CIRCUIT BEHAVIOUR")
        print("-" * 60)


        print(
            "\nThe observed circuit output matches "
            "the expected 74283 output."
        )


        print(
            "\nNo externally visible output anomaly "
            "was detected for this input combination."
        )


        return


    # --------------------------------------------------------
    # ANOMALY DETECTED
    # --------------------------------------------------------

    print()
    print("-" * 60)
    print("RESULT: OUTPUT ANOMALY DETECTED")
    print("-" * 60)


    print(
        "\nThe observed circuit output does not "
        "match the expected 74283 output."
    )


    print(
        "\nRunning ML-based anomaly classification..."
    )


    # --------------------------------------------------------
    # BUILD ML OBSERVATION
    # --------------------------------------------------------

    observation = build_ml_observation(
        A=A,
        B=B,
        Cin=Cin,
        observed_sum_bits=observed_sum_bits,
        observed_cout=observed_cout
    )


    # --------------------------------------------------------
    # DISPLAY ML OBSERVATION
    # --------------------------------------------------------

    print()
    print("ML INPUT OBSERVATION")
    print("-" * 60)


    display_observation(
        observation
    )


    # --------------------------------------------------------
    # RUN SVM
    # --------------------------------------------------------

    prediction = predict_observation(
        observation
    )


    info = ANOMALY_INFO[
        prediction
    ]


    # --------------------------------------------------------
    # DISPLAY ML RESULT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ML ANOMALY CLASSIFICATION")
    print("=" * 60)


    print(
        f"\nClosest known anomaly : "
        f"{prediction}"
    )


    print(
        f"Likely location       : "
        f"{info['location']}"
    )


    print(
        f"Gate modification     : "
        f"{info['modification']}"
    )


    print()
    print("-" * 60)
    print("IMPORTANT")
    print("-" * 60)


    print(
        """
The SVM was trained to distinguish only four
known anomaly classes:

z2  : NAND2 -> AND2
z17 : AND5 -> NAND5
z18 : AND2 -> NAND2
o1  : NOR5 -> OR5

Therefore, the prediction represents the known
anomaly class whose learned I/O behaviour most
closely matches the observation.

A completely different or unseen physical fault
may still be assigned to one of these four classes.
"""
    )


# ============================================================
# 10. TEST ONE HELD-OUT OBSERVATION
# ============================================================

def random_test_sample():

    sample = test_data.sample(
        n=1
    ).iloc[0]


    actual = sample[
        "anomaly_class"
    ]


    predicted = predict_observation(
        sample
    )


    print()
    print("=" * 60)
    print("UNSEEN TEST DATASET OBSERVATION")
    print("=" * 60)


    print(
        "\nThis observation comes from the "
        "held-out test dataset."
    )


    print(
        "It was NOT used to train the SVM."
    )


    display_observation(
        sample
    )


    print()
    print("ML CLASSIFICATION")
    print("-" * 50)


    print(
        f"Actual anomaly    : "
        f"{actual}"
    )


    print(
        f"Predicted anomaly : "
        f"{predicted}"
    )


    print(
        f"Predicted location: "
        f"{ANOMALY_INFO[predicted]['location']}"
    )


    print(
        f"Modification      : "
        f"{ANOMALY_INFO[predicted]['modification']}"
    )


    if actual == predicted:

        print(
            "\nRESULT: CORRECT CLASSIFICATION"
        )

    else:

        print(
            "\nRESULT: MISCLASSIFIED"
        )


# ============================================================
# 11. TEST MULTIPLE HELD-OUT OBSERVATIONS
# ============================================================

def multiple_test_samples():

    print()
    print("=" * 60)
    print("HELD-OUT TEST DATASET EVALUATION")
    print("=" * 60)


    print(
        f"\nAvailable held-out observations: "
        f"{len(test_data)}"
    )


    while True:

        try:

            number = int(
                input(
                    "\nHow many unseen samples "
                    "do you want to test? "
                )
            )


            if number <= 0:

                raise ValueError


            break


        except ValueError:

            print(
                "Please enter a positive integer."
            )


    number = min(
        number,
        len(test_data)
    )


    samples = test_data.sample(
        n=number,
        random_state=None
    )


    correct = 0


    print()
    print("-" * 60)


    for count, (_, sample) in enumerate(
        samples.iterrows(),
        start=1
    ):

        actual = sample[
            "anomaly_class"
        ]


        predicted = predict_observation(
            sample
        )


        if actual == predicted:

            correct += 1


        print(
            f"{count:3d}. "
            f"Actual: {actual:4s}   "
            f"Predicted: {predicted:4s}"
        )


    accuracy = (
        correct /
        len(samples)
    ) * 100


    print()
    print("-" * 60)


    print(
        f"Correct predictions : "
        f"{correct}/{len(samples)}"
    )


    print(
        f"Evaluation accuracy : "
        f"{accuracy:.2f}%"
    )


    if number == len(test_data):

        print(
            "\nThis evaluated the complete "
            "held-out test dataset."
        )


# ============================================================
# 12. PROJECT / MODEL INFORMATION
# ============================================================

def model_information():

    print()
    print("=" * 60)
    print("PROJECT INFORMATION")
    print("=" * 60)


    print(
        """
Project:
Machine Learning-Based Anomaly
Classification in Digital Circuits

Circuit:
74283 4-bit binary adder

Dataset Samples:
1680

Training Samples:
1344

Held-Out Test Samples:
336

Machine Learning Features:
9 input signals
5 output signals
14 total features

Known Anomaly Classes:

z2
    Internal node z2
    NAND2 -> AND2

z17
    Internal node z17
    AND5 -> NAND5

z18
    Internal node z18
    AND2 -> NAND2

o1
    Output node o1
    NOR5 -> OR5


Models Compared:

Logistic Regression : 43.75%
Decision Tree       : 51.19%
Random Forest       : 51.49%
SVM (RBF)           : 70.24%


Final Model:
Support Vector Machine
RBF Kernel

Held-Out Test Accuracy:
70.24%

Correct Test Predictions:
236 / 336
"""
    )


    print()
    print("-" * 60)
    print("DATASET SIGNAL MAPPING")
    print("-" * 60)


    print(
        """
Inputs:

i1 = A3
i2 = B3
i3 = A2
i4 = B2
i5 = A1
i6 = B1
i7 = Cin
i8 = A0
i9 = B0


Outputs:

o1 = S3
o2 = Cout
o3 = S2
o4 = S1
o5 = S0
"""
    )


    print(
        "Mapping status:"
    )


    print(
        "Inferred from the available anomalous "
        "dataset behaviour."
    )


    print(
        "The original benchmark .sys file has not "
        "been available for direct verification."
    )


    print()
    print("-" * 60)
    print("LIMITATION")
    print("-" * 60)


    print(
        """
The ML classifier recognizes only the four
anomaly classes represented in its training data.

An unseen physical fault may therefore be assigned
to one of the four known classes.

The ML output should be interpreted as the closest
known anomaly pattern rather than definitive
physical fault verification.
"""
    )


# ============================================================
# 13. MAIN MENU
# ============================================================

while True:

    print()
    print()
    print("=" * 60)
    print("     74283 DIGITAL CIRCUIT ANOMALY CLASSIFIER")
    print("=" * 60)


    print(
        """
1. Interactive 74283 circuit diagnosis
2. Test one unseen dataset sample
3. Test multiple unseen dataset samples
4. Show project/model information
5. Show inferred signal mapping
6. Exit
"""
    )


    choice = input(
        "Enter your choice: "
    ).strip()


    # --------------------------------------------------------
    # OPTION 1
    # --------------------------------------------------------

    if choice == "1":

        interactive_diagnosis()


    # --------------------------------------------------------
    # OPTION 2
    # --------------------------------------------------------

    elif choice == "2":

        random_test_sample()


    # --------------------------------------------------------
    # OPTION 3
    # --------------------------------------------------------

    elif choice == "3":

        multiple_test_samples()


    # --------------------------------------------------------
    # OPTION 4
    # --------------------------------------------------------

    elif choice == "4":

        model_information()


    # --------------------------------------------------------
    # OPTION 5
    # --------------------------------------------------------

    elif choice == "5":

        display_signal_mapping()


    # --------------------------------------------------------
    # OPTION 6
    # --------------------------------------------------------

    elif choice == "6":

        print()
        print(
            "Exiting anomaly classifier."
        )

        break


    # --------------------------------------------------------
    # INVALID OPTION
    # --------------------------------------------------------

    else:

        print()
        print(
            "Invalid choice. Select 1-6."
        )