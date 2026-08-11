import os
import joblib
import pandas as pd
import streamlit as st
import urllib.parse
import urllib.request


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="74283 AI Circuit Diagnostic System",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# 2. PROJECT SETTINGS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "svm_anomaly_classifier.pkl"
)

# ESP32 LCD SETTINGS
# Enter the IP address shown on the ESP32 LCD / Serial Monitor.
ESP32_DEFAULT_IP = ""
ESP32_TIMEOUT = 3

def send_to_esp32(fault, modification, esp32_ip):
    """Send the ML diagnosis to the ESP32 LCD over Wi-Fi."""

    if not esp32_ip:
        return False, "ESP32 IP address is empty."

    params = urllib.parse.urlencode({
        "fault": str(fault),
        "mod": str(modification)
    })

    url = f"http://{esp32_ip}/diagnosis?{params}"

    try:
        with urllib.request.urlopen(url, timeout=ESP32_TIMEOUT) as response:
            response_text = response.read().decode("utf-8", errors="ignore")

        return True, response_text

    except Exception as error:
        return False, str(error)


FEATURES = [
    "i1", "i2", "i3", "i4", "i5",
    "i6", "i7", "i8", "i9",
    "o1", "o2", "o3", "o4", "o5"
]


ANOMALY_INFO = {

    "z2": {
        "location": "Internal node z2",
        "modification": "NAND2 → AND2"
    },

    "z17": {
        "location": "Internal node z17",
        "modification": "AND5 → NAND5"
    },

    "z18": {
        "location": "Internal node z18",
        "modification": "AND2 → NAND2"
    },

    "o1": {
        "location": "Output node o1",
        "modification": "NOR5 → OR5"
    }
}


# ============================================================
# 3. DATASET SIGNAL MAPPING
# ============================================================

# INPUT MAPPING
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
# OUTPUT MAPPING
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
# This mapping was inferred mathematically from
# the available anomalous 74283 datasets.
#
# The original benchmark .sys file has not been
# available for direct verification.


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


if not os.path.exists(MODEL_PATH):

    st.error(
        "Trained SVM model was not found. "
        "Run train_model.py first."
    )

    st.stop()


model = load_model()


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.title(
    "⚡ 74283 AI Diagnostic"
)


page = st.sidebar.radio(
    "Navigation",
    [
        "Circuit Diagnosis",
        "ML Performance",
        "Project Information"
    ]
)


st.sidebar.divider()


st.sidebar.write(
    "**Final Model:** SVM (RBF)"
)

st.sidebar.write(
    "**Held-Out Test Accuracy:** 70.24%"
)

st.sidebar.write(
    "**Known Anomaly Classes:** 4"
)


st.sidebar.divider()

st.sidebar.subheader("ESP32 LCD")

esp32_ip = st.sidebar.text_input(
    "ESP32 IP Address",
    value=ESP32_DEFAULT_IP,
    placeholder="192.168.1.105",
    help="Enter the IP address displayed by your ESP32. The PC and ESP32 must be on the same Wi-Fi network."
).strip()

st.sidebar.caption(
    "The ML diagnosis will be sent to the physical LCD over Wi-Fi."
)


# ============================================================
# PAGE 1 — CIRCUIT DIAGNOSIS
# ============================================================

if page == "Circuit Diagnosis":

    st.title(
        "⚡ 74283 Digital Circuit Anomaly Classifier"
    )


    st.caption(
        "Machine Learning-Based Fault Diagnosis "
        "for a 4-bit Binary Adder"
    )


    # --------------------------------------------------------
    # OPERATING MODE
    # --------------------------------------------------------

    st.subheader(
        "Operating Mode"
    )


    input_mode = st.radio(
        "Select how the circuit output is obtained",
        [
            "Manual Mode",
            "Hardware Mode (ESP32)"
        ],
        horizontal=True
    )


    if input_mode == "Manual Mode":

        st.info(
            "Manual Mode: Enter the observed Sum and "
            "Cout manually. This mode is used for "
            "software testing and dataset validation."
        )


    else:

        st.info(
            "Hardware Mode: The observed Sum and Cout "
            "will be obtained automatically from an "
            "ESP32 connected to the circuit under test."
        )


    st.divider()


    # --------------------------------------------------------
    # CIRCUIT INPUTS
    # --------------------------------------------------------

    st.subheader(
        "1. Apply Circuit Inputs"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        A = st.number_input(
            "Input A",
            min_value=0,
            max_value=15,
            value=11,
            step=1
        )


    with col2:

        B = st.number_input(
            "Input B",
            min_value=0,
            max_value=15,
            value=2,
            step=1
        )


    with col3:

        Cin = st.selectbox(
            "Carry Input (Cin)",
            [0, 1],
            index=1
        )


    A = int(A)
    B = int(B)
    Cin = int(Cin)


    A_bits = f"{A:04b}"
    B_bits = f"{B:04b}"


    st.write(
        f"**Binary inputs:** "
        f"A = `{A_bits}` &nbsp;&nbsp; "
        f"B = `{B_bits}` &nbsp;&nbsp; "
        f"Cin = `{Cin}`"
    )


    # --------------------------------------------------------
    # EXPECTED OUTPUT
    # --------------------------------------------------------

    expected_total = (
        A + B + Cin
    )


    expected_sum = (
        expected_total & 0b1111
    )


    expected_cout = (
        1 if expected_total > 15 else 0
    )


    expected_sum_bits = (
        f"{expected_sum:04b}"
    )


    st.subheader(
        "2. Expected 74283 Output"
    )


    expected_col1, expected_col2 = (
        st.columns(2)
    )


    with expected_col1:

        st.metric(
            "Expected Sum",
            expected_sum_bits
        )


    with expected_col2:

        st.metric(
            "Expected Cout",
            expected_cout
        )


    st.caption(
        "Calculated using the ideal 74283 function: "
        "A + B + Cin."
    )


    st.divider()


    # ========================================================
    # MANUAL MODE
    # ========================================================

    if input_mode == "Manual Mode":

        st.subheader(
            "3. Enter Observed Circuit Output"
        )


        obs_col1, obs_col2 = st.columns(2)


        with obs_col1:

            observed_sum_bits = st.text_input(
                "Observed Sum (4-bit)",
                value=expected_sum_bits,
                max_chars=4
            )


        with obs_col2:

            observed_cout = st.selectbox(
                "Observed Cout",
                [0, 1],
                index=expected_cout
            )


        observed_cout = int(
            observed_cout
        )


        diagnose = st.button(
            "🔍 Diagnose Circuit",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # HARDWARE MODE
    # ========================================================

    else:

        st.subheader(
            "3. Hardware Circuit Output"
        )


        st.warning(
            "ESP32 communication has not yet been "
            "implemented. This interface is prepared "
            "for the hardware integration stage."
        )


        st.code(
            """
74283 / Circuit Under Test
          |
          | S3 S2 S1 S0 Cout
          v
        ESP32
          |
          | Serial / Wi-Fi
          v
   Python / Streamlit
          |
          v
Expected vs Observed
          |
          v
      SVM Model
          |
          v
  Fault Classification
"""
        )


        st.write(
            "### ESP32 Status"
        )


        status_col1, status_col2 = (
            st.columns(2)
        )


        with status_col1:

            st.metric(
                "Connection",
                "Not Connected"
            )


        with status_col2:

            st.metric(
                "Hardware Status",
                "Waiting"
            )


        st.write(
            "### Hardware Measurements"
        )


        hardware_col1, hardware_col2 = (
            st.columns(2)
        )


        with hardware_col1:

            st.metric(
                "Observed Sum from ESP32",
                "Waiting..."
            )


        with hardware_col2:

            st.metric(
                "Observed Cout from ESP32",
                "Waiting..."
            )


        st.button(
            "🔌 Connect ESP32",
            disabled=True,
            use_container_width=True
        )


        st.caption(
            "The connection button will be enabled "
            "after ESP32 serial communication is added."
        )


        observed_sum_bits = None
        observed_cout = None
        diagnose = False


    # ========================================================
    # DIAGNOSIS
    # ========================================================

    if diagnose:

        observed_sum_bits = (
            observed_sum_bits.strip()
        )


        # ----------------------------------------------------
        # VALIDATE OBSERVED SUM
        # ----------------------------------------------------

        if (
            len(observed_sum_bits) != 4
            or
            not all(
                bit in "01"
                for bit in observed_sum_bits
            )
        ):

            st.error(
                "Observed Sum must contain exactly "
                "four binary bits, for example 0110."
            )

            st.stop()


        # ----------------------------------------------------
        # CIRCUIT TEST RESULT
        # ----------------------------------------------------

        st.subheader(
            "4. Circuit Test Result"
        )


        result_col1, result_col2 = (
            st.columns(2)
        )


        with result_col1:

            st.write(
                "### Expected"
            )

            st.code(
                f"Sum  = {expected_sum_bits}\n"
                f"Cout = {expected_cout}"
            )


        with result_col2:

            st.write(
                "### Observed"
            )

            st.code(
                f"Sum  = {observed_sum_bits}\n"
                f"Cout = {observed_cout}"
            )


        # ----------------------------------------------------
        # NORMAL CIRCUIT
        # ----------------------------------------------------

        if (
            observed_sum_bits == expected_sum_bits
            and
            observed_cout == expected_cout
        ):

            st.success(
                "✅ NORMAL CIRCUIT BEHAVIOUR"
            )


            st.write(
                "The observed circuit output matches "
                "the ideal 74283 output."
            )


            st.write(
                "**No externally visible output "
                "anomaly was detected for this "
                "input combination.**"
            )


            # ------------------------------------------------
            # SEND NORMAL STATUS TO ESP32 LCD
            # ------------------------------------------------

            if esp32_ip:

                lcd_ok, lcd_message = send_to_esp32(
                    fault="NORMAL",
                    modification="74283-OK",
                    esp32_ip=esp32_ip
                )

                if lcd_ok:
                    st.success("📺 ESP32 LCD updated: NORMAL")
                else:
                    st.warning(
                        "ESP32 LCD update failed: "
                        f"{lcd_message}"
                    )


        # ----------------------------------------------------
        # ANOMALY DETECTED
        # ----------------------------------------------------

        else:

            st.warning(
                "⚠️ OUTPUT ANOMALY DETECTED"
            )


            st.write(
                "The observed circuit output does not "
                "match the expected 74283 output."
            )


            st.write(
                "The observed I/O pattern will now be "
                "classified using the trained SVM."
            )


            # ------------------------------------------------
            # INPUT BIT EXTRACTION
            # ------------------------------------------------

            A3 = int(A_bits[0])
            A2 = int(A_bits[1])
            A1 = int(A_bits[2])
            A0 = int(A_bits[3])


            B3 = int(B_bits[0])
            B2 = int(B_bits[1])
            B1 = int(B_bits[2])
            B0 = int(B_bits[3])


            # ------------------------------------------------
            # OBSERVED OUTPUT BITS
            # ------------------------------------------------

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


            # ------------------------------------------------
            # BUILD 14-FEATURE ML OBSERVATION
            # ------------------------------------------------

            observation = pd.DataFrame(
                [[
                    A3,             # i1
                    B3,             # i2
                    A2,             # i3
                    B2,             # i4
                    A1,             # i5
                    B1,             # i6
                    Cin,            # i7
                    A0,             # i8
                    B0,             # i9

                    S3,             # o1
                    observed_cout,  # o2
                    S2,             # o3
                    S1,             # o4
                    S0              # o5
                ]],

                columns=FEATURES
            )


            # ------------------------------------------------
            # ML PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                observation
            )[0]


            info = ANOMALY_INFO[
                prediction
            ]



            # ------------------------------------------------
            # SEND ML DIAGNOSIS TO ESP32 LCD
            # ------------------------------------------------

            lcd_status = None

            if esp32_ip:

                lcd_modification = (
                    info["modification"]
                    .replace("→", "->")
                    .replace(" ", "")
                )

                lcd_ok, lcd_message = send_to_esp32(
                    fault=prediction,
                    modification=lcd_modification,
                    esp32_ip=esp32_ip
                )

                if lcd_ok:
                    lcd_status = "📺 ESP32 LCD updated successfully."
                else:
                    lcd_status = (
                        "⚠️ ESP32 LCD update failed: "
                        f"{lcd_message}"
                    )

            else:
                lcd_status = (
                    "ℹ️ Enter the ESP32 IP address in the "
                    "sidebar to send this diagnosis to the LCD."
                )


            # ------------------------------------------------
            # DISPLAY CLASSIFICATION
            # ------------------------------------------------

            st.subheader(
                "5. ML Fault Classification"
            )


            pred_col1, pred_col2, pred_col3 = (
                st.columns(3)
            )


            with pred_col1:

                st.metric(
                    "Closest Known Anomaly Class",
                    prediction
                )


            with pred_col2:

                st.metric(
                    "Likely Fault Location",
                    info["location"]
                )


            with pred_col3:

                st.metric(
                    "Likely Gate Modification",
                    info["modification"]
                )


            st.info(
                "The observed I/O behaviour most "
                "closely matches the predicted "
                "anomaly class learned by the SVM. "
                "This is classification among four "
                "known anomaly classes and is not "
                "definitive physical fault verification."
            )


            if lcd_status:
                st.write(lcd_status)


            # ------------------------------------------------
            # ML OBSERVATION
            # ------------------------------------------------

            with st.expander(
                "View 14-feature ML observation"
            ):

                st.dataframe(
                    observation,
                    use_container_width=True,
                    hide_index=True
                )


            # ------------------------------------------------
            # SIGNAL MAPPING
            # ------------------------------------------------

            with st.expander(
                "View dataset signal mapping"
            ):

                st.write(
                    """
**Input mapping used**

- i1 = A3
- i2 = B3
- i3 = A2
- i4 = B2
- i5 = A1
- i6 = B1
- i7 = Cin
- i8 = A0
- i9 = B0

**Output mapping inferred from dataset behaviour**

- o1 = S3
- o2 = Cout
- o3 = S2
- o4 = S1
- o5 = S0
"""
                )


                st.caption(
                    "The mapping was inferred from "
                    "the available anomalous datasets. "
                    "The original .sys benchmark file "
                    "has not been available for direct "
                    "official verification."
                )


# ============================================================
# PAGE 2 — ML PERFORMANCE
# ============================================================

elif page == "ML Performance":

    st.title(
        "📊 Machine Learning Performance"
    )


    st.write(
        "The machine-learning models were evaluated "
        "using the same stratified 80/20 train-test "
        "split."
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    col1.metric(
        "Total Samples",
        "1,680"
    )


    col2.metric(
        "Training Samples",
        "1,344"
    )


    col3.metric(
        "Held-Out Test Samples",
        "336"
    )


    col4.metric(
        "SVM Accuracy",
        "70.24%"
    )


    st.divider()


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader(
        "Model Comparison"
    )


    comparison_path = os.path.join(
        "outputs",
        "model_comparison.png"
    )


    if os.path.exists(
        comparison_path
    ):

        st.image(
            comparison_path,
            caption="ML Model Accuracy Comparison"
        )

    else:

        st.warning(
            "model_comparison.png was not found. "
            "Run train_model.py."
        )


    model_table = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "SVM (RBF)"
        ],

        "Held-Out Test Accuracy": [
            "43.75%",
            "51.19%",
            "51.49%",
            "70.24%"
        ]
    })


    st.dataframe(
        model_table,
        use_container_width=True,
        hide_index=True
    )


    st.success(
        "SVM with the RBF kernel achieved the "
        "highest held-out test accuracy and was "
        "therefore selected as the final classifier."
    )


    st.divider()


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "SVM Confusion Matrix"
    )


    confusion_path = os.path.join(
        "outputs",
        "svm_confusion_matrix.png"
    )


    if os.path.exists(
        confusion_path
    ):

        st.image(
            confusion_path,
            caption=(
                "Confusion Matrix on "
                "336 Held-Out Test Samples"
            )
        )

    else:

        st.warning(
            "svm_confusion_matrix.png was not found. "
            "Run train_model.py."
        )


    st.write(
        """
The SVM correctly classified **236 of the 336**
held-out test observations.

**Overall accuracy: 70.24%**

The major confusion patterns are:

- **o1 ↔ z17**
- **z18 ↔ z2**

The model does not directly observe the internal
fault nodes. Instead, it learns differences in
external circuit input/output behaviour.
"""
    )


    st.divider()


    # --------------------------------------------------------
    # PER-CLASS PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "Per-Class Performance"
    )


    performance_table = pd.DataFrame({

        "Class": [
            "o1",
            "z17",
            "z18",
            "z2"
        ],

        "Precision": [
            "87.04%",
            "46.08%",
            "100.00%",
            "66.96%"
        ],

        "Recall": [
            "46.08%",
            "87.04%",
            "63.11%",
            "100.00%"
        ],

        "F1-Score": [
            "60.26%",
            "60.26%",
            "77.38%",
            "80.21%"
        ],

        "Support": [
            102,
            54,
            103,
            77
        ]
    })


    st.dataframe(
        performance_table,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        """
Interpretation:

• z2 achieved 100% recall, meaning all held-out
z2 observations were detected as z2.

• z18 achieved 100% precision, meaning observations
predicted as z18 were correct in the held-out test.

• o1 and z17 show stronger confusion with each
other because their external I/O fault behaviour
can overlap.
"""
    )


# ============================================================
# PAGE 3 — PROJECT INFORMATION
# ============================================================

elif page == "Project Information":

    st.title(
        "ℹ️ Project Information"
    )


    st.header(
        "Machine Learning-Based Anomaly "
        "Classification in Digital Circuits"
    )


    st.write(
        """
This project demonstrates machine-learning-based
anomaly classification using data generated from
modified 74283 digital circuits.

The 74283 performs 4-bit binary addition:

**A + B + Cin → Sum + Cout**

The application calculates the ideal output expected
from a 4-bit binary adder and compares it with the
observed circuit output.

If the observed output differs from the ideal output,
the 14-feature circuit observation is supplied to a
trained Support Vector Machine classifier.
"""
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.subheader(
        "Dataset"
    )


    st.write(
        """
The combined dataset contains **1,680 observations**
from four known anomaly classes.

Each machine-learning observation contains:

- 9 circuit input signals: i1-i9
- 5 circuit output signals: o1-o5
- 14 total ML features
"""
    )


    # --------------------------------------------------------
    # ANOMALY CLASSES
    # --------------------------------------------------------

    st.subheader(
        "Known Anomaly Classes"
    )


    anomaly_table = pd.DataFrame({

        "Class": [
            "z2",
            "z17",
            "z18",
            "o1"
        ],

        "Location": [
            "Internal node z2",
            "Internal node z17",
            "Internal node z18",
            "Output node o1"
        ],

        "Gate Modification": [
            "NAND2 → AND2",
            "AND5 → NAND5",
            "AND2 → NAND2",
            "NOR5 → OR5"
        ]
    })


    st.dataframe(
        anomaly_table,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # SIGNAL MAPPING
    # --------------------------------------------------------

    st.subheader(
        "74283 Dataset Signal Mapping"
    )


    mapping_table = pd.DataFrame({

        "Dataset Signal": [
            "i1",
            "i2",
            "i3",
            "i4",
            "i5",
            "i6",
            "i7",
            "i8",
            "i9",
            "o1",
            "o2",
            "o3",
            "o4",
            "o5"
        ],

        "Interpreted Signal": [
            "A3",
            "B3",
            "A2",
            "B2",
            "A1",
            "B1",
            "Cin",
            "A0",
            "B0",
            "S3",
            "Cout",
            "S2",
            "S1",
            "S0"
        ]
    })


    st.dataframe(
        mapping_table,
        use_container_width=True,
        hide_index=True
    )


    st.warning(
        "The signal mapping was inferred "
        "mathematically from the available anomalous "
        "datasets by maximizing agreement with ideal "
        "74283 arithmetic. The original .sys benchmark "
        "file has not been available for direct "
        "verification of the official port names."
    )


    # --------------------------------------------------------
    # ML PIPELINE
    # --------------------------------------------------------

    st.subheader(
        "Machine Learning Pipeline"
    )


    st.code(
        """
4 Anomaly Dataset Files
          |
          v
  Combine + Label Data
          |
          v
  1,680 Observations
          |
          v
 Stratified 80/20 Split
       /        \\
      /          \\
1,344 Train     336 Test
      |             |
      v             |
Train 4 Models      |
      |             |
      v             |
 SVM (RBF) <--------+
      |
      v
236 / 336 Correct
      |
      v
70.24% Test Accuracy
      |
      v
Interactive Diagnosis
"""
    )


    # --------------------------------------------------------
    # CURRENT SOFTWARE FLOW
    # --------------------------------------------------------

    st.subheader(
        "Current Software Diagnosis Flow"
    )


    st.code(
        """
A, B, Cin
    |
    v
Calculate Ideal 74283 Output
    |
    v
Expected Sum / Cout
    |
    v
Observed Sum / Cout
    |
    v
Expected vs Observed
    |
    +-----------------------+
    |                       |
    v                       v
 MATCH                   MISMATCH
    |                       |
    v                       v
 NORMAL                  ANOMALY
                            |
                            v
                Convert to i1-i9/o1-o5
                            |
                            v
                        SVM (RBF)
                            |
                            v
                  Closest Known Fault
"""
    )


    # --------------------------------------------------------
    # HARDWARE EXTENSION
    # --------------------------------------------------------

    st.subheader(
        "Planned IoT / Hardware Extension"
    )


    st.code(
        """
       74283 / Circuit Under Test
                  |
          S3 S2 S1 S0 Cout
                  |
                  v
                ESP32
                  |
          Serial / Wi-Fi
                  |
                  v
          Python / Streamlit
                  |
                  v
        Expected vs Observed
                  |
                  v
              SVM Model
                  |
                  v
      z2 / z17 / z18 / o1
"""
    )


    st.info(
        "The ESP32 LCD integration is now active. The "
        "current system sends the ML diagnosis from "
        "Streamlit to the ESP32 over Wi-Fi, and the ESP32 "
        "displays the diagnosis on the physical I²C LCD. "
        "Automatic reading of the 74283 circuit outputs "
        "from the ESP32 is a separate future hardware stage."
    )


    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.subheader(
        "Project Limitations"
    )


    st.warning(
        """
This project is a proof-of-concept anomaly
classification system.

The trained model recognizes only the four anomaly
classes represented in the training dataset.

A completely different or previously unseen physical
fault may still be classified as one of these four
known classes.

The ML classification therefore represents the
closest learned anomaly pattern rather than
definitive physical fault verification.

The physical dataset signal mapping was inferred
from circuit behaviour because the original
benchmark .sys file was unavailable.
"""
    )