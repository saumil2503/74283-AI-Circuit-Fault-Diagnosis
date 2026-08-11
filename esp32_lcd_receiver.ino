#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ============================================================
// WIFI SETTINGS
// ============================================================

const char* WIFI_SSID = "Enter Wifi Name";
const char* WIFI_PASSWORD = "Enter Wifi Password";

// ============================================================
// LCD
// ============================================================

LiquidCrystal_I2C lcd(0x27, 16, 2);

// ESP32 I2C pins
#define SDA_PIN 21
#define SCL_PIN 22

// ============================================================
// WEB SERVER
// ============================================================

WebServer server(80);

// ============================================================
// LCD DISPLAY FUNCTION
// ============================================================

void displayMessage(String line1, String line2) {

  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print(line1.substring(0, 16));

  lcd.setCursor(0, 1);
  lcd.print(line2.substring(0, 16));
}

// ============================================================
// HOME PAGE
// ============================================================

void handleRoot() {

  String message =
    "ESP32 PROJECT 1\n"
    "WiFi connection OK";

  server.send(200, "text/plain", message);
}

// ============================================================
// DIAGNOSIS ENDPOINT
//
// Example:
//
// http://ESP32_IP/diagnosis?fault=z18&mod=AND2-NAND2
// ============================================================

void handleDiagnosis() {

  String fault = server.arg("fault");
  String modification = server.arg("mod");

  if (fault.length() == 0) {
    fault = "UNKNOWN";
  }

  if (modification.length() == 0) {
    modification = "UNKNOWN";
  }

  // First LCD screen
  displayMessage(
    "ANOMALY: " + fault,
    modification
  );

  // Print received data to Serial Monitor
  Serial.println();
  Serial.println("================================");
  Serial.println("DIAGNOSIS RECEIVED");
  Serial.println("================================");
  Serial.print("Fault: ");
  Serial.println(fault);
  Serial.print("Modification: ");
  Serial.println(modification);
  Serial.println("================================");

  server.send(
    200,
    "text/plain",
    "Diagnosis received by ESP32"
  );
}

// ============================================================
// WIFI CONNECTION
// ============================================================

void connectWiFi() {

  Serial.println();
  Serial.println("Connecting to Wi-Fi...");

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;

  while (WiFi.status() != WL_CONNECTED) {

    delay(500);

    Serial.print(".");

    attempts++;

    if (attempts >= 40) {

      Serial.println();
      Serial.println("Wi-Fi connection failed.");

      displayMessage(
        "WIFI FAILED",
        "Check settings"
      );

      return;
    }
  }

  Serial.println();
  Serial.println("================================");
  Serial.println("WIFI CONNECTED");
  Serial.println("================================");

  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());

  Serial.println("================================");

  displayMessage(
    "WIFI CONNECTED",
    WiFi.localIP().toString()
  );
}

// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  delay(1000);

  // Initialize I2C
  Wire.begin(SDA_PIN, SCL_PIN);

  // Initialize LCD
  lcd.init();
  lcd.backlight();

  displayMessage(
    "PROJECT 1",
    "Starting..."
  );

  delay(1500);

  // Connect Wi-Fi
  connectWiFi();

  // Start web server
  server.on("/", handleRoot);

  server.on("/diagnosis", handleDiagnosis);

  server.begin();

  Serial.println();
  Serial.println("Web server started.");

  if (WiFi.status() == WL_CONNECTED) {

    Serial.print("Open this address: http://");
    Serial.println(WiFi.localIP());

  }
}

// ============================================================
// LOOP
// ============================================================

void loop() {

  server.handleClient();

}