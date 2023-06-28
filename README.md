# FrauenML4BApp
VoiceChoice ist ein Python-Programm zur Extraktion von Audio-Merkmalen und zur Klassifizierung von Audiodaten. 
Das Programm verwendet verschiedene Bibliotheken und Algorithmen, um die Audiodaten zu analysieren und Muster zu erkennen. 
Es ist darauf ausgelegt, Sprachdaten zu verarbeiten und die Aufnahme von Frauen- und Männerstimmen zu unterscheiden.

# Hintergrundinformationen
Die Spracherkennung basiert auf einem neuronalen Netzwerk, das mit TensorFlow trainiert wurde. Das Modell wurde auf einem umfangreichen
Datensatz von Sprachaufnahmen mit bekannten Geschlechtern trainiert, um Muster und Merkmale zu erlernen, die auf das Geschlecht der
sprechenden Person hinweisen.

Das Modell verwendet die Bibliothek Librosa, um die Audioaufnahme in eine repräsentative Darstellung zu transformieren, die dann vom
neuronalen Netzwerk analysiert wird. Die Ergebnisse werden an die Streamlit App zurückgegeben und auf dem Bildschirm angezeigt.

# Voraussetzungen
Um den Recognizer verwenden zu können, müssen die folgenden Voraussetzungen erfüllt sein:
- Python 3.7 oder höher
- Die erforderlichen Python-Bibliotheken, die in der Datei requirements.txt aufgeführt sind.

# Verwendung
1. Starte das Programm, indem du die Datei Recognizer.py ausführst
2. Lade eine Audiodatei hoch, indem du den "Wählen Sie eine Datei zum Hochladen aus" Button verwendest.
3. Das Programm extrahiert automatisch Merkmale aus der Audiodatei und führt eine Klassifizierung durch.
4. Die Ergebnisse werden angezeigt, einschließlich der Klassifikation (Frau/Mann) und weiterer Informationen zu den extrahierten Merkmalen.

# Mitwirken
Wenn du Fehler entdeckst, Verbesserungsvorschläge hast oder zur Entwicklung beitragen möchtest, 
freuen wir uns über deine Mitarbeit. Bitte eröffne ein Issue oder sende uns einen Pull Request auf GitHub.

Wir hoffen, dass diese App dir einen Einblick in die Spracherkennung und die Anwendung von neuronalen Netzwerken vermittelt. 
Bei Fragen oder Anregungen könnt ihr euch gerne an uns wenden. Viel Spaß beim Ausprobieren!

